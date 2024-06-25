import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from tqdm import trange
from jax.tree_util import tree_map, tree_reduce, tree_unflatten
from jax.experimental.ode import odeint
from jax import grad
import wandb
from pathlib import Path
import pickle
from flax.training import checkpoints

from .train_state import TrainState
from .utils import jax_div
from utils.distribution import FuncDistribution
from utils.metrics import *


class SCVM():
    def __init__(self, config, net, kl, optimizer):
        self.config = config
        self.net = net
        self.kl = kl
        self.optimizer = optimizer
    
    def velocity_at(self, params, t, x):
        return self.net.apply({'params': params}, t, x)
    
    def get_init_args(self):
        return (0.0, jnp.zeros([self.kl.get_dim()]))
    
    def forward_multi_t(self, params, tpts, x0):
        assert x0.ndim == 1
        assert tpts.ndim == 1
        assert tpts.shape[0] > 1

        def v_fn(t, x):
            return self.velocity_at(params, t, x)
        
        ode_v_fn = lambda x, t: v_fn(t, x)
        # Note: odeint returns NaN if t[1] == t[0], etc. So we
        # use a hack here.
        tpts = jnp.concatenate([jnp.array([tpts[0]]), tpts[1:] + 1e-32])
        sol = odeint(ode_v_fn, x0, tpts, rtol=self.config['ode_rtol'], atol=self.config['ode_atol'])
        return sol
    
    def forward_multi_t_with_score(self, params, tpts, x0, score0):
        '''
        Args:
            params: model parameters
            tpts: (T,), an increasing sequence of time points
            x0: (D,), initial condition
            score0: (D,), score at x0

        Returns:
        
        '''
        assert x0.ndim == 1
        assert tpts.ndim == 1
        assert tpts.shape[0] > 1
        D = x0.shape[0]

        def v_fn(t, x):
            return self.velocity_at(params, t, x)
        
        tpts = jnp.concatenate([jnp.array([tpts[0]]), tpts[1:] + 1e-32])

        v_div = jax_div(v_fn, argnums=1)
        v_grad_div = jax.grad(v_div, argnums=1)

        def x_score_wrapper(x, t):
            x, score = x[:D], x[D:]
            fn = lambda x: (v_fn(t, x) * score).sum()
            score_dot = -v_grad_div(t, x) - jax.grad(fn)(x)

            return jnp.concatenate([v_fn(t, x), score_dot])
        
        sol = odeint(x_score_wrapper, jnp.concatenate([x0, score0]), 
                     tpts, rtol=self.config['ode_rtol'], atol=self.config['ode_atol'])
        return sol[:, :D], sol[:, D:]

    def create_train_state(self):
        rng = random.PRNGKey(self.config['seed'])
        rng, net_rng = random.split(rng)
        params = self.net.init(net_rng, *self.get_init_args())['params']
        return TrainState.create(rng=rng, params=params, optimizer=self.optimizer)

    @property
    def global_step(self):
        return self.state.step
    
    def _init_models(self):
        ''' 
        self.state:
            step, rng, optimizer, params, opt_state
        '''
        forward_vmap = jax.vmap(self.forward_multi_t, in_axes=(None, None, 0))
        self.forward_vmap = jax.jit(forward_vmap)

        v_vmap = jax.vmap(self.velocity_at, in_axes=(None, None, 0))
        v_vmap = jax.vmap(v_vmap, in_axes=(None, 0, 1), out_axes=1)
        self.v_vmap = jax.jit(v_vmap)

        info_in_axes = {'samples': 1, 'params': None}
        if self.config['use_ibp']:
            info_in_axes['v_fn'] = None
            self.v_dot_ibp_vmap = jax.vmap(jax.vmap(self.kl.get_v_dot_ibp, in_axes=(0, 0, info_in_axes)),
                                           in_axes=(0, None, None)
                                           )
        if self.config['ode_score']:
            self.v_goal_vmap = jax.vmap(jax.vmap(self.kl.get_v_goal_with_score, 
                                                 in_axes=(0, 0, 0, info_in_axes)), 
                                        in_axes=(0, None, 0, None))
            score_fn = self.forward_multi_t_with_score
            self.score_vmap = jax.vmap(score_fn, in_axes=(None, None, 0, 0))
        else:
            self.v_goal_vmap = jax.vmap(jax.vmap(self.kl.get_v_goal, 
                                                 in_axes=(0, 0, info_in_axes)), 
                                        in_axes=(0, None, None))

        self.state = self.create_train_state()
        self.load_checkpoint()
        self.train_step = self.create_train_step()
    
    def create_train_step(self):
        '''
        Create a train step function that takes in a batch of data and updates the model parameters.
        '''
        def loss_fn(params, last_params, batch, rng, iter):
            '''
            Get the loss
            args:
                params: model parameters
                batch: a batch of data
                rng: random number generator
                iter: current iteration number
            '''
            t, x0 = batch['t'], batch['x0']
            xt = self.forward_vmap(last_params, t, x0) # (B, T, D)
            xt = jax.lax.stop_gradient(xt)

            vt = self.v_vmap(params, t, xt) # (B, T, D)
            info = {
                'params': params,
                'samples': xt, # (B, T, D)
                'v_fn': self.velocity_at
            }
            v_dot = self.v_dot_ibp_vmap(xt, t, info) # (B, T)
            loss = (vt * vt).sum(-1) - 2 * v_dot    # (B, T)
            loss = loss.mean()

            return loss


        def train_step(state, rng, iter):
            last_params = state.params
            rng, j_rng, loss_rng = jax.random.split(rng, 3)
            batch = self.sample_batch(state, j_rng)
            params = state.params
            loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params, last_params, batch, loss_rng, iter)
            state = state.apply_grad(grads=grads)

            diff_tree = tree_map(lambda p1, p2: ((p1-p2)**2).sum(), last_params, state.params)
            params_diff_norm = tree_reduce(lambda p, s: p+s, diff_tree, 0)
            loss_dict = {'loss': loss, 'params_diff_norm': params_diff_norm}

            return state, loss_dict
        
        train_step = jax.jit(train_step)
        return train_step
    
    def sample_batch(self, state, rng, rescale=False):
        B = self.config['batch_size']
        T = self.config['num_tpts']
        total_time = self.kl.total_evolve_time

        rng, t_rng, x0_rng = jax.random.split(rng, 3)
        
        th = total_time / T
        anchor = jnp.arange(T) * th
        t = anchor + random.uniform(t_rng, (T,))
        if rescale:
            t = t * total_time
        # Note: jax's odeint has bug if the starting time is not zero.
        t = jnp.concatenate([jnp.zeros([1]), t])

        prior = self.kl.get_prior()
        x0 = prior.sample(batch_size=B, rng=x0_rng)
        if x0.ndim == 1:
            x0 = x0[:,None]

        return {
            't': t, 
            'x0': x0,
        }
    
    def _train(self):
        train_range = trange(self.global_step, self.config['num_train_steps'])
        for iter in train_range:
            rng, i_rng = random.split(self.state.rng)
            self.state, loss_dict = self.train_step(self.state, i_rng, iter)
            if self.global_step % self.config['save_freq'] == 0:
                self.save_checkpoint()
            if self.global_step % self.config['val_freq'] == 0:
                self._validate(all_val=False)
            train_range.set_description(self.log_loss(loss_dict))
    
    def _get_ckpt_dir(self):
        return self.config['ckpt_name']
    
    def load_checkpoint(self):
        ckpt_dir = self._get_ckpt_dir()
        print(f'checkpoint path: {ckpt_dir}')
        if Path(ckpt_dir).exists():
            self.state = checkpoints.restore_checkpoint(
                self._get_ckpt_dir(),
                target=self.state)
            print('Restoring checkpoint at {}...'.format(ckpt_dir))
    
    def save_checkpoint(self):
        checkpoints.save_checkpoint(
            self._get_ckpt_dir(),
            target=self.state,
            step=self.state.step,
            overwrite=True,
            keep=1,
            )

    def log_loss(self, loss_dict):
        loss = loss_dict['loss']
        param_diff = loss_dict['params_diff_norm']
        log_dict = {'Flow Loss': loss, 'Param Diff': param_diff}
        wandb.log(log_dict, step=self.global_step)
        return f'Step {self.global_step} | Loss: {loss} | Param Diff: {param_diff}'
    
    def run(self):
        self._init_models()
        if self.config['is_val']:
            self._validate()
        else:
            self._train()
    
    def _validate(self, all_val=True):
        if all_val:
            times = np.linspace(0.0, self.kl.total_evolve_time, self.config['num_val_time']['val'])
        else:
            times = np.linspace(0.0, self.kl.total_evolve_time, self.config['num_val_time']['train'])
        data_metric = {metric: [] for metric in self.config['eval_metrics']}
        data = []
        for i in trange(0, times.shape[0]):
            t = times[i]
            pt = self.eval_density(t)
            cur = []
            for metric in self.config['eval_metrics']:
                s = 0.0
                raw = []
                for seed in range(999, 999+self.config['metric_repeat']):
                    m = compute_metric(pt, self.kl.target_p, metric=metric,
                                       num_sample=self.config['num_val_sample'], seed=seed)
                    raw.append(m)
                    s+=m
                data_metric[metric].append(raw)
                cur.append(s/self.config['metric_repeat'])
            data.append([t, *cur])
        
        if not all_val:
            table = wandb.Table(data=data, columns=['t', *self.config['eval_metrics']])
            step = self.global_step
            for k, metric in enumerate(self.config['eval_metrics']):
                wandb.log({f'{metric}_plot': wandb.plot.line(table, 't', metric, title=f'{metric} vs. Time'),
                            f'final_{metric}': data[-1][k+1],
                            }, step=step)
        else:
            # save the trajectories and metrics
            save_metrics = {'times': times}
            for metric in self.config['eval_metrics']:
                save_metrics[metric] = np.array(data_metric[metric])  # (len(times), 5)
                
            with open('results/metrics_{:}.pickle'.format(self.config['chaotic_sys']), 'wb') as file:
                pickle.dump(save_metrics, file)

            prior = self.kl.get_prior()
            B = self.config['num_cluster'] * self.config['traj_per_cluster']
            x0 = prior.sample(batch_size=B, concat=False)  # list of (traj_per_cluster, D)
            x0 = jnp.array(x0)  # (num_cluster, traj_per_cluster, D)
            x0 = jnp.concatenate(x0, axis=0)  # (num_cluster*traj_per_cluster, D)
            trajs = []
            for j in range(times.shape[0]):
                t = jnp.concatenate([jnp.zeros([1]), jnp.array([times[j]])])
                xt = self.forward_vmap(self.state.params, t, x0) # (B, 2, D)
                trajs.append(xt[:,-1,:]) 
            trajs = jnp.array(trajs)  # (num_time, B, D)

            with open('results/scvm_trajs_{:}.pickle'.format(self.config['chaotic_sys']), 'wb') as file:
                pickle.dump(trajs, file)
            

            

    def eval_density(self, t):
        '''evaluate the density of the learned model at time t'''
        def sample_fn(rng, batch_size):
            x0 = self.kl.get_prior().sample(batch_size=batch_size, rng=rng)
            if x0.ndim == 1:
                x0 = x0[:,None]
            return self.forward_vmap(self.state.params,
                                     jnp.array([0, t]),
                                     x0)[:,-1,:]    # (B, D)
        return FuncDistribution(sample_fn, log_p_batch_fn=None)
