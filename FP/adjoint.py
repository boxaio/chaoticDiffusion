import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from tqdm import trange
from jax.tree_util import tree_map, tree_reduce, tree_unflatten
from jax.experimental.ode import odeint
from jax import grad
import wandb
import os
from pathlib import Path
import pickle
from flax.training import checkpoints

from .utils import jax_div
from utils.distribution import FuncDistribution
from .train_state import TrainState
from utils.metrics import *


class AdjointSolver():
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
        self.state = self.create_train_state()
        self.load_checkpoint()
        self.train_step = self.create_train_step()
    
    def create_train_step(self):
        prior = self.kl.get_prior()
        # v_fn = lambda params, t, x: self.net.apply({'params': params}, t, x)
        v_div = jax_div(self.velocity_at, argnums=2)
        grad_v_div = jax.grad(v_div, argnums=2)

        D = self.kl.get_dim()
        total_time = self.kl.get_total_evolve_time()

        def psi_fn(params, s, t):
            x, xi = s[:D], s[D:]
            vx = lambda x: self.velocity_at(params, t, x)
            if self.config['use_vjp']:
                f = lambda x: (xi * self.velocity_at(params, t, x)).sum()
                xi_dt = -jax.grad(f)(x) - grad_v_div(params, t, x) 
            else:
                _, jvp = jax.jvp(vx, (x,), (xi,))
                xi_dt = -grad_v_div(params, t, x) - jvp
            return jnp.concatenate([self.velocity_at(params, t, x), xi_dt])
        
        def g_fn(params, s, t):
            x, xi = s[:D], s[D:]
            v_target = self.kl.get_v_goal_with_score(x=x, t=t, score=xi, info=None)
            v_pred = self.velocity_at(params, t, x)
            return ((v_target - v_pred)**2).sum()
        
        def forward_fn(params, x0, xi0):
            s0 = jnp.concatenate([x0, xi0])
            ts = jnp.array([0., total_time])
            psi_dt = lambda s, t: psi_fn(params, s, t)
            sol = odeint(psi_dt, s0, ts, rtol=self.config['ode_rtol'], atol=self.config['ode_atol'])  # (2, 2*D)
            return sol[1]
        
        def backward_fn(params, sT, ts):
            '''
            ts the increasing time points
            '''
            # assert jnp.all(ts[1:] >= ts[:-1])
            def dstate_dt(state, t):
                s, a = state[:2*D], state[2*D:]
                ds_dt = psi_fn(params, s, t)
                dg_ds = jax.grad(g_fn, argnums=1)(params, s, t)
                fn = lambda s: psi_fn(params, s, t)
                if self.config['use_vjp']:
                    fn = lambda s: (a * psi_fn(params, s, t)).sum()
                    jvp = jax.grad(fn)(s)
                else:
                    _, jvp = jax.jvp(fn, (s,), (a,))

                return jnp.concatenate([ds_dt, -dg_ds-jvp])
            
            aT = jnp.zeros([2*D])
            stateT = jnp.concatenate([sT, aT])
            # solve backward in time
            ts = jnp.concatenate([jnp.zeros(1), total_time-jnp.flip(ts)])
            sol = odeint(lambda state, t: -dstate_dt(state, t), 
                         stateT, ts, rtol=self.config['ode_rtol'], atol=self.config['ode_atol'])
            
            return (jnp.flip(sol[1:, :2*D], 0),
                    jnp.flip(sol[1:, 2*D:], 0))
        
        
        self.forward_vmap = jax.vmap(forward_fn, in_axes=(None, 0, 0))
        self.backward_vmap = jax.vmap(backward_fn, in_axes=(None, 0, None))
            
        def loss_fn(params, x0, xi0, ts):
            sT = self.forward_vmap(params, x0, xi0)   # (B, 2*D)
            s_ts, a_ts = self.backward_vmap(params, sT, ts)  # (B, T, 2*D)
            s_ts = jax.lax.stop_gradient(s_ts)
            a_ts = jax.lax.stop_gradient(a_ts)
            
            psi_vmap = jax.vmap(jax.vmap(psi_fn, in_axes=(None, 0, 0)), in_axes=(None, 0, None))
            g_vmap = jax.vmap(jax.vmap(g_fn, in_axes=(None, 0, 0)), in_axes=(None, 0, None))
            loss = (a_ts * psi_vmap(params, s_ts, ts)).sum(-1)  # (B, T)
            loss += g_vmap(params, s_ts, ts)    # (B, T)
            return loss.mean()
        
        def train_step(state, rng):
            rng, j_rng = jax.random.split(rng, 2)
            batch = self.sample_batch(state, j_rng)
            x0 = batch['x0']
            ts = batch['t']
            xi0 = jax.vmap(jax.grad(prior.log_p))(x0)  # (B, D)
            params = state.params
            loss, grads = jax.value_and_grad(loss_fn, argnums=0)(params, x0, xi0, ts)
            state = state.apply_grad(grads=grads)
            return state, loss
        
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
            self.state, loss_dict = self.train_step(self.state, i_rng)
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
        
    def log_loss(self, loss):
        wandb.log({'Adjoint Loss': loss}, step=self.global_step)
        return f'Step {self.global_step} | Loss: {loss}'
    
    def run(self):
        self._init_models()
        if self.config['is_val']:
            self._validate()
        else:
            self._train()

    def _validate(self, all_val=True):
        if all_val:
            # scaled times
            times = np.linspace(0.0, 1.0, self.config['num_val_time']['val'])
        else:
            times = np.linspace(0.0, 1.0, self.config['num_val_time']['train'])
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
            data.append([t*self.kl.total_evolve_time, *cur])
        
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
                
            with open(os.getcwd()+'/results/metrics_{:}.pickle'.format(self.config['name']), 'wb') as file:
                pickle.dump(save_metrics, file)

            prior = self.kl.get_prior()
            B = self.config['num_cluster'] * self.config['traj_per_cluster']
            x0 = prior.sample(batch_size=B)   
            x0 = jnp.array(x0)  # (B, D)
            
            trajs = []
            nn_trajs = dict()
            for j in range(times.shape[0]):
                t = jnp.concatenate([jnp.zeros([1]), jnp.array([times[j]])])
                xt = self.forward_vmap(self.state.params, t, x0) # (B, 2, D)
                trajs.append(xt[:,-1,:]) 
            nn_trajs['trajs'] = jnp.array(trajs)  # (num_time, B, D)

            x0 = prior.sample(batch_size=8000)   
            x0 = jnp.array(x0)  
            xt_T = self.forward_vmap(self.state.params, jnp.concatenate([jnp.zeros([1]), jnp.array([times[-1]])]), x0)  
            xT = xt_T[:,-1,:]
            nn_trajs['samples_T'] = jnp.array(xT)

            with open(os.getcwd()+'/results/adj_trajs_{:}.pickle'.format(self.config['name']), 'wb') as file:
                pickle.dump(nn_trajs, file)

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
