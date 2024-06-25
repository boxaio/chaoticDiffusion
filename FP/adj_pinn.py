import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import matplotlib.pyplot as plt
import optax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_reduce
import jax.random as random
from flax import serialization
import flax.linen as nn
from tqdm import trange
import jax
import wandb
from jax import jvp, grad, value_and_grad, vjp
from jax.experimental.ode import odeint
from pathlib import Path
import pickle
from flax.training import checkpoints

from utils.distribution import FuncDistribution
from .train_state import TrainState
from .utils import jax_div, divergence_fn
from utils.metrics import *


class ADJ():
    def __init__(self, config, net, kl, optimizer):
        self.config = config
        self.net = net
        self.kl = kl
        self.optimizer = optimizer

    def get_init_args(self):
        return (0.0, jnp.zeros([self.kl.get_dim()]))
    
    def forward_multi_t(self, params, tpts, x0):
        assert x0.ndim == 1
        assert tpts.ndim == 1
        assert tpts.shape[0] > 1

        def v_fn(t, x):
            return self.net.apply({'params': params}, t, x) \
                                     - self.kl.get_target_potential_gradient(x, self.config['diffusion_coeff'])
        
        ode_v_fn = lambda x, t: v_fn(t, x)
        # Note: odeint returns NaN if t[1] == t[0], etc. So we
        # use a hack here.
        tpts = jnp.concatenate([jnp.array([tpts[0]]), tpts[1:] + 1e-32])
        sol = odeint(ode_v_fn, x0, tpts, rtol=self.config['ode_rtol'], atol=self.config['ode_atol'])
        return sol
    
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

        self.state = self.create_train_state()
        # self.load_checkpoint()
        self.train_step = self.create_train_step()

    def create_train_step(self):
        '''
        Create a train step function that takes in a batch of data and updates the model parameters.
        '''
        params = self.state.params
        params_flat, params_tree = tree_flatten(params)
        # (params, (), (D,)) -> (D,)
        vn = lambda params, t, x: self.net.apply({'params': params}, t, x) 
        v_fn = lambda params, t, x: self.net.apply({'params': params}, t, x) \
                                     - self.kl.get_target_potential_gradient(x, self.config['diffusion_coeff'])
        # v_div = jax_div(v_fn, 2)    # (params, (), (D,)) -> ()
        # v_grad_div = jax.grad(v_div, 2)    # (params, (), (D,)) -> (D,)

        def loss_and_grad_fn(params, batch, rng, iter):
            '''
             compute x(T) by solve IVP (I) & compute the loss
            '''
            # ================ Forward ===================
            x_0 = batch
            prior = self.kl.get_prior()
            score_0 = prior.score(x_0)  # score, s
            loss_0 = jnp.zeros(1)
            states_0 = [x_0, score_0, loss_0]

            def ode_forward(states, t):
                x, score = states[0], states[1]
                v_at_x = lambda x: v_fn(params, t, x) # (D,) -> (D,)
                vn_at_x = lambda x: vn(params, t, x) # (D,) -> (D,)
                dx = v_fn(params, t, x)   # (B, D)

                def score_dt(score, x):
                    div_v_at_x = lambda x: divergence_fn(v_at_x, x).sum(axis=0)
                    grad_div_v = grad(div_v_at_x)
                    _, vjp_v = vjp(v_at_x, x)
                    return -grad_div_v(x) - vjp_v(score)[0]
                
                dscore = score_dt(score, x)  # (B, D)

                def g_t(score, x):
                    v = vn_at_x(x)
                    reg = self.config['reg_coeff'] * jnp.sum(v**2, axis=(1,))
                    return jnp.mean(jnp.sum((v + self.config['diffusion_coeff'] * score)**2, axis=(1,)) + reg)
                
                dloss = g_t(score, x)  
                return [dx, dscore, dloss]
            
            ts = jnp.array((0., self.kl.get_total_evolve_time()))
            sol_forward = odeint(ode_forward, states_0, ts, atol=self.config['ode_atol'], rtol=self.config['ode_rtol'])
            x_T = sol_forward[0][1]     # (B, D)
            score_T = sol_forward[1][1]    # (B, D)
            loss_f = sol_forward[2][1]  # ()

            # ================ Backward ==================
            ''' compute dl/d_theta via adjoint method '''
            a_T = jnp.zeros_like(x_T)
            b_T = jnp.zeros_like(x_T)
            grad_T = [jnp.zeros_like(_var) for _var in params_flat]
            loss_T = jnp.zeros(1)
            states_T = [x_T, a_T, b_T, score_T, loss_T, grad_T]

            def ode_backward(states, t):
                t = self.kl.get_total_evolve_time() - t
                x, a_x, a_s, score = states[0], states[1], states[2], states[3]

                vn_at_t = lambda _params, _x: vn(_params, t, _x) 
                v_fn_at_t = lambda _params, _x: v_fn(_params, t, _x)
                dx = v_fn_at_t(params, x)  # (B, D)

                _, vjp_fx_fn = vjp(lambda _x: v_fn_at_t(params, _x), x)
                vjp_fx_ax = vjp_fx_fn(a_x)[0]   # (B, D)
                _, vjp_ftheta_fn = vjp(lambda _params: v_fn_at_t(_params, x), params)
                vjp_ftheta_ax = vjp_ftheta_fn(a_x)[0]  

                def score_dt(score, x, theta):
                    v_fn_at_x = lambda _x: v_fn_at_t(theta, _x)
                    div_v = lambda _x: divergence_fn(v_fn_at_x, _x).sum(axis=0)
                    grad_div_v = grad(div_v)
                    _, vjp_v = vjp(v_fn_at_x, x)
                    return -grad_div_v(x) - vjp_v(score)[0]
                
                _, vjp_fscore_fn = vjp(lambda _score: score_dt(_score, x, params), score)
                vjp_fscore_as = vjp_fscore_fn(a_s)[0]   # (B, D)
                _, vjp_fx_fn = vjp(lambda _x: score_dt(score, _x, params), x)
                vjp_fx_as = vjp_fx_fn(a_s)[0]   # (B, D)
                _, vjp_ftheta_fn = vjp(lambda _params: score_dt(score, x, _params), params)
                vjp_ftheta_as = vjp_ftheta_fn(a_s)[0]  

                def g_t(score, x, theta):
                    v = vn_at_t(theta, x)
                    reg = self.config['reg_coeff'] * jnp.sum(v**2, axis=(1,))
                    return jnp.mean(jnp.sum((v + self.config['diffusion_coeff'] * score)**2, axis=(1,)) + reg)
                
                dg_ds = jax.grad(g_t, argnums=0)
                dg_dx = jax.grad(g_t, argnums=1)
                dg_dtheta = jax.grad(g_t, argnums=2)

                dax = -vjp_fx_ax - vjp_fx_as - dg_dx(score, x, params)
                das = -vjp_fscore_as - dg_ds(score, x, params)
                dscore = score_dt(score, x, params)
                dloss = g_t(score, x, params)[None]

                vjp_ftheta_ax_flat, _ = tree_flatten(vjp_ftheta_ax)
                vjp_ftheta_as_flat, _ = tree_flatten(vjp_ftheta_as)
                dg_dtheta_flat, _ = tree_flatten(dg_dtheta(score, x, params))
                dgrad = [_dgrad1/x.shape[0] + _dgrad2/x.shape[0] + _dgrad3 for _dgrad1, _dgrad2, _dgrad3 in 
                         zip(vjp_ftheta_ax_flat, vjp_ftheta_as_flat, dg_dtheta_flat)]

                return [-dx, -dax, -das, -dscore, dloss, dgrad]


            ts = jnp.array((0., self.kl.get_total_evolve_time()))
            sol_backward = odeint(ode_backward, states_T, ts, atol=self.config['ode_atol'], rtol=self.config['ode_rtol'])
            grad_T = tree_unflatten(params_tree, [_var[1] for _var in sol_backward[5]])
            x_0_b = sol_backward[0][1]
            score_0_b = sol_backward[3][1]
            err_x = jnp.mean(jnp.sum((x_0_b - x_0).reshape(1,-1)**2, axis=(1,)))
            err_score = jnp.mean(jnp.sum((score_0_b - score_0).reshape(1,-1)**2, axis=(1,)))
            loss_b = sol_backward[4][1]

            return loss_f, loss_b, err_x, err_score, grad_T
        
        def train_step(state, rng, iter):
            rng, j_rng, loss_rng = jax.random.split(rng, 3)
            batch = self.sample_batch(state, j_rng)
            last_params = state.params
            loss_f, loss_b, err_x, err_score, grad_T = loss_and_grad_fn(last_params, batch, loss_rng, iter)
            state = state.apply_grad(grads=grad_T)

            diff_tree = tree_map(lambda p1, p2: ((p1-p2)**2).sum(), last_params, state.params)
            params_diff_norm = tree_reduce(lambda p, s: p+s, diff_tree, 0)

            return state, loss_b, loss_f, err_x, err_score, params_diff_norm
        
        return jax.jit(train_step)

    def sample_batch(self, state, rng):
        B = self.config['batch_size']
        rng, x0_rng = jax.random.split(rng, 2)
        prior = self.kl.get_prior()
        x0 = prior.sample(batch_size=B, rng=x0_rng)
        if x0.ndim == 1:
            x0 = x0[:,None]
        return x0
    
    def _train(self):
        train_range = trange(self.global_step, self.config['num_train_steps'])
        for iter in train_range:
            rng, i_rng = random.split(self.state.rng)
            self.state, loss_b, loss_f, err_x, err_score, params_diff_norm = self.train_step(self.state, i_rng, iter)
            loss_dict = {'loss_f': loss_f, 'loss_b': loss_b, 'err_x': err_x, 'err_score': err_score, 
                         'params_diff_norm': params_diff_norm}
            if self.global_step % self.config['save_freq'] == 0:
                self.save_checkpoint()
            if self.global_step % self.config['val_freq'] == 0:
                self._validate(all_val=False)
            train_range.set_description(self.log_loss(loss_dict))
            # if params_diff_norm < 6e-8:
            #     break
    
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
        loss_f = loss_dict['loss_f']
        loss_b = loss_dict['loss_b']
        err_x = loss_dict['err_x']
        err_score = loss_dict['err_score']
        params_diff_norm = loss_dict['params_diff_norm']
        log_dict = {'Loss forward': loss_f, 
                    'Loss backward': loss_b,
                    'Error of x': err_x,
                    'Error of score': err_score,
                    'Param Diff': params_diff_norm}
        wandb.log(log_dict, step=self.global_step)
        return f'Step {self.global_step} | Loss forward: {loss_f} | Loss backward: {loss_b} | Error of x: {err_x} | Error of score: {err_score} | Param Diff: {params_diff_norm}'
    
    def run(self):
        self._init_models()
        if self.config['is_val']:
            self.load_checkpoint()
            self._validate()
        else:
            self._train()
    
    def _validate(self, all_val=True):
        if all_val:
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
                
            with open(os.getcwd()+'/results/FP_ADJ_metrics_{:}.pickle'.format(self.config['name']), 'wb') as file:
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

            with open(os.getcwd()+'/results/FP_ADJ_trajs_{:}.pickle'.format(self.config['name']), 'wb') as file:
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
