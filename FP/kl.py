import jax
import jax.numpy as jnp
from .utils import jax_div


class KLDivergence():
    def __init__(self, dim, prior, target_p, target_log_p, sample_fn, total_evolve_time):
        assert(target_log_p is not None or sample_fn is not None)
        self.dim = dim
        self.prior = prior
        self.target_p = target_p
        self.target_log_p = target_log_p
        if target_log_p is not None:
            assert callable(target_log_p)
            self.target_grad_log_p = jax.grad(target_log_p)
        self.sample_fn = sample_fn
        self.total_evolve_time = total_evolve_time

    def get_dim(self):
        return self.dim
    
    def get_prior(self):
        return self.prior
    
    def get_total_evolve_time(self):
        return self.total_evolve_time
    
    def get_target_potential_gradient(self, x, beta=1.0):
        if x.ndim == 1:
            return jnp.matmul(self.target_p.cov_inv/beta, x - self.target_p.mean)
        else:
            v_matmul = jax.vmap(jnp.matmul, in_axes=(None, 0))
            return v_matmul(self.target_p.cov_inv/beta, x - self.target_p.mean)
    
    def get_v_goal(self, x, t, info):
        log_p_fn = info['log_p_fn']
        params = info['params']
        grad_log_p_fn = jax.grad(log_p_fn, argnums=2)
        return self.target_grad_log_p(x) - grad_log_p_fn(params, t, x)
    
    def get_v_goal_with_score(self, x, t, score, info):
        return self.target_grad_log_p(x) - score
    
    def get_v_dot_ibp(self, x, t, info):
        v_fn = info['v_fn']
        params = info['params']
        v = v_fn(params, t, x)
        div = jax_div(v_fn, argnums=2)(params, t, x)
        v_dot = div + (v * self.target_grad_log_p(x)).sum(-1)
        return v_dot
        