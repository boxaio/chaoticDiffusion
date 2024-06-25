import jax
import jax.numpy as jnp


def jax_div(func, argnums):
    '''
    divergence operator
    args:
        func: (..., D, ...) -> (D,)
    returns: 
        divergence of func
        (..., D, ...) -> ()
    '''
    jac = jax.jacfwd(func, argnums=argnums)
    diver_fn = lambda *a, **kw: jnp.trace(jac(*a, **kw))
    return diver_fn


def _divergence_fn(f, _x, _v):
    # Hutchinson’s Estimator
    # computes the divergence of net at x with random vector v
    _, u = jax.jvp(f, (_x,), (_v,))
    # print(u.shape, _x.shape, _v.shape)
    return jnp.sum(u * _v)


# f_list = [lambda x: f(x)[i]]

def _divergence_bf_fn(f, _x):
    # brute-force implementation of the divergence operator
    # _x should be a d-dimensional vector
    jacobian = jax.jacfwd(f)
    a = jacobian(_x)
    return jnp.sum(jnp.diag(a))



batch_div_bf_fn = jax.vmap(_divergence_bf_fn, in_axes=[None, 0])

batch_div_fn = jax.vmap(_divergence_fn, in_axes=[None, None, 0])


def divergence_fn(f, _x, _v=None):
    if _v is None:
        return batch_div_bf_fn(f, _x)
    else:
        return batch_div_fn(f, _x, _v).mean(axis=0)