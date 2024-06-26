import jax 
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from functools import partial
import ot
import numpy as np
import warnings


def kaplan_yorke_dimension(spectrum0):
    """Calculate the Kaplan-Yorke dimension, given a list of Lyapunov exponents"""
    spectrum = np.sort(spectrum0)[::-1]
    d = len(spectrum)
    cspec = np.cumsum(spectrum)
    j = np.max(np.where(cspec >= 0))
    if j > d - 2:
        j = d - 2
        warnings.warn("Cumulative sum of Lyapunov exponents never crosses zero. System may be ill-posed or undersampled.")

    dky = 1 + j + cspec[j] / np.abs(spectrum[j + 1])

    return dky


def exp_diff_norm(X, Y):
    '''
    Compute E[||X-Y||^2] given samples X, Y.
    Args:
      X: (B, D)
      Y: (B', D)
    '''
    norm_fn = lambda x, y: ((x-y) ** 2).sum()
    norm_vmap = jax.vmap(jax.vmap(norm_fn, in_axes=(0, None)),
                         in_axes=(None, 0), out_axes=1)
    return norm_vmap(X, Y).sum() / (X.shape[0] * Y.shape[0])


@partial(jax.jit, static_argnames=['dist', 'bandwidth'])
def compute_ksd(X, dist, bandwidth=1.0):
    """
    Compute kernelized Stein discrepancy
    Args:
        X: (B, D)
        score_fn (D,) -> (D,)
    """
    # get score function
    score_fn = jax.grad(dist.log_p)

    def kernel_fn(x, y):
        """
        x, y : (D,)
        """
        return jnp.exp(-((x-y)**2).sum() / (2*bandwidth**2))
    # get gradient of kernel 
    dkdx = jax.jacfwd(kernel_fn, argnums=0)
    dkdy = jax.jacfwd(kernel_fn, argnums=1)
    d2k = jax.jacrev(dkdy, argnums=0)

    def u_fn(x, y):
        """
        x, y : (D,)
        """
        score_x = score_fn(x)
        score_y = score_fn(y)
        tmp = (score_x * score_y).sum(-1) * kernel_fn(x, y)
        tmp += (score_x * dkdy(x, y)).sum(-1)
        tmp += (score_y * dkdx(x, y)).sum(-1)
        tmp += jnp.trace(d2k(x, y))
        return tmp
    
    B = X.shape[0]
    u_vmap = jax.vmap(jax.vmap(u_fn, in_axes=(None, 0)), in_axes=(0, None))
    u_all = u_vmap(X, X) # (B, B)
    mask = 1 - jnp.eye(B)

    return (u_all * mask).sum() / (B * (B - 1))


def compute_ot(samples1, samples2, num_sample):
    weights1 = np.ones(num_sample) / num_sample
    weights2 = np.ones(num_sample) / num_sample
    M = ot.dist(np.array(samples1), np.array(samples2))
    W = ot.emd2(weights1, weights2, M)
    return W


def compute_w2_gauss(mean1, cov1, mean2, cov2):
    tmp1 = ((mean1 - mean2) ** 2).sum()
    tmp2 = (cov1 @ cov2)
    tmp2 = scipy.linalg.sqrtm(tmp2)
    tmp2 = np.real(tmp2)
    tmp2 = np.trace(cov1 + cov2 - 2 * tmp2)
    dist = tmp1 + tmp2
    return dist


def compute_metric(dist1, dist2, *, metric, num_sample, 
                   samples1=None, samples2=None, seed=126):
    """
    Args:
        dist1, dist2: instances of distribution
        metric: [ 'sym_kl', 'ed', 'sym_pos_kl', 'ksd', 'ot' ]
    """
    rng = jax.random.PRNGKey(seed)

    if metric == 'sym_kl':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)
        log_p_11 = dist1.log_p_batch(samples1)
        log_p_21 = dist1.log_p_batch(samples2)
        log_p_12 = dist2.log_p_batch(samples1)
        log_p_22 = dist2.log_p_batch(samples2)
        kl_12 = (log_p_11 - log_p_12).mean()
        kl_21 = (log_p_22 - log_p_21).mean()
        return kl_12 + kl_21
    elif metric == 'sym_pos_kl':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)
        log_p_11 = dist1.log_p_batch(samples1)
        log_p_21 = dist1.log_p_batch(samples2)
        log_p_12 = dist2.log_p_batch(samples1)
        log_p_22 = dist2.log_p_batch(samples2)
        kl_12 = (log_p_11 - log_p_12).mean()
        kl_21 = (log_p_22 - log_p_21).mean()
        return 0.5 * ((log_p_11 - log_p_12) ** 2 + (log_p_21 - log_p_22) ** 2).mean()
    elif metric == 'ksd':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        return compute_ksd(samples1, dist2)
    elif metric == 'ot':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)

        import ot
        weights1 = np.ones(num_sample) / num_sample
        weights2 = np.ones(num_sample) / num_sample
        M = ot.dist(np.array(samples1), np.array(samples2))
        W = ot.emd2(weights1, weights2, M)
        return W
    elif metric == 'w2_gauss':
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None:
            samples1 = dist1.sample(rng1, num_sample)
        mean1 = jnp.mean(samples1, axis=0)
        cov1 = jnp.cov(samples1, rowvar=False)
        if samples2 is None:
            from distribution import Gaussian
            assert(isinstance(dist2, Gaussian))
            mean2 = dist2.mean
            cov2 = dist2.get_cov()
        else:
            mean2 = jnp.mean(samples2, axis=0)
            cov2 = jnp.cov(samples2, rowvar=False)
        return compute_w2_gauss(mean1, cov1, mean2, cov2)
    elif metric == 'kl':
        if samples1 is None :
            samples1 = dist1.sample(rng, num_sample)
        log_p1 = dist1.log_p_batch(samples1)
        log_p2 = dist2.log_p_batch(samples1)
        return (log_p1 - log_p2).mean()
    else :
        assert(metric == 'ed')
        rng1, rng2 = jax.random.split(rng)
        if samples1 is None :
            samples1 = dist1.sample(rng1, num_sample)
        if samples2 is None :
            samples2 = dist2.sample(rng2, num_sample)
        SS = exp_diff_norm(samples1, samples1)
        ST = exp_diff_norm(samples1, samples2)
        TT = exp_diff_norm(samples2, samples2)
        return (2 * ST - SS - TT)


