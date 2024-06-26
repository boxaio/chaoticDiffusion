from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
import math
from jax import random
from functools import partial
from collections import namedtuple
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC
import sklearn.datasets as skd
from einops import rearrange


class Distribution(ABC):
    @abstractmethod
    def sample(self, rng, batch_size):
        '''
        Args:
          rng: Jax's rng.
        Returns:
          (B, D), samples
        '''
        pass

    def log_p(self, X):
        '''
        Args:
          X: (D,), a single point.
        '''
        pass

    def p(self, X):
        '''
        Args:
          X: (D,), a single point.
        '''
        pass

    def log_p_batch(self, X):
        '''
        Args:
          X: (B, D), a batch of points.
        '''
        return jax.vmap(lambda X: self.log_p(X))(X)

    def p_batch(self, X):
        '''
        Args:
          X: (B, D), a batch of points.
        '''
        return jax.vmap(lambda X: self.p(X))(X)


class FuncDistribution(Distribution):
    def __init__(self, sample_fn, log_p_fn=None, log_p_batch_fn=None):
        self.sample_fn = sample_fn
        self.log_p_fn = log_p_fn
        self.log_p_batch_fn = log_p_batch_fn

    def sample(self, rng, batch_size):
        return self.sample_fn(rng, batch_size)

    def p(self, X):
        if self.log_p_fn is None :
            return None
        return jnp.exp(self.log_p_fn(X))

    def log_p(self, X):
        if self.log_p_fn is None:
            return None
        return self.log_p_fn(X)

    def p_batch(self, X):
        # p_batch_fn can contain jitted code.
        if self.log_p_batch_fn is not None:
            return jnp.exp(self.log_p_batch(X))
        if self.log_p_fn is None:
            return None
        return super().p_batch(X)

    def log_p_batch(self, X):
        # log_p_batch_fn can contain jitted code.
        if self.log_p_batch_fn is not None:
            return self.log_p_batch_fn(X)
        if self.log_p_fn is None:
            return None
        return super().log_p_batch(X)
    

def gaussian_unnormalized_log_p(X, mean, cov_inv):
    X_centered = X - mean
    tmp = -0.5 * jnp.matmul(jnp.expand_dims(X_centered, -2),
                            jnp.matmul(cov_inv, jnp.expand_dims(X_centered, -1)))
    tmp = jnp.squeeze(tmp, (-2, -1))
    return tmp

def gaussian_log_Z(cov_sqrt):
    dim = cov_sqrt.shape[-1]
    log_Z = (-dim/2 * np.log(2 * np.pi) - np.linalg.slogdet(cov_sqrt)[1])
    return log_Z

def gaussian_sample(rng, batch_size, mean, cov_sqrt):
    dim = cov_sqrt.shape[-1]
    Z = jax.random.normal(rng, (batch_size, dim))
    X = jnp.squeeze(jnp.expand_dims(cov_sqrt, 0) @ jnp.expand_dims(Z, -1), -1)
    return X + mean


class Gaussian(Distribution):
    def __init__(self, mean, cov_sqrt):
        '''
        Args:
          mean: (D,)
          cov_sqrt: (D, D), actual covariance is (cov_sqrt @ cov_sqrt^T).
        '''
        self.dim = mean.shape[0]
        self.mean = mean
        self.cov_sqrt = cov_sqrt

        self.log_Z = gaussian_log_Z(self.cov_sqrt)
        self.cov_inv = jnp.linalg.inv(self.cov_sqrt @ self.cov_sqrt.T)

    def sample(self, rng, batch_size):
        return gaussian_sample(rng, batch_size, self.mean, self.cov_sqrt)

    def log_p(self, X):
        return gaussian_unnormalized_log_p(X, self.mean, self.cov_inv) + self.log_Z

    def get_cov(self):
        return self.cov_sqrt @ self.cov_sqrt.T
    
    def score(self, x: jnp.ndarray):
        if self.cov_inv.shape[0]==1:
            return (self.mean - x)/self.get_cov()
        else:
            v_matmul = jax.vmap(jnp.matmul, in_axes=(None, 0))
            return v_matmul(self.cov_inv, self.mean - x)



# useful implementation for verbose
MHState = namedtuple("MHState", ["u", "rng_key"])
class MetropolisHastings(numpyro.infer.mcmc.MCMCKernel):
    sample_field = "u"
    def __init__(self, potential_fn, step_size=0.1):
        self.potential_fn = potential_fn
        self.step_size = step_size

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        return MHState(init_params, rng_key)

    def sample(self, state, model_args, model_kwargs):
        u, rng_key = state
        rng_key, key_proposal, key_accept = random.split(rng_key, 3)
        u_proposal = dist.Normal(u, self.step_size).sample(key_proposal)
        accept_prob = jnp.exp(self.potential_fn(u_proposal)-self.potential_fn(u))
        u_new = jnp.where(dist.Uniform().sample(key_accept) < accept_prob, u_proposal, u)
        return MHState(u_new, rng_key)


class Multimodal(Distribution):
    def __init__(self, seed, locs, scales):
        self.dim = 1
        self.seed = seed
        locs = jnp.array(locs)
        scales = jnp.array(scales)
        assert locs.ndim==1 and scales.ndim==1 and locs.shape[0]==scales.shape[0]

        num_mode = locs.shape[0]
        mixing_dist = dist.Categorical(probs=jnp.ones(num_mode) / num_mode)
        component_dist = dist.Normal(loc=locs, scale=scales)
        self.mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
    
    def sample(self, batch_size, rng=None):
        if rng is None:
            samples = self.mixture.sample(random.PRNGKey(self.seed), (batch_size,))
        else:
            samples = self.mixture.sample(rng, (batch_size,))
        return samples


class Uniform(Distribution):
    def __init__(self, seed, low:float, high:float):
        self.seed = seed
        self.dim = 1
        self.low = low
        self.high = high
    
    def sample(self, batch_size, rng=None):
        if rng is None:
            u0 = random.uniform(key=random.PRNGKey(self.seed), shape=(batch_size,))
        else:
            u0 = random.uniform(key=rng, shape=(batch_size,))
        u = (u0 + jnp.sqrt(2.0)*jnp.arange(batch_size)) % 1
        return u*(self.high-self.low) + self.low


class MultiUniform(Distribution):
    def __init__(self, seed, n, low, high):
        self.seed = seed
        self.dim = n
        
        if isinstance(low, (np.ndarray, jnp.ndarray, list)):
            self.low = low.reshape(1,n)
        else:
            self.low = low*jnp.ones((1,n))
        if isinstance(low, (np.ndarray, jnp.ndarray, list)):
            self.high = high.reshape(1,n)
        else:
            self.high = high*jnp.ones((1,n))

    
    def sample(self, batch_size, rng=None):
        if rng is None:
            u0 = random.uniform(key=random.PRNGKey(self.seed), shape=(batch_size, self.dim))
        else:
            u0 = random.uniform(key=rng, shape=(batch_size, self.dim))
        u = (u0 + jnp.sqrt(2.0)*jnp.tile(jnp.arange(batch_size).reshape(-1,1), (1,self.dim))) % 1
        return u*jnp.tile(self.high-self.low, (batch_size,1)) + jnp.tile(self.low, (batch_size,1))


class MixMultiVariateNormal(Distribution):
    def __init__(self, seed, radius=12, num=8, phi=0, scale=None, sigmas=None):
        # build mu's and sigma's
        self.dim = 2 
        self.seed = seed
        self.num = num
        arc = 2*jnp.pi/num
        xs = [jnp.sin(arc*idx+phi)*radius for idx in range(num)]
        ys = [jnp.cos(arc*idx+phi)*radius for idx in range(num)]
        self.mus = [jnp.array([x,y]) for x,y in zip(xs,ys)]
        dim = len(self.mus[0])
        self.scale = scale if scale is not None else 1.0
        self.sigmas = [jnp.eye(dim) for _ in range(num)] if sigmas is None else list(sigmas)

        self.dists=[
            dist.MultivariateNormal(mu, self.scale*sigma) for mu, sigma in zip(self.mus, self.sigmas)
            # td.multivariate_normal.MultivariateNormal(mu, sigma) for mu, sigma in zip(mus, sigmas)
        ]
        
        self.covs, self.cov_invs, self.dets = [], [], []

        for sigma in self.sigmas:
            if sigma.ndim != 0:
                assert sigma.shape[0] == sigma.shape[1] # make sure sigma is a square matrix
                cov = jnp.matmul(sigma, sigma.T)
                cov_inv = jnp.linalg.inv(cov)
                det = jnp.linalg.det(cov)
            else:
                # sigma is a scalar
                cov = sigma**2
                cov_inv = 1./cov
                det = sigma**(2*self.dim)
            self.covs.append(cov)
            self.cov_invs.append(cov_inv)
            self.dets.append(det)
        
        self.covs = jnp.stack(self.covs)
        self.cov_invs = jnp.stack(self.cov_invs)
        self.dets = jnp.stack(self.dets)
        self.mus = jnp.stack(self.mus)

    def log_p(self, x):
        # assume equally-weighted
        densities=[jnp.exp(dist.log_prob(x)) for dist in self.dists]
        return jnp.log(sum(densities)/len(self.dists))

    def sample(self, batch_size, rng=None, concat=True):
        ind_sample = batch_size/self.num
        if rng is None:
            samples = [dist.sample(random.PRNGKey(self.seed+_), sample_shape=(int(ind_sample),)) for _, dist in enumerate(self.dists)]
        else:
            samples = [dist.sample(rng, sample_shape=(int(ind_sample),)) for dist in self.dists]
        if concat:
            samples = jnp.concatenate(samples, axis=0)
        return samples
    
    def score(self, xs: jnp.ndarray):
        return v_score_gmm(xs, self.mus, self.cov_invs, self.dets)


def _density_gaussian(x, mu, cov_inv, det):
    # computes the density in a single Gaussian of a single point
    a = x - mu
    dim = x.shape[0]
    if cov_inv.ndim == 0:
        return jnp.exp(-0.5*jnp.dot(a, a) * cov_inv) / jnp.sqrt((2*jnp.pi)**dim * det)
    else:
        return jnp.exp(-0.5*jnp.dot(a, jnp.matmul(cov_inv, a))) / jnp.sqrt((2*jnp.pi)**dim * det)

v_density_gaussian = jax.vmap(_density_gaussian, in_axes=[None, 0, 0, 0])
# computes the density in several Gaussians of a single point

def _logdensity_gmm(x, mus, cov_invs, dets):
    # computes log densities of gmm of multiple points
    densities = v_density_gaussian(x, mus, cov_invs, dets)
    # densities : (self.n_Gaussians)
    return jnp.log(jnp.mean(densities, axis=0))

v_logdensity_gmm = jax.vmap(_logdensity_gmm, in_axes=[0, None, None, None])
# computes log densities of gmm of multiple points

_score_gmm = jax.grad(_logdensity_gmm)
# compute the gradient w.r.t. x

v_score_gmm = jax.vmap(_score_gmm, in_axes=[0, None, None, None])




class CheckerBoard(Distribution):
    def __init__(self, seed):
        self.dim = 2
        self.seed = seed

    def sample(self, batch_size, rng=None):
        if rng is None:
            rng1, rng2, rng3 = random.split(random.PRNGKey(self.seed), 3)
        else:
            rng1, rng2, rng3 = random.split(rng, 3)
        x1 = random.uniform(rng1, (batch_size,)) * 4 - 2
        x2_ = random.uniform(rng2, (batch_size,)) - random.randint(rng3, shape=(batch_size,), minval=0, maxval=2) * 2
        x2 = x2_ + (jnp.floor(x1) % 2)
        samples = jnp.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return samples


class Spiral(Distribution):
    def __init__(self, seed, noise):
        self.seed = seed
        self.dim = 2
        self.noise = noise

    def sample(self, batch_size, rng=None):
        if rng is None:
            random_state = self.seed
        else:
            random_state = int(random.randint(rng, shape=(1,),minval=0, maxval=100000))
        samples = skd.make_swiss_roll(n_samples=batch_size, noise=self.noise, 
                                      random_state=random_state)[0]
        samples = samples.astype("float32")[:, [0, 2]]
        return jnp.array(samples/2.0)


class Mixture(Distribution):
    def __init__(self, seed, mixtures, weights):
        '''
        Args:
            mixture: a list of distributions
            weights: weights, sum up to 1
        '''
        self.seed = seed
        self.mixtures = mixtures
        self.num_mixture = len(mixtures)
        self.weights = weights

        assert self.weights.shape[0] == self.num_mixture
        self.logit_weights = jnp.log(weights)

        self.select_fn = jax.vmap(lambda s_all, c: s_all[:, c])
    
    def sample(self, batch_size, rng=None):
        if rng is None:
            rng = random.PRNGKey(self.seed)
        choices = random.categorical(rng, self.logit_weights, axis=-1, shape=(batch_size,))

        rngs = random.split(rng, self.num_mixture+1)
        rng = rngs[0]
        rngs = rngs[1:]

        samples_each = []
        for i, mixture in enumerate(self.mixtures):
            samples_each.append(mixture.sample(rngs[i], batch_size))
        samples_all = jnp.stack(samples_each, -1)  # (batch, dim, mixture)

        samples = self.select_fn(samples_all, choices)
        return samples


class Olympic(Distribution):
    def __init__(self, seed, noise):
        self.seed = seed
        self.dim = 2
        self.noise = noise
    
    def circle_generate_sample(self, N, rng):
        angle = random.uniform(rng, shape=(N,), minval=0.0, maxval=2*jnp.pi)
        random_noise = random.normal(rng, shape=(N, 2)) * jnp.sqrt(0.2)
        pos = jnp.concatenate([jnp.cos(angle), jnp.sin(angle)])
        pos = rearrange(pos, "(b c) -> c b", b=2)
        return pos + self.noise * random_noise

    def sample(self, batch_size, concat=False):
        w = 3.5
        h = 1.5
        centers = jnp.array([[-w, h], [0.0, h], [w, h], [-w * 0.6, -h], [w * 0.6, -h]])
        pos = [
            self.circle_generate_sample(batch_size//5, random.PRNGKey(self.seed+i)) + centers[i : i + 1] / 2 for i in range(5)
        ]
        if concat:
            return jnp.concatenate(pos)
        else:
            return jnp.array(pos)  # (5, N//5, 2)
        
    



# def circle_generate_sample(N, noise=0.25):
#     angle = np.random.uniform(high=2 * np.pi, size=N)
#     random_noise = np.random.normal(scale=np.sqrt(0.2), size=(N, 2))
#     pos = np.concatenate([np.cos(angle), np.sin(angle)])
#     pos = rearrange(pos, "(b c) -> c b", b=2)
#     return pos + noise * random_noise


# def olympic_generate_sample(N, noise=0.25, concat=False):
#     w = 3.5
#     h = 1.5
#     centers = np.array([[-w, h], [0.0, h], [w, h], [-w * 0.6, -h], [w * 0.6, -h]])
#     pos = [
#         circle_generate_sample(N // 5, noise) + centers[i : i + 1] / 2 for i in range(5)
#     ]
#     if concat:
#         return np.concatenate(pos)
#     else:
#         return np.array(pos)  # (5, N//5, 2)

