import jax.numpy as jnp
import jax
from flax import linen as nn

from .activation import ActivationFactory
from .time_emb import TimeEmbedding, SpaceEmbedding


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    embed_dim: int
    key: jnp.ndarray
    scale: float = 30.

    @nn.compact
    def __call__(self, x):
        def _init(key, embed_dim, scale):
            W = jax.random.normal(key, [embed_dim // 2])
            W = W * scale
            return W

        kernel = self.param('random_feature', _init, self.embed_dim, self.scale)
        x_proj = x[:, None] * kernel[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    


# class MLPNet(nn.Module):
#     dim: int  # the ambient dimension of the problem
#     key: jnp.ndarray
#     hidden_dims = [64, 64, 64]
#     embed_dim: int
#     activation: str

#     def setup(self):
#         self.embed = GaussianFourierProjection(embed_dim=self.embed_dim, key=self.key)
#         self.embed_dense = nn.Dense(features=self.embed_dim)
#         self.layers = [nn.Dense(dim_out) for dim_out in self.hidden_dims + [self.dim]]
#         self.embed_denses = [nn.Dense(dim_out) for dim_out in self.hidden_dims + [self.dim]]
#         # for dim_out in self.hidden_dims + [self.dim]:
#         #     layer = ConcatSquashLinear(dim_out)
#         #     self.layers.append(layer)
#         # self.act = lambda x: x * jax.nn.sigmoid(x)
#         self.act = ActivationFactory.create(self.activation)

#     def __call__(self, t, x):

#         if type(t) is float:
#             t = jnp.ones(1) * t
#         elif t.ndim == 0:
#             t = jnp.ones(1) * t

#         embed = jnp.squeeze(self.act(self.embed_dense(self.embed(t))))

#         dx = x
#         for i, (layer, embed_dense) in enumerate(zip(self.layers, self.embed_denses)):
#             dx = layer(dx) + embed_dense(embed)
#             if i < len(self.layers) - 1:
#                 dx = self.act(dx)

#         return dx
    

class LinearSkipConnection(nn.Module):
    rank: int
    layer_size: int
    num_layer: int

    @nn.compact
    def __call__(self, t, x):
        '''
        Args:
          t: (T,) time after some embedding.
          x: (D,), a point to evaluate the velocity at.
        '''
        dim = x.shape[-1]
        # t = TimeEmbedding(self.embed_time_dim)(t)
        out_size = (2 * self.rank + 1) * dim

        cur = t
        for i in range(self.num_layer):
            cur = nn.Dense(self.layer_size)(cur)
            cur = nn.silu(cur)
        Wb = nn.Dense(out_size)(cur)

        U, V, b = (Wb[:self.rank * dim],
                   Wb[self.rank * dim:2 * self.rank * dim],
                   Wb[-dim:])
        U = U.reshape(self.rank, dim)
        V = V.reshape(self.rank, dim)

        out = (jnp.transpose(U) @ (V @ jnp.expand_dims(x, -1))).squeeze(-1) + b
        return out


class MLPNet(nn.Module):
    dim: int
    num_layer: int
    layer_size: int
    activation: str
    kernel_var: float
    embed_time_dim: int   # 0 if not embedding_time
    embed_space_dim: int
    use_skip: bool
    use_residual: bool
    skip_only: bool
    layer_norm: bool

    @nn.compact
    def __call__(self, t, x):
        '''
        Note this is not batched.

        Args:
          t: A scalar, time.
          x: (D,), a point to evaluate the velocity at.
        Returns:
          (D,), velocity at (x, t).
        '''
        kernel_init = nn.initializers.variance_scaling(self.kernel_var, 'fan_in', 'truncated_normal')

        if self.skip_only:
            assert(self.use_skip)
        if self.embed_time_dim > 0:
            t = TimeEmbedding(self.embed_time_dim)(t)
        else:
            t = jnp.expand_dims(t, -1)
        # t = jnp.ones((x.shape[0],1)) * t[None,:]
        if self.use_skip:
            x_skip = LinearSkipConnection(rank=20,
                                          layer_size=self.layer_size,
                                          num_layer=self.num_layer)(t, x)
        if self.embed_space_dim > 0:
            x = SpaceEmbedding(sigma=1.0,
                               in_dim=self.dim,
                               out_dim=self.embed_space_dim)(x)
            
        if x.ndim == 2:
            t = jnp.ones((x.shape[0],1)) * t[None,:]
            
        x_t_ori = jnp.concatenate([x, t], axis=-1)
        x = x_t_ori
        for i in range(self.num_layer):
            x = nn.Dense(self.layer_size, kernel_init=kernel_init)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            if self.use_residual and i > 0:
                y = nn.Dense(self.layer_size, kernel_init=kernel_init)(x_t_ori)
                x += y
            x = ActivationFactory.create(self.activation)(x)

        x = nn.Dense(self.dim, kernel_init=kernel_init)(x)

        if self.use_skip:
            if self.skip_only:
                x = x_skip
            else:
                x += x_skip

        return x
