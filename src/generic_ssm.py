# %%
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.initializers import lecun_normal

from convolutional_ssm import casual_convolve
from init_ssm import discretize, log_step_initializer
from naive_kernel import naive_kernel

# %%


class SSMLayer(nnx.Module):
    def __init__(self, rngs, N):
        lecun_n = lecun_normal()
        delta_init = log_step_initializer()
        self.A = nnx.Param(lecun_n(rngs.params(), (N, N)))
        self.B = nnx.Param(lecun_n(rngs.params(), (N, 1)))
        self.C = nnx.Param(lecun_n(rngs.params(), (1, N)))
        self.D = nnx.Param(jnp.ones((1,)))
        self.log_step = nnx.Param(jnp.array([delta_init(rngs.params(), (1,))]))

    def __call__(self, u, use_cache=False, cache_state=True):
        step = jnp.exp(self.log_step)
        Ab, Bb, Cb = discretize(self.A, self.B, self.C, step)
        K = naive_kernel(Ab, Bb, Cb, u.shape[0])
        return casual_convolve(u, K)
