# %%
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.initializers import lecun_normal

from convolutional_ssm import casual_convolve
from init_ssm import discretize, log_step_initializer
from naive_kernel import naive_kernel

# %%


class SSMLayer(nnx.Module):
    def __init__(self, rngs, N, H):
        lecun_n = lecun_normal()
        delta_init = log_step_initializer()
        self.As = nnx.Param(lecun_n(rngs.params(), (H, N, N)))
        self.Bs = nnx.Param(lecun_n(rngs.params(), (H, N, 1)))
        self.Cs = nnx.Param(lecun_n(rngs.params(), (H, 1, N)))
        self.Ds = nnx.Param(jnp.ones((H,)))
        self.log_steps = nnx.Param(jnp.array(delta_init(rngs.params(), (H,))))

    def ssm(u, log_step, A, B, C, D):
        step = jnp.exp(log_step)
        Ab, Bb, Cb = discretize(A, B, C, step)
        K = naive_kernel(Ab, Bb, Cb, u.shape[0])
        return casual_convolve(u, K) + D * u

    def __call__(self, inp):
        return nnx.vmap(SSMLayer.ssm, in_axes=(1, 0, 0, 0, 0, 0), out_axes=1)(inp, self.log_steps, self.As, self.Bs, self.Cs, self.Ds)
