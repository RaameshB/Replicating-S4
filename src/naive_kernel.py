# %%
import jax.numpy as jnp
from flax import nnx
from jax import lax

# %%


def naive_kernel(Ab, Bb, Cb, L):
    """
    A (less) bad way to generate a kernel for an SSM. I made the implementation parallel because the provided one bothered me.
    We shall use a better baseline to evaluate S4 against.
    """
    broadcasted = jnp.broadcast_to(Ab, (L - 1,) + Ab.shape)
    concated = jnp.concat((jnp.eye(Ab.shape[0])[jnp.newaxis], broadcasted))
    matrix_powers = lax.associative_scan(jnp.matmul, concated)
    return nnx.vmap(lambda Abpower: (Cb @ Abpower @ Bb).squeeze())(matrix_powers)
