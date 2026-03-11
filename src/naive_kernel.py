import jax.numpy as jnp
from jax.numpy.linalg import matrix_power

def naive_kernel(Ab, Bb, Cb, L):
    return jnp.array(
        [(Cb @ matrix_power(Ab, l) @ Bb).squeeze() for l in range(L)]
    )
