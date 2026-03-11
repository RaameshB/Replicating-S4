import jax.numpy as jnp
from jax.numpy.linalg import matrix_power


def naive_kernel(Ab, Bb, Cb, L):
    """
    A really bad way to generate a kernel for an SSM. Like come on, its not even parallel.
    """
    return jnp.array([(Cb @ matrix_power(Ab, l) @ Bb).squeeze() for l in range(L)])
