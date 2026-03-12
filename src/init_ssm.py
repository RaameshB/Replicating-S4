# %%
import jax.numpy as jnp
import jax.random as jrand
from jax.numpy.linalg import inv


# This gets us our weight parameters.
def random_ssm(rng, N):
    """
    Generate random parameters for a discrete-time SSM.

    Args:
        rng: A JAX random key.
        N: The state size.

    Returns:
        A tuple of (A, B, C) where A is the state transition matrix,
        B is the input matrix, and C is the output matrix.
    """
    a_r, b_r, c_r = jrand.split(rng, 3)
    A = jrand.uniform(a_r, shape=(N, N))
    B = jrand.uniform(b_r, shape=(N, 1))
    C = jrand.uniform(c_r, shape=(1, N))
    return A, B, C


# The parameters we generate via random_ssm are for a continous-time SSM.
# To discretize the SSM we use the bilinear method, as per the S4 paper.
# (Note, most modern SSMs use Zero-Order Hold (ZOH) discretization.)
def discretize(A, B, C, step):
    """
    Discretize a continuous-time SSM using the bilinear method.

    Args:
        A: The state transition matrix.
        B: The input matrix.
        C: The output matrix.
        step: The time step.

    Returns:
        A tuple of (Ab, Bb, C) where Ab is the discretized state transition matrix ($\bar{A}$),
        Bb is the discretized input matrix ($\bar{B}$), and C is the output matrix ($\bar{C}$).
    """
    I = jnp.eye(A.shape[0])
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """
    This is sigificantly different than what's in the Annotated S4, but I feel like it is a better implementation of what they have
    """
    return lambda key, shape: jrand.uniform(
        key, shape, minval=jnp.log(dt_min), maxval=jnp.log(dt_max)
    )
