## Reccurent SSM
# %% Imports
from functools import partial
import jax
from jax import lax
import jax.random as jrand
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve
from init_ssm import discretize
import init_ssm
from icecream import ic

# %% The Recurrent Definition

# This is the recurrent representation of the SSM.
def scan_SSM(Ab, Bb, Cb, u, x0):
    """
    Run the SSM on an input sequence.

    Args:
        Ab: The discretized state transition matrix.
        Bb: The discretized input matrix.
        Cb: The discretized output matrix.
        u: The input sequence.
        x0: The initial state.

    Returns:
        A tuple of (x_k, y_k) where x_k is the state sequence and y_k is the output sequence.
    """
    def step(x_k_1, u_k):
        """
        Step function for the SSM.

        Args:
            x_k_1: The previous state.
            u_k: The current input.

        Returns:
            The current state and output.
        """
        x_k = Ab @ x_k_1 + Bb @ u_k
        y_k = Cb @ x_k
        return x_k, y_k
    return lax.scan(step, x0, u)

# %% Run the SSM recurrently
def run_ssm(A, B, C, u):
    L = u.shape[0]
    N = A.shape[0]
    Ab, Bb, Cb = discretize(A, B, C, step=1.0/L)
    return scan_SSM(Ab, Bb, Cb, u[:, jnp.newaxis], x0=jnp.zeros((N,)))[1]

# Test the Reccurent SSM
if __name__ == "__main__":
    key = jrand.PRNGKey(0)
    key, branch = jrand.split(key)
    A, B, C = init_ssm.random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (15,))
    x = run_ssm(A, B, C, u)
    ic(x)
