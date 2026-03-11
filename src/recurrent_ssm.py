## Reccurent SSM
# %% Imports
import jax.numpy as jnp
import jax.random as jrand
from icecream import ic
from jax import lax

import init_ssm
from init_ssm import discretize

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
def run_ssm(Ab, Bb, Cb, u):
    """
    Literally just a wrapper for scan SSM that passes in an initial state. 
    Keeping this because sometimes you don't want an initial state and I don't want to refactor my code.
    """
    N = Ab.shape[0]
    return scan_SSM(Ab, Bb, Cb, u[:, jnp.newaxis], x0=jnp.zeros((N,)))[1]


# Test the Reccurent SSM
if __name__ == "__main__":
    key = jrand.PRNGKey(0)
    L = 15
    key, branch = jrand.split(key)
    A, B, C = init_ssm.random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (L,))
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    x = run_ssm(Ab, Bb, Cb, u)
    ic(x)
