import jax.random as jrand
from flax import nnx

from generic_ssm import SSMLayer

def test_generic_ssm():
    N = 16
    H = 8
    rngs = nnx.Rngs(0)
    model = SSMLayer(rngs, N, H)
    test_inp = jrand.uniform(rngs.random(), (64,H))
    model(test_inp)