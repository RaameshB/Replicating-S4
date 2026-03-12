import jax.random as jrand
from flax import nnx

from generic_ssm import SSMLayer


def test_generic_ssm():
    rngs = nnx.Rngs(0)
    model = SSMLayer(rngs, 16)
    test_inp = jrand.uniform(rngs.random(), (64,))
    model(test_inp)
