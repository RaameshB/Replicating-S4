import jax.numpy as jnp
import jax.random as jrand
from icecream import ic

from init_ssm import discretize, random_ssm
from naive_kernel import naive_kernel
from recurrent_ssm import run_ssm
from ssm_convolution import casual_convolve


def test_rnn_is_cnn_conv():
    L = 16
    key = jrand.PRNGKey(0)
    key, branch = jrand.split(key)
    A, B, C = random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (L,))
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    rnn_out = run_ssm(Ab, Bb, Cb, u)
    kernel = naive_kernel(Ab, Bb, Cb, L)
    cnn_out = casual_convolve(u, kernel, fft=False)
    ic(rnn_out[:, 0], cnn_out)
    assert jnp.allclose(rnn_out[:, 0], cnn_out, rtol=1e-3)


def test_rrn_is_cnn_fft():
    L = 16
    key = jrand.PRNGKey(0)
    key, branch = jrand.split(key)
    A, B, C = random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (L,))
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    rnn_out = run_ssm(Ab, Bb, Cb, u)
    kernel = naive_kernel(Ab, Bb, Cb, L)
    cnn_out = casual_convolve(u, kernel, fft=True)
    ic(rnn_out[:, 0], cnn_out)
    assert jnp.allclose(rnn_out[:, 0], cnn_out, rtol=1e-3)


def test_cnn_conv_is_cnn_fft():
    L = 16
    key = jrand.PRNGKey(0)
    key, branch = jrand.split(key)
    A, B, C = random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (L,))
    Ab, Bb, Cb = discretize(A, B, C, step=1.0 / L)
    kernel = naive_kernel(Ab, Bb, Cb, L)
    cnn_conv_out = casual_convolve(u, kernel, fft=False)
    cnn_fft_out = casual_convolve(u, kernel, fft=True)
    ic(cnn_conv_out, cnn_fft_out)
    assert jnp.allclose(cnn_conv_out, cnn_fft_out, rtol=1e-3)
