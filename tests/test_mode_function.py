from recurrent_ssm import run_ssm
from init_ssm import random_ssm, discretize
from naive_kernel import naive_kernel
from ssm_convolution import casual_convolve
import jax.random as jrand
import jax.numpy as jnp
import pytest
from icecream import ic

def test_rnn():
    L = 16
    key = jrand.PRNGKey(0)
    key, branch = jrand.split(key)
    A, B, C = random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (L,))
    Ab, Bb, Cb = discretize(A, B, C, step=1.0/L)
    run_ssm(Ab, Bb, Cb, u)

def test_cnn_conv():
    L = 16
    key = jrand.PRNGKey(0)
    key, branch = jrand.split(key)
    A, B, C = random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (L,))
    Ab, Bb, Cb = discretize(A, B, C, step=1.0/L)
    kernel = naive_kernel(Ab, Bb, Cb, L)
    casual_convolve(u, kernel, fft=False)

def test_cnn_fft():
    L = 16
    key = jrand.PRNGKey(0)
    key, branch = jrand.split(key)
    A, B, C = random_ssm(branch, N=6)
    key, _ = jrand.split(key)
    u = jrand.normal(key, (L,))
    Ab, Bb, Cb = discretize(A, B, C, step=1.0/L)
    kernel = naive_kernel(Ab, Bb, Cb, L)
    casual_convolve(u, kernel, fft=True)
