import jax.numpy as jnp

def casual_convolve(u, K, fft=True):
    if fft:
        # Here we want to make sure that both sequences can slide past each other without them overlapping.
        # To prevent convolution artifacts, we pad both sequences to be at least as long as the other.
        ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
        Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
        out = ud * Kd
        # We get rid of the results in the padded region
        return jnp.fft.irfft(out)[: u.shape[0]]
    else:
        # We get rid of the results in the padded region
        return jnp.convolve(u, K, mode='full')[: u.shape[0]]
