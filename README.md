
# Replicating-S4
Based off annotated S4, with some optimizations here and there.
I'm also updating everything to the new flax nnx api instead of using the flax linen api.

## Setup instructions
1. [Install uv if you haven't already](https://docs.astral.sh/uv/getting-started/installation/)
2. Run `uv sync` in the project root
3. Run `uv pip install --force-reinstall "put your jax accelerator-specific install here"`
   - i.e. `uv pip install "jax[cuda13]"` for Nvidia GPUs
   - You can skip this step if you plan on using your CPU
   - Use the [official docs](https://docs.jax.dev/en/latest/installation.html) to see what exactly you need to install for your specific hardware
4. Run `uv run -m pytest` to make sure everything works as intended

## TODO
- [x] The Recurrent Representation of an SSM
- [x] The Naive Convolution Representation of an SSM
- [x] FFTs for SSM Convolution
- [ ] Training SSMs
- [ ] S4
