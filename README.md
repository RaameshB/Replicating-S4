
# Replicating-S4
(Using the annotated S4 as a guide)

## Setup instructions
1. [Install uv if you haven't already](https://docs.astral.sh/uv/getting-started/installation/)
2. Run `uv sync` in the project root
3. Run `uv pip install --force-reinstall "put your jax accelerator-specific install here"`
   - i.e. `uv pip install "jax[cuda13]"` for Nvidia gpus
   - use the [official docs](https://docs.jax.dev/en/latest/installation.html) to see what exactly you need to install for your specific hardware
4. Run `uv run -m pytest` to make sure everything works as intended

## TODO
- [x] The Recurrent Representation of an SSM
- [x] The Naive Convolution Representation of an SSM
- [x] FFTs for SSM Convolution
- [ ] S4
