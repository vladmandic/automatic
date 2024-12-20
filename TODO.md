# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Pending

- LoRA direct with caching
- Previewer issues
- Redesign postprocessing

## Future Candidates

- Flux IPAdapter: <https://github.com/huggingface/diffusers/pull/10261>
- Flux NF4: <https://github.com/huggingface/diffusers/issues/9996>
- GGUF: <https://github.com/huggingface/diffusers/pull/9964>

## Other

- IPAdapter negative: <https://github.com/huggingface/diffusers/discussions/7167>
- Control API enhance scripts compatibility
- PixelSmith: <https://github.com/Thanos-DB/Pixelsmith>

## Code TODO

- python 3.12.4 or higher cause a mess with pydantic
- enable ROCm for windows when available
- enable full VAE mode for resize-latent
- remove duplicate mask params
- fix flux loader for civitai nf4 models
- implement model in-memory caching
- hypertile vae breaks for diffusers when using non-standard sizes
- forcing reloading entire model as loading transformers only leads to massive memory usage
- lora-direct with bnb
- make lora for quantized flux
- control script process
- monkey-patch for modernui missing tabs.select event
