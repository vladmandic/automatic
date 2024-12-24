# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Pending

- LoRA direct with caching
- Previewer issues
- Redesign postprocessing

## Future Candidates

- Flux NF4 loader: <https://github.com/huggingface/diffusers/issues/9996>
- IPAdapter negative: <https://github.com/huggingface/diffusers/discussions/7167>
- Control API enhance scripts compatibility
- PixelSmith: <https://github.com/Thanos-DB/Pixelsmith>

## Code TODO

- TODO install: python 3.12.4 or higher cause a mess with pydantic
- TODO install: enable ROCm for windows when available
- TODO resize image: enable full VAE mode for resize-latent
- TODO processing: remove duplicate mask params
- TODO flux: fix loader for civitai nf4 models
- TODO model loader: implement model in-memory caching
- TODO hypertile: vae breaks when using non-standard sizes
- TODO model load: force-reloading entire model as loading transformers only leads to massive memory usage
- TODO lora load: direct with bnb
- TODO lora make: support quantized flux
- TODO control: support scripts via api
- TODO modernui: monkey-patch for missing tabs.select event
