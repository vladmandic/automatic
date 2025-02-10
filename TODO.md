# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Future Candidates

- Redesign postprocessing  
- Native FP8 compute  
- Flux NF4 loader: <https://github.com/huggingface/diffusers/issues/9996>  
- IPAdapter negative: <https://github.com/huggingface/diffusers/discussions/7167>  
- Control API enhance scripts compatibility  

## Code TODO

- flux: loader for civitai nf4 models (fixme)
- hypertile: vae breaks when using non-standard sizes (fixme)
- install: enable ROCm for windows when available (fixme)
- lora make support quantized flux (fixme)
- lora: add other quantization types (fixme)
- model load: force-reloading entire model as loading transformers only leads to massive memory usage (fixme)
- model loader: implement model in-memory caching (fixme)
- modernui: monkey-patch for missing tabs.select event (fixme)
- processing: remove duplicate mask params (fixme)
- resize image: enable full VAE mode for resize-latent (fixme)
- sana: fails when quantized (fixme)
- support scripts via api (fixme)
- transformer from-single-file with quant (fixme)
- vlm: add additional models (fixme)
