# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Fix

- ultralytics package install

## Future Candidates

- stable diffusion 3.0: unreleased
- boxdiff <https://github.com/huggingface/diffusers/pull/7947>
- animatediff-sdxl <https://github.com/huggingface/diffusers/pull/6721>
- async lowvram: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14855>
- fp8: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14031>
- profiling: <https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/716>
- kohya-hires-fix: <https://github.com/huggingface/diffusers/pull/7633>
- hunyuan-dit: <https://github.com/huggingface/diffusers/pull/8290>
- init latents: variations, img2img
- diffusers public callbacks  
- include reference styles
- lora: sc lora, dora, etc
- controlnet: additional models
- resadapter: <https://github.com/bytedance/res-adapter>
- t-gate: <https://huggingface.co/docs/diffusers/main/en/optimization/tgate>

## Experimental

- [MuLan](https://github.com/mulanai/MuLan) Multi-langunage prompts - wirte your prompts in ~110 auto-detected languages!
  Compatible with SD15 and SDXL
  Enable in scripts -> MuLan and set encoder to `InternVL-14B-224px` encoder
  (that is currently only supported encoder, but others will be added)
  Note: Model will be auto-downloaded on first use: note its huge size of 27GB
  Even executing it in FP16 context will require ~16GB of VRAM for text encoder alone
  *Note*: Uses fixed prompt parser, so no prompt attention will be used
- [SDXL Flash Mini](https://huggingface.co/sd-community/sdxl-flash-mini)  
  SDXL type that weighs less, consumes less video memory, and the quality has not dropped much  
  to use, simply select from *networks -> models -> reference -> SDXL Flash Mini*  
  recommended parameters: steps: 6-9, cfg scale: 2.5-3.5, sampler: DPM++ SDE  

### Missing

- control api scripts compatibility
