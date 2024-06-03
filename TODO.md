# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Future Candidates

- stable diffusion 3.0: unreleased
- boxdiff <https://github.com/huggingface/diffusers/pull/7947>
- animatediff-sdxl <https://github.com/huggingface/diffusers/pull/6721>
- async lowvram: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14855>
- fp8: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14031>
- profiling: <https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/716>
- init latents: variations, img2img
- diffusers public callbacks  
- include reference styles
- lora: sc lora, dora, etc
- resadapter: <https://github.com/bytedance/res-adapter>

## Experimental

- [SDXL Flash Mini](https://huggingface.co/sd-community/sdxl-flash-mini)  
  SDXL type that weighs less, consumes less video memory, and the quality has not dropped much  
  to use, simply select from *networks -> models -> reference -> SDXL Flash Mini*  
  recommended parameters: steps: 6-9, cfg scale: 2.5-3.5, sampler: DPM++ SDE  

### Missing

- control api scripts compatibility
