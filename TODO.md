# TODO

Main ToDo list can be found at [GitHub projects](https://github.com/users/vladmandic/projects)

## Candidates for next release

### Models & Pipelines

- stable diffusion 3.0

### Features

- animatediff-sdxl <https://github.com/huggingface/diffusers/pull/6721>
- async lowvram: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14855>
- fp8: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14031>
- init latents: variations, img2img
- diffusers public callbacks  
- include reference styles
- lora: sc lora, dora, etc

### Missing

- control api scripts compatibility

## Future release notes

### requires `diffusers-0.28.0.dev0`

- PixArt-Σ
- IP adapter masking
- InstantStyle
- Sampler timesteps

## Note

New release has been baking in `dev` for a longer than usual, but changes are massive...

First, this version of [SD.Next](https://github.com/vladmandic/automatic) ships with a preview of the new [ModernUI](https://github.com/BinaryQuantumSoul/sdnext-modernui)  
For details on how to enable and use it, see [Home](https://github.com/BinaryQuantumSoul/sdnext-modernui) and [WiKi](https://github.com/vladmandic/automatic/wiki/Themes)  
**ModernUI** is still in early development and not all features are available yet, please report [issues and feedback](https://github.com/BinaryQuantumSoul/sdnext-modernui/issues)  
Thanks to @BinaryQuantumSoul for his hard work on this project!  

*What else?*

### New built-in features

- HiDiffusion allows generating very-high resolution images out-of-the-box using standard models  
- Perturbed-Attention Guidance (PAG) enhances sample quality in addition to standard CFG scale  
- IP adapter masking allows to use multiple input images for each segment of the input image  
- IP adapter InstantStyle implementation  
- Token Downsampling (ToDO) provides significant speedups with minimal-to-none quality loss  
- Samplers optimizations that allow normal samplers to complete work in 1/3 of the steps!  
  Yup, even popular DPM++2M can now run in 10 steps with quality equaling 30 steps  
- Better outpainting

### New models

While still waiting for *Stable Diffusion 3.0*, there have been some significant models released in the meantime:
- [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/), high end diffusion Transformer model (DiT) with a T5 encoder/decoder capable of directly generating images at 4K resolution  
- [SDXS](https://github.com/IDKiro/sdxs), an extremely fast 1-step generation consistency model that also uses TAESD as quick VAE out-of-the-box  
- [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD), 1-step, 2-step, 4-step and 8-step optimized models

*Note*  
[SD.Next](https://github.com/vladmandic/automatic) is no longer marked as a fork of [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) and github project has been fully detached  
Given huge number of changes with *+3443/-3342* commits diff (at the time of fork detach) over the past year,  
a completely different backend/engine and a change of focus, it is time to give credit to original [author](https://github.com/auTOMATIC1111),  and move on!  
