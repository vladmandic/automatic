<div align="center">
<img src="https://github.com/vladmandic/automatic/blob/master/html/logo-transparent.png" width=200 alt="SD.Next">

**Image Diffusion implementation with advanced features**

![Last update](https://img.shields.io/github/last-commit/vladmandic/automatic?svg=true)
![License](https://img.shields.io/github/license/vladmandic/automatic?svg=true)
[![Discord](https://img.shields.io/discord/1101998836328697867?logo=Discord&svg=true)](https://discord.gg/VjvR2tabEX)
[![Sponsors](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/vladmandic)

[Wiki](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.gg/VjvR2tabEX) | [Changelog](CHANGELOG.md)

</div>
</br>

## Table of contents

- [SD.Next Features](#sdnext-features)
- [Model support](#model-support)
- [Platform support](#platform-support)
- [Getting started](#getting-started)

## SD.Next Features

All individual features are not listed here, instead check [ChangeLog](CHANGELOG.md) for full list of changes
- Multiple UIs!  
  ▹ **Standard | Modern**  
- Multiple diffusion models!  
- Built-in Control for Text, Image, Batch and video processing!  
- Multiplatform!  
 ▹ **Windows | Linux | MacOS | nVidia | AMD | IntelArc/IPEX | DirectML | OpenVINO | ONNX+Olive | ZLUDA**
- Multiple backends!  
  ▹ **Diffusers | Original**  
- Platform specific autodetection and tuning performed on install  
- Optimized processing with latest `torch` developments with built-in support for `torch.compile`  
  and multiple compile backends: *Triton, ZLUDA, StableFast, DeepCache, OpenVINO, NNCF, IPEX, OneDiff*  
- Improved prompt parser  
- Built-in queue management  
- Enterprise level logging and hardened API  
- Built in installer with automatic updates and dependency management  
- Mobile compatible  

<br>

*Main interface using **StandardUI***:  
![screenshot-standardui](https://github.com/user-attachments/assets/cab47fe3-9adb-4d67-aea9-9ee738df5dcc)

*Main interface using **ModernUI***:  

![screenshot-modernui](https://github.com/user-attachments/assets/39e3bc9a-a9f7-4cda-ba33-7da8def08032)

For screenshots and informations on other available themes, see [Themes Wiki](https://github.com/vladmandic/automatic/wiki/Themes)

<br>

## Model support

Additional models will be added as they become available and there is public interest in them  
See [models overview](wiki/Models) for details on each model, including their architecture, complexity and other info  

- [RunwayML Stable Diffusion](https://github.com/Stability-AI/stablediffusion/) 1.x and 2.x *(all variants)*
- [StabilityAI Stable Diffusion XL](https://github.com/Stability-AI/generative-models), [StabilityAI Stable Diffusion 3.0](https://stability.ai/news/stable-diffusion-3-medium) Medium, [StabilityAI Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) Medium, Large, Large Turbo
- [StabilityAI Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) Base, XT 1.0, XT 1.1
- [StabilityAI Stable Cascade](https://github.com/Stability-AI/StableCascade) *Full* and *Lite*
- [Black Forest Labs FLUX.1](https://blackforestlabs.ai/announcing-black-forest-labs/) Dev, Schnell  
- [AuraFlow](https://huggingface.co/fal/AuraFlow)
- [AlphaVLLM Lumina-Next-SFT](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)  
- [Playground AI](https://huggingface.co/playgroundai/playground-v2-256px-base) *v1, v2 256, v2 512, v2 1024 and latest v2.5*
- [Tencent HunyuanDiT](https://github.com/Tencent/HunyuanDiT)
- [OmniGen](https://arxiv.org/pdf/2409.11340)  
- [Meissonic](https://github.com/viiika/Meissonic)  
- [Kwai Kolors](https://huggingface.co/Kwai-Kolors/Kolors)  
- [CogView 3+](https://huggingface.co/THUDM/CogView3-Plus-3B)
- [LCM: Latent Consistency Models](https://github.com/openai/consistency_models)
- [aMUSEd](https://huggingface.co/amused/amused-256) 256 and 512
- [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega), [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B), [Segmind SegMoE](https://github.com/segmind/segmoe) *SD and SD-XL*, [Segmind SD Distilled](https://huggingface.co/blog/sd_distillation) *(all variants)*
- [Kandinsky](https://github.com/ai-forever/Kandinsky-2) *2.1 and 2.2 and latest 3.0*
- [PixArt-α XL 2](https://github.com/PixArt-alpha/PixArt-alpha) *Medium and Large*, [PixArt-Σ](https://github.com/PixArt-alpha/PixArt-sigma)
- [Warp Wuerstchen](https://huggingface.co/blog/wuertschen)
- [Tsinghua UniDiffusion](https://github.com/thu-ml/unidiffuser)
- [DeepFloyd IF](https://github.com/deep-floyd/IF) *Medium and Large*
- [ModelScope T2V](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)
- [BLIP-Diffusion](https://dxli94.github.io/BLIP-Diffusion-website/)
- [KOALA 700M](https://github.com/youngwanLEE/sdxl-koala)
- [VGen](https://huggingface.co/ali-vilab/i2vgen-xl)
- [SDXS](https://github.com/IDKiro/sdxs)
- [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)

## Platform support

- *nVidia* GPUs using **CUDA** libraries on both *Windows and Linux*  
- *AMD* GPUs using **ROCm** libraries on *Linux*  
  Support will be extended to *Windows* once AMD releases ROCm for Windows  
- *Intel Arc* GPUs using **OneAPI** with *IPEX XPU* libraries on both *Windows and Linux*  
- Any GPU compatible with *DirectX* on *Windows* using **DirectML** libraries  
  This includes support for AMD GPUs that are not supported by native ROCm libraries  
- Any GPU or device compatible with **OpenVINO** libraries on both *Windows and Linux*  
- *Apple M1/M2* on *OSX* using built-in support in Torch with **MPS** optimizations  
- *ONNX/Olive*  
- *AMD* GPUs on Windows using **ZLUDA** libraries

## Getting started

- Get started with **SD.Next** by following the [installation instructions](wiki/Installation)  
- For more details, check out [advanced installation](wiki/Advanced-Install) guide  
- List and explanation of [command line arguments](wiki/CLI-Arguments)
- Install walkthrough [video](https://www.youtube.com/watch?v=nWTnTyFTuAs)

> [!TIP]
> And for platform specific information, check out  
> [WSL](wiki/WSL) | [Intel Arc](wiki/Intel-ARC) | [DirectML](wiki/DirectML) | [OpenVINO](wiki/OpenVINO) | [ONNX & Olive](wiki/ONNX-Runtime) | [ZLUDA](wiki/ZLUDA) | [AMD ROCm](wiki/AMD-ROCm) | [MacOS](wiki/MacOS-Python.md) | [nVidia](wiki/nVidia)

> [!WARNING]
> If you run into issues, check out [troubleshooting](wiki/Troubleshooting) and [debugging](wiki/Debug) guides  

> [!TIP]
> All command line options can also be set via env variable
> For example `--debug` is same as `set SD_DEBUG=true`  

## Backend support

**SD.Next** supports two main backends: *Diffusers* and *Original*:

- **Diffusers**: Based on new [Huggingface Diffusers](https://huggingface.co/docs/diffusers/index) implementation  
  Supports *all* models listed below  
  This backend is set as default for new installations  
- **Original**: Based on [LDM](https://github.com/Stability-AI/stablediffusion) reference implementation and significantly expanded on by [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
  This backend and is fully compatible with most existing functionality and extensions written for *A1111 SDWebUI*  
  Supports **SD 1.x** and **SD 2.x** models  
  All other model types such as *SD-XL, LCM, Stable Cascade, PixArt, Playground, Segmind, Kandinsky, etc.* require backend **Diffusers**  

### Collab

- We'd love to have additional maintainers (with comes with full repo rights). If you're interested, ping us!  
- In addition to general cross-platform code, desire is to have a lead for each of the main platforms  
This should be fully cross-platform, but we'd really love to have additional contributors and/or maintainers to join and help lead the efforts on different platforms  

### Credits

- Main credit goes to [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for original codebase  
- Additional credits are listed in [Credits](https://github.com/AUTOMATIC1111/stable-diffusion-webui/#credits)  
- Licenses for modules are listed in [Licenses](html/licenses.html)  

### Evolution

<a href="https://star-history.com/#vladmandic/automatic&Date">
  <picture width=640>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=vladmandic/automatic&type=Date&theme=dark" />
    <img src="https://api.star-history.com/svg?repos=vladmandic/automatic&type=Date" alt="starts" width="320">
  </picture>
</a>

- [OSS Stats](https://ossinsight.io/analyze/vladmandic/automatic#overview)

### Docs

If you're unsure how to use a feature, best place to start is [Wiki](https://github.com/vladmandic/automatic/wiki) and if its not there,  
check [ChangeLog](CHANGELOG.md) for when feature was first introduced as it will always have a short note on how to use it  

### Sponsors

<div align="center">
<!-- sponsors --><a href="https://github.com/allangrant"><img src="https://github.com/allangrant.png" width="60px" alt="Allan Grant" /></a><a href="https://github.com/BrentOzar"><img src="https://github.com/BrentOzar.png" width="60px" alt="Brent Ozar" /></a><a href="https://github.com/inktomi"><img src="https://github.com/inktomi.png" width="60px" alt="Matthew Runo" /></a><a href="https://github.com/mantzaris"><img src="https://github.com/mantzaris.png" width="60px" alt="a.v.mantzaris" /></a><a href="https://github.com/CurseWave"><img src="https://github.com/CurseWave.png" width="60px" alt="" /></a><a href="https://github.com/smlbiobot"><img src="https://github.com/smlbiobot.png" width="60px" alt="SML (See-ming Lee)" /></a><!-- sponsors -->
</div>

<br>
