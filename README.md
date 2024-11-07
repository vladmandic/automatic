<div align="center">
<img src="https://github.com/vladmandic/automatic/blob/dev/html/favicon.png" width=200 alt="SD.Next">

**Stable Diffusion implementation with advanced features**

[![Sponsors](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/vladmandic)
![Last Commit](https://img.shields.io/github/last-commit/vladmandic/automatic?svg=true)
![License](https://img.shields.io/github/license/vladmandic/automatic?svg=true)
[![Discord](https://img.shields.io/discord/1101998836328697867?logo=Discord&svg=true)](https://discord.gg/VjvR2tabEX)

[Wiki](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.gg/VjvR2tabEX) | [Changelog](CHANGELOG.md)

</div>
</br>

## Table of contents

- [SD.Next Features](#sdnext-features)
- [Model support](#model-support)
- [Platform support](#platform-support)
- [Backend support](#backend-support)
- [Examples](#examples)
- [Install](#install)
- [Notes](#notes)

## SD.Next Features

All individual features are not listed here, instead check [ChangeLog](CHANGELOG.md) for full list of changes
- Multiple backends!  
  ▹ **Diffusers | Original**  
- Multiple UIs!  
  ▹ **Standard | Modern**  
- Multiple diffusion models!  
  ▹ **Stable Diffusion 1.5/2.1/XL/3.0/3.5 | LCM | Lightning | Segmind | Kandinsky | Pixart-α | Pixart-Σ | Stable Cascade | FLUX.1 | AuraFlow | Würstchen | Alpha Lumina | Kwai Kolors | aMUSEd | DeepFloyd IF | UniDiffusion | SD-Distilled | BLiP Diffusion | KOALA | SDXS | Hyper-SD | HunyuanDiT | CogView | OmniGen | Meissonic | etc.**
- Built-in Control for Text, Image, Batch and video processing!  
  ▹ **ControlNet | ControlNet XS | Control LLLite | T2I Adapters | IP Adapters**  
- Multiplatform!  
 ▹ **Windows | Linux | MacOS | nVidia | AMD | IntelArc/IPEX | DirectML | OpenVINO | ONNX+Olive | ZLUDA**
- Platform specific autodetection and tuning performed on install
- Optimized processing with latest `torch` developments with built-in support for `torch.compile`  
  and multiple compile backends: *Triton, ZLUDA, StableFast, DeepCache, OpenVINO, NNCF, IPEX, OneDiff*  
- Improved prompt parser  
- Enhanced *Lora*/*LoCon*/*Lyco* code supporting latest trends in training  
- Built-in queue management  
- Enterprise level logging and hardened API  
- Built in installer with automatic updates and dependency management  
- Modernized UI with theme support and number of built-in themes *(dark and light)*  
- Mobile compatible

<br>

*Main interface using **StandardUI***:  
![screenshot-text2image](https://github.com/user-attachments/assets/87ac2813-65c2-45f4-80b8-67b26ccf5cd6)

*Main interface using **ModernUI***:  

![screenshot-modernui-f1](https://github.com/user-attachments/assets/b509a280-8d3b-48b5-8525-363bad8c1ed2)
![screenshot-modernui](https://github.com/user-attachments/assets/fef33127-f733-4e78-b66e-17729539512f)
![screenshot-modernui-sd3](https://github.com/user-attachments/assets/1ed02ecc-23e4-4fda-8ae5-2d7393dc530c)

For screenshots and informations on other available themes, see [Themes Wiki](https://github.com/vladmandic/automatic/wiki/Themes)

<br>

## Model support

Additional models will be added as they become available and there is public interest in them  
See [models overview](https://github.com/vladmandic/automatic/wiki/Models) for details on each model, including their architecture, complexity and other info  

- [RunwayML Stable Diffusion](https://github.com/Stability-AI/stablediffusion/) 1.x and 2.x *(all variants)*
- [StabilityAI Stable Diffusion XL](https://github.com/Stability-AI/generative-models)
- [StabilityAI Stable Diffusion](https://stability.ai/news/stable-diffusion-3-medium) 
- [Stable Diffusion 3.x](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) 3.0 Medium, 3.5 Medium, 3.5 Large, 3.5 Large Turbo
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
- [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega)
- [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B)
- [Segmind SegMoE](https://github.com/segmind/segmoe) *SD and SD-XL*
- [Segmind SD Distilled](https://huggingface.co/blog/sd_distillation) *(all variants)*
- [Kandinsky](https://github.com/ai-forever/Kandinsky-2) *2.1 and 2.2 and latest 3.0*
- [PixArt-α XL 2](https://github.com/PixArt-alpha/PixArt-alpha) *Medium and Large*
- [PixArt-Σ](https://github.com/PixArt-alpha/PixArt-sigma)
- [Warp Wuerstchen](https://huggingface.co/blog/wuertschen)
- [Tsinghua UniDiffusion](https://github.com/thu-ml/unidiffuser)
- [DeepFloyd IF](https://github.com/deep-floyd/IF) *Medium and Large*
- [ModelScope T2V](https://huggingface.co/damo-vilab/text-to-video-ms-1.7b)
- [BLIP-Diffusion](https://dxli94.github.io/BLIP-Diffusion-website/)
- [KOALA 700M](https://github.com/youngwanLEE/sdxl-koala)
- [VGen](https://huggingface.co/ali-vilab/i2vgen-xl)
- [SDXS](https://github.com/IDKiro/sdxs)
- [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)


Also supported are modifiers such as:
- **LCM**, **Turbo** and **Lightning** (*adversarial diffusion distillation*) networks
- All **LoRA** types such as LoCon, LyCORIS, HADA, IA3, Lokr, OFT
- **IP-Adapters** for SD 1.5 and SD-XL
- **InstantID**, **FaceSwap**, **FaceID**, **PhotoMerge**  
- **AnimateDiff** for SD 1.5
- **MuLAN** multi-language support  

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

## Backend support

**SD.Next** supports two main backends: *Diffusers* and *Original*:

- **Diffusers**: Based on new [Huggingface Diffusers](https://huggingface.co/docs/diffusers/index) implementation  
  Supports *all* models listed below  
  This backend is set as default for new installations  
  See [wiki article](https://github.com/vladmandic/automatic/wiki/Diffusers) for more information  
- **Original**: Based on [LDM](https://github.com/Stability-AI/stablediffusion) reference implementation and significantly expanded on by [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
  This backend and is fully compatible with most existing functionality and extensions written for *A1111 SDWebUI*  
  Supports **SD 1.x** and **SD 2.x** models  
  All other model types such as *SD-XL, LCM, Stable Cascade, PixArt, Playground, Segmind, Kandinsky, etc.* require backend **Diffusers**  

## Examples

*IP Adapters*:
![screenshot-ipadapter](https://github.com/user-attachments/assets/92830894-845c-49ec-92d9-18c8a577d04f)

*Color grading*:  
![screenshot-control](https://github.com/user-attachments/assets/cdad2722-ae7c-4c9c-94d6-5ea35a4b1356)

*InstantID*:  
![screenshot-instantid](https://github.com/user-attachments/assets/f38a5660-32b3-4235-9da1-c79eccf5372f)

> [!IMPORTANT]
> - Loading any model other than standard SD 1.x / SD 2.x requires use of backend **Diffusers**  
> - Loading any other models using **Original** backend is not supported  
> - Loading manually download model `.safetensors` files is supported for specified models only (typically SD 1.x / SD 2.x / SD-XL models only)  
> - For all other model types, use backend **Diffusers** and use built in Model downloader or  
  select model from Networks -> Models -> Reference list in which case it will be auto-downloaded and loaded  

## Install

- [Step-by-step install guide](https://github.com/vladmandic/automatic/wiki/Installation)
- [Advanced install notes](https://github.com/vladmandic/automatic/wiki/Advanced-Install)
- [Video: install and use](https://www.youtube.com/watch?v=nWTnTyFTuAs)
- [Common installation errors](https://github.com/vladmandic/automatic/discussions/1627)  
- [FAQ](https://github.com/vladmandic/automatic/discussions/1011)

> [!TIP]
> - If you can't run SD.Next locally, try cloud deployment using [RunDiffusion](https://rundiffusion.com?utm_source=github&utm_medium=referral&utm_campaign=SDNext)!
> - Server can run with or without virtual environment,  
  Recommended to use `VENV` to avoid library version conflicts with other applications  
> - **nVidia/CUDA** / **AMD/ROCm** / **Intel/OneAPI** are auto-detected if present and available,  
  For any other use case such as **DirectML**, **ONNX/Olive**, **OpenVINO** specify required parameter explicitly  
  or wrong packages may be installed as installer will assume CPU-only environment  
> - Full startup sequence is logged in `sdnext.log`,  
  so if you encounter any issues, please check it first  

### Run

Once SD.Next is installed, simply run `webui.ps1` or `webui.bat` (*Windows*) or `webui.sh` (*Linux or MacOS*)

For list of available command line options, run `webui --help` for the full & up-to-date list

> [!TIP]
> All command line options can also be set via env variable
> For example `--debug` is same as `set SD_DEBUG=true`  

## Notes

> [!TIP]
> If you don't want to use built-in `venv` support and prefer to run SD.Next in your own environment such as *Docker* container, *Conda* environment or any other virtual environment, you can skip `venv` create/activate and launch SD.Next directly using `python launch.py` (command line flags noted above still apply).

### Quantization

**SD.Next** comes with broad quantization support, including support for BitsAndBytes, Optimum.Quanto, TorchAO, NNCF and GGUF
See [Quantization Wiki](https://github.com/vladmandic/automatic/wiki/Quantization)

### Control

**SD.Next** comes with built-in control for all types of text2image, image2image, video2video and batch processing

*Control interface*:  
![screenshot-control](https://github.com/user-attachments/assets/cdad2722-ae7c-4c9c-94d6-5ea35a4b1356)

*Control processors*:  
![screenshot-processors](https://github.com/user-attachments/assets/7bccb82b-366e-4bdb-ae57-cc53fac95d3c)

*Masking*:
![screenshot-mask](https://github.com/user-attachments/assets/4b057e65-64f0-44ea-93b4-c3b69bc55532)

### Extensions

SD.Next comes with several extensions pre-installed:

- [System Info](https://github.com/vladmandic/sd-extension-system-info)
- [chaiNNer](https://github.com/vladmandic/sd-extension-chainner)
- [RemBg](https://github.com/vladmandic/sd-extension-rembg)
- [Agent Scheduler](https://github.com/ArtVentureX/sd-webui-agent-scheduler)
- [Modern UI](https://github.com/BinaryQuantumSoul/sdnext-modernui)

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

- [Wiki](https://github.com/vladmandic/automatic/wiki)
- [ReadMe](README.md)
- [ToDo](TODO.md)  
- [ChangeLog](CHANGELOG.md)
- [CLI Tools](cli/README.md)

### Sponsors

<div align="center">
<!-- sponsors --><a href="https://github.com/allangrant"><img src="https://github.com/allangrant.png" width="60px" alt="Allan Grant" /></a><a href="https://github.com/BrentOzar"><img src="https://github.com/BrentOzar.png" width="60px" alt="Brent Ozar" /></a><a href="https://github.com/inktomi"><img src="https://github.com/inktomi.png" width="60px" alt="Matthew Runo" /></a><a href="https://github.com/mantzaris"><img src="https://github.com/mantzaris.png" width="60px" alt="a.v.mantzaris" /></a><a href="https://github.com/CurseWave"><img src="https://github.com/CurseWave.png" width="60px" alt="" /></a><a href="https://github.com/smlbiobot"><img src="https://github.com/smlbiobot.png" width="60px" alt="SML (See-ming Lee)" /></a><!-- sponsors -->
</div>

<br>
