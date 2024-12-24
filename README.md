<div align="center">
<img src="https://github.com/vladmandic/automatic/raw/master/html/logo-transparent.png" width=200 alt="SD.Next">

**Image Diffusion implementation with advanced features**

![Last update](https://img.shields.io/github/last-commit/vladmandic/automatic?svg=true)
![License](https://img.shields.io/github/license/vladmandic/automatic?svg=true)
[![Discord](https://img.shields.io/discord/1101998836328697867?logo=Discord&svg=true)](https://discord.gg/VjvR2tabEX)
[![Sponsors](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub&color=%23fe8e86)](https://github.com/sponsors/vladmandic)

[Docs](https://vladmandic.github.io/sdnext-docs/) | [Wiki](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.gg/VjvR2tabEX) | [Changelog](CHANGELOG.md)

</div>
</br>

## Table of contents

- [Documentation](https://vladmandic.github.io/sdnext-docs/)
- [SD.Next Features](#sdnext-features)
- [Model support](#model-support) and [Specifications]()
- [Platform support](#platform-support)
- [Getting started](#getting-started)

## SD.Next Features

All individual features are not listed here, instead check [ChangeLog](CHANGELOG.md) for full list of changes
- Multiple UIs!  
  ▹ **Standard | Modern**  
- Multiple [diffusion models](https://vladmandic.github.io/sdnext-docs/Model-Support/)!  
- Built-in Control for Text, Image, Batch and video processing!  
- Multiplatform!  
 ▹ **Windows | Linux | MacOS | nVidia | AMD | IntelArc/IPEX | DirectML | OpenVINO | ONNX+Olive | ZLUDA**
- Multiple backends!  
  ▹ **Diffusers | Original**  
- Platform specific autodetection and tuning performed on install  
- Optimized processing with latest `torch` developments with built-in support for `torch.compile`  
  and multiple compile backends: *Triton, ZLUDA, StableFast, DeepCache, OpenVINO, NNCF, IPEX, OneDiff*  
- Built-in queue management  
- Built in installer with automatic updates and dependency management  
- Mobile compatible  

<br>

*Main interface using **StandardUI***:  
![screenshot-standardui](https://github.com/user-attachments/assets/cab47fe3-9adb-4d67-aea9-9ee738df5dcc)

*Main interface using **ModernUI***:  

![screenshot-modernui](https://github.com/user-attachments/assets/39e3bc9a-a9f7-4cda-ba33-7da8def08032)

For screenshots and informations on other available themes, see [Themes](https://vladmandic.github.io/sdnext-docs/Themes/)

<br>

## Model support

SD.Next supports broad range of models: [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/) and [model specs](https://vladmandic.github.io/sdnext-docs/Models/)  

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

- Get started with **SD.Next** by following the [installation instructions](https://vladmandic.github.io/sdnext-docs/Installation/)  
- For more details, check out [advanced installation](https://vladmandic.github.io/sdnext-docs/Advanced-Install/) guide  
- List and explanation of [command line arguments](https://vladmandic.github.io/sdnext-docs/CLI-Arguments/)
- Install walkthrough [video](https://www.youtube.com/watch?v=nWTnTyFTuAs)

> [!TIP]
> And for platform specific information, check out  
> [WSL](https://vladmandic.github.io/sdnext-docs/WSL/) | [Intel Arc](https://vladmandic.github.io/sdnext-docs/Intel-ARC/) | [DirectML](https://vladmandic.github.io/sdnext-docs/DirectML/) | [OpenVINO](https://vladmandic.github.io/sdnext-docs/OpenVINO/) | [ONNX & Olive](https://vladmandic.github.io/sdnext-docs/ONNX-Runtime/) | [ZLUDA](https://vladmandic.github.io/sdnext-docs/ZLUDA/) | [AMD ROCm](https://vladmandic.github.io/sdnext-docs/AMD-ROCm/) | [MacOS](https://vladmandic.github.io/sdnext-docs/MacOS-Python/) | [nVidia](https://vladmandic.github.io/sdnext-docs/nVidia/) | [Docker](https://vladmandic.github.io/sdnext-docs/Docker/)

> [!WARNING]
> If you run into issues, check out [troubleshooting](https://vladmandic.github.io/sdnext-docs/Troubleshooting/) and [debugging](https://vladmandic.github.io/sdnext-docs/Debug/) guides  

> [!TIP]
> All command line options can also be set via env variable  
> For example `--debug` is same as `set SD_DEBUG=true`  

### Credits

- Main credit goes to [Automatic1111 WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for the original codebase  
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

If you're unsure how to use a feature, best place to start is [Docs](https://vladmandic.github.io/sdnext-docs/) and if its not there,  
check [ChangeLog](https://vladmandic.github.io/sdnext-docs/CHANGELOG/) for when feature was first introduced as it will always have a short note on how to use it  

### Sponsors

<div align="center">
<!-- sponsors --><a href="https://github.com/allangrant"><img src="https://github.com/allangrant.png" width="60px" alt="Allan Grant" /></a><a href="https://github.com/BrentOzar"><img src="https://github.com/BrentOzar.png" width="60px" alt="Brent Ozar" /></a><a href="https://github.com/mantzaris"><img src="https://github.com/mantzaris.png" width="60px" alt="a.v.mantzaris" /></a><a href="https://github.com/CurseWave"><img src="https://github.com/CurseWave.png" width="60px" alt="" /></a><a href="https://github.com/smlbiobot"><img src="https://github.com/smlbiobot.png" width="60px" alt="SML (See-ming Lee)" /></a><!-- sponsors -->
</div>

<br>
