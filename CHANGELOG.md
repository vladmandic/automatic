# Change Log for SD.Next

## Update for 2025-01-01

- [Allegro Video](https://huggingface.co/rhymes-ai/Allegro)  
  - optimizations: full offload and quantization support  
  - *reference values*: width 1280 height 720 frames 88 steps 100 guidance 7.5  
  - *note*: allegro model is really sensitive to input width/height/frames/steps  
    and may result in completely corrupt output if those are not within expected range  
- **Logging**:
  - reverted enable debug by default  
  - updated [debug wiki](https://github.com/vladmandic/automatic/wiki/debug)  
  - sort logged timers by duration  
  - allow min duration env variable for timers: `SD_MIN_TIMER=0.1` (default)  
  - update installer messages  
- **Detailer**:
  - add explicit detailer steps setting  
- **Fixes**:
  - explict clear caches on model load  
  - lock adetailer commit: `#a89c01d`  
  - xyzgrid fix progress calculation  
  - vae tiling use default value if not set  

## Update for 2024-12-31

NYE refresh release with quite a few optimizatios and bug fixes...  
Commit hash: `master: #dcfc9f3` `dev: #935cac6`  

- **LoRA**:  
  - LoRA load/apply/unapply methods have been changed in 12/2024 Xmass release and further tuned in this release
  - for details on available methods, see <https://github.com/vladmandic/automatic/wiki/Lora#lora-loader>  
  - **Sana** support  
  - quantized models support  
  - add fuse support with on-demand apply/unapply (new default)  
  - add legacy option in *settings -> networks*  
- **HunyuanVideo**:  
  - optimizations: full offload, quantization and tiling support  
- **LTXVideo**:  
  - optimizations: full offload, quantization and tiling support  
  - [TeaCache](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4LTX-Video/README.md) integration  
- **VAE**:  
  - tiling granular options in *settings -> variable auto encoder*  
- **UI**:  
  - live preview optimizations and error handling  
  - live preview high quality output, thanks @Disty0  
  - CSS optimizations when log view is disabled  
- **Samplers**:  
  - add flow shift options and separate dynamic thresholding from dynamic shifting  
  - autodetect matching sigma capabilities  
- **API**:  
  - better default values for generate  
- **Refactor**:  
  - remove all LDM imports if running in native mode  
  - startup optimizatios  
- **Torch**:  
  - support for `torch==2.6.0`  
- **OpenVINO**:  
  - disable re-compile on resolution change  
  - fix shape mismatch on resolution change  
- **Fixes**:  
  - flux pipeline switches: txt/img/inpaint  
  - flux custom unet loader for bnb  
  - flux do not requantize already quantized model
  - interrogate caption with T5  
  - on-the-fly quantization using TorchAO  
  - remove concurrent preview requests  
  - xyz grid recover on error  
  - hires batch  
  - sdxl refiner  
  - increase progress timeout
  - kandinsky matmul  
  - do not show disabled networks  
  - enable debug logging by default
  - image width/height calculation when doing img2img  
  - corrections with batch processing  
  - hires with refiner prompt and batch processing  
  - processing with nested calls  
  - ui networks initial sort  
  - esrgan on cpu devices  

## Update for 2024-12-24

### Highlights for 2024-12-24

### SD.Next Xmass edition: *What's new?*

While we have several new supported models, workflows and tools, this release is primarily about *quality-of-life improvements*:  
- New memory management engine  
  list of changes that went into this one is long: changes to GPU offloading, brand new LoRA loader, system memory management, on-the-fly quantization, improved gguf loader, etc.  
  but main goal is enabling modern large models to run on standard consumer GPUs  
  without performance hits typically associated with aggressive memory swapping and needs for constant manual tweaks  
- New [documentation website](https://vladmandic.github.io/sdnext-docs/)  
  with full search and tons of new documentation  
- New settings panel with simplified and streamlined configuration  

We've also added support for several new models such as highly anticipated [NVLabs Sana](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px) (see [supported models](https://vladmandic.github.io/sdnext-docs/Model-Support/) for full list)  
And several new SOTA video models: [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video), [Hunyuan Video](https://huggingface.co/tencent/HunyuanVideo) and [Genmo Mochi.1 Preview](https://huggingface.co/genmo/mochi-1-preview)  

And a lot of **Control** and **IPAdapter** goodies  
- for **SDXL** there is new [ProMax](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0), improved *Union* and *Tiling* models 
- for **FLUX.1** there are [Flux Tools](https://blackforestlabs.ai/flux-1-tools/) as well as official *Canny* and *Depth* models,  
  a cool [Redux](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) model as well as [XLabs](https://huggingface.co/XLabs-AI/flux-ip-adapter-v2) IP-adapter
- for **SD3.5** there are official *Canny*, *Blur* and *Depth* models in addition to existing 3rd party models  
  as well as [InstantX](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter) IP-adapter  

Plus couple of new integrated workflows such as [FreeScale](https://github.com/ali-vilab/FreeScale) and [Style Aligned Image Generation](https://style-aligned-gen.github.io/)  

And it wouldn't be a *Xmass edition* without couple of custom themes: *Snowflake* and *Elf-Green*!  
All-in-all, we're around ~180 commits worth of updates, check the changelog for full list  

[ReadMe](https://github.com/vladmandic/automatic/blob/master/README.md) | [ChangeLog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [Docs](https://vladmandic.github.io/sdnext-docs/) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

## Details for 2024-12-24

### New models and integrations

- [NVLabs Sana](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px)
  support for 1.6B 2048px, 1.6B 1024px and 0.6B 512px models  
  **Sana** can synthesize high-resolution images with strong text-image alignment by using **Gemma2** as text-encoder  
  and its *fast* - typically at least **2x** faster than sd-xl even for 1.6B variant and maintains performance regardless of resolution  
  e.g., rendering at 4k is possible in less than 8GB vram  
  to use, select from *networks -> models -> reference* and models will be auto-downloaded on first use  
  *reference values*: sampler: default (or any flow-match variant), steps: 20, width/height: 1024, guidance scale: 4.5  
  *note* like other LLM-based text-encoders, sana prefers long and descriptive prompts  
  any short prompt below 300 characters will be auto-expanded using built in Gemma LLM before encoding while long prompts will be passed as-is  
- **ControlNet**
  - improved support for **Union** controlnets with granular control mode type
  - added support for latest [Xinsir ProMax](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) all-in-one controlnet  
  - added support for multiple **Tiling** controlnets, for example [Xinsir Tile](https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0)  
    *note*: when selecting tiles in control settings, you can also specify non-square ratios  
    in which case it will use context-aware image resize to maintain overall composition  
    *note*: available tiling options can be set in settings -> control  
- **IP-Adapter**  
  - FLUX.1 [XLabs](https://huggingface.co/XLabs-AI/flux-ip-adapter-v2) v1 and v2 IP-adapter  
  - FLUX.1 secondary guidance, enabled using *Attention guidance* in advanced menu  
  - SD 3.5 [InstantX](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter) IP-adapter  
- [Flux Tools](https://blackforestlabs.ai/flux-1-tools/)  
  **Redux** is actually a tool, **Fill** is inpaint/outpaint optimized version of *Flux-dev*  
  **Canny** & **Depth** are optimized versions of *Flux-dev* for their respective tasks: they are *not* ControlNets that work on top of a model  
  to use, go to image or control interface and select *Flux Tools* in scripts  
  all models are auto-downloaded on first use  
  *note*: All models are [gated](https://github.com/vladmandic/automatic/wiki/Gated) and require acceptance of terms and conditions via web page  
  *recommended*: Enable on-the-fly [quantization](https://github.com/vladmandic/automatic/wiki/Quantization) or [compression](https://github.com/vladmandic/automatic/wiki/NNCF-Compression) to reduce resource usage  
  *todo*: support for Canny/Depth LoRAs  
  - [Redux](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev): ~0.1GB  
    works together with existing model and basically uses input image to analyze it and use that instead of prompt  
    *optional* can use prompt to combine guidance with input image  
    *recommended*: low denoise strength levels result in more variety  
  - [Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev): ~23.8GB, replaces currently loaded model  
    *note*: can be used in inpaint/outpaint mode only  
  - [Canny](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev): ~23.8GB, replaces currently loaded model  
    *recommended*: guidance scale 30  
  - [Depth](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev): ~23.8GB, replaces currently loaded model  
    *recommended*: guidance scale 10  
- [Flux ControlNet LoRA](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora)  
  alternative to standard ControlNets, FLUX.1 also allows LoRA to help guide the generation process  
  both **Depth** and **Canny** LoRAs are available in standard control menus  
- [StabilityAI SD35 ControlNets](https://huggingface.co/stabilityai/stable-diffusion-3.5-controlnets)
  - In addition to previously released `InstantX` and `Alimama`, we now have *official* ones from StabilityAI  
- [Style Aligned Image Generation](https://style-aligned-gen.github.io/)  
  enable in scripts, compatible with sd-xl  
  enter multiple prompts in prompt field separated by new line  
  style-aligned applies selected attention layers uniformly to all images to achive consistency  
  can be used with or without input image in which case first prompt is used to establish baseline  
  *note:* all prompts are processes as a single batch, so vram is limiting factor  
- [FreeScale](https://github.com/ali-vilab/FreeScale)  
  enable in scripts, compatible with sd-xl for text and img2img  
  run iterative generation of images at different scales to achieve better results  
  can render 4k sdxl images  
  *note*: disable live preview to avoid memory issues when generating large images  

### Video models

- [Lightricks LTX-Video](https://huggingface.co/Lightricks/LTX-Video)  
  model size: 27.75gb  
  support for 0.9.0, 0.9.1 and custom safetensor-based models with full quantization and offloading support  
  support for text-to-video and image-to-video, to use, select in *scripts -> ltx-video*  
  *refrence values*: steps 50, width 704, height 512, frames 161, guidance scale 3.0  
- [Hunyuan Video](https://huggingface.co/tencent/HunyuanVideo)  
  model size: 40.92gb  
  support for text-to-video, to use, select in *scripts -> hunyuan video*  
  basic support only  
  *refrence values*: steps 50, width 1280, height 720, frames 129, guidance scale 6.0  
- [Genmo Mochi.1 Preview](https://huggingface.co/genmo/mochi-1-preview)  
  support for text-to-video, to use, select in *scripts -> mochi.1 video*  
  basic support only  
  *refrence values*: steps 64, width 848, height 480, frames 19, guidance scale 4.5  

*Notes*:
- all video models are very large and resource intensive!  
  any use on gpus below 16gb and systems below 48gb ram is experimental at best  
- sdnext support for video models is relatively basic with further optimizations pending community interest  
  any future optimizations would likely have to go into partial loading and excecution instead of offloading inactive parts of the model  
- new video models use generic llms for prompting and due to that requires very long and descriptive prompt  
- you may need to enable sequential offload for maximum gpu memory savings  
- optionally enable pre-quantization using bnb for additional memory savings  
- reduce number of frames and/or resolution to reduce memory usage  

### UI and workflow improvements

- **Docs**:
  - New documentation site! <https://vladmandic.github.io/sdnext-docs/>
  - Additional Wiki content: Styles, Wildcards, etc.
- **LoRA** handler rewrite:  
  - LoRA weights are no longer calculated on-the-fly during model execution, but are pre-calculated at the start  
    this results in perceived overhead on generate startup, but results in overall faster execution as LoRA does not need to be processed on each step  
    thanks @AI-Casanova  
  - LoRA weights can be applied/unapplied as on each generate or they can store weights backups for later use  
    this setting has large performance and resource implications, see [Offload](https://github.com/vladmandic/automatic/wiki/Offload) wiki for details  
  - LoRA name in prompt can now also be an absolute path to a LoRA file, even if LoRA is not indexed  
    example: `<lora:/test/folder/my-lora.safetensors:1.0>`
  - LoRA name in prompt can now also be path to a LoRA file op `huggingface`  
    example: `<lora:/huggingface.co/vendor/repo/my-lora.safetensors:1.0>`
- **Model loader** improvements:  
  - detect model components on model load fail  
  - allow passing absolute path to model loader  
  - Flux, SD35: force unload model  
  - Flux: apply `bnb` quant when loading *unet/transformer*  
  - Flux: all-in-one safetensors  
    example: <https://civitai.com/models/646328?modelVersionId=1040235>  
  - Flux: do not recast quants  
- **Memory** improvements:  
  - faster and more compatible *balanced offload* mode  
  - balanced offload: units are now in percentage instead of bytes  
  - balanced offload: add both high and low watermark, defaults as below  
    `0.25` for low-watermark: skip offload if memory usage is below 25%  
    `0.70` high-watermark: must offload if memory usage is above 70%  
  - balanced offload will attempt to run offload as non-blocking and force gc at the end  
  - change-in-behavior:  
    low-end systems, triggered by either `lowvrwam` or by detection of <=4GB will use *sequential offload*  
    all other systems use *balanced offload* by default (can be changed in settings)  
    previous behavior was to use *model offload* on systems with <=8GB and `medvram` and no offload by default  
  - VAE upcase is now disabled by default on all systems  
    if you have issues with image decode, you'll need to enable it manually  
- **UI**:  
  - improved stats on generate completion  
  - improved live preview display and performance  
  - improved accordion behavior  
  - auto-size networks height for sidebar  
  - control: hide preview column by default
  - control: optionn to hide input column
  - control: add stats
  - settings: reorganized and simplified  
  - browser -> server logging framework  
  - add addtional themes: `black-reimagined`, thanks @Artheriax  
- **Batch**
  - image batch processing will use caption files if they exist instead of default prompt  

### Updates

- **Quantization**
  - Add `TorchAO` *pre* (during load) and *post* (during execution) quantization  
    **torchao** supports 4 different int-based and 3 float-based quantization schemes  
  This is in addition to existing support for:  
  - `BitsAndBytes` with 3 float-based quantization schemes  
  - `Optimium.Quanto` with 3 int-based and 2 float-based quantizations schemes  
  - `GGUF` with pre-quantized weights  
  - Switch `GGUF` loader from custom to diffuser native
- **IPEX**: update to IPEX 2.5.10+xpu  
- **OpenVINO**:  
  - update to 2024.6.0  
  - disable model caching by default  
- **Sampler** improvements  
  - UniPC, DEIS, SA, DPM-Multistep: allow FlowMatch sigma method and prediction type  
  - Euler FlowMatch: add sigma methods (*karras/exponential/betas*)  
  - Euler FlowMatch: allow using timestep presets to set sigmas  
  - DPM FlowMatch: update all and add sigma methods  
  - BDIA-DDIM: *experimental* new scheduler  
  - UFOGen: *experimental* new scheduler  

### Fixes  

- add `SD_NO_CACHE=true` env variable to disable file/folder caching  
- add settings -> networks -> embeddings -> enable/disable
- update `diffusers`  
- fix README links  
- fix sdxl controlnet single-file loader  
- relax settings validator  
- improve js progress calls resiliency  
- fix text-to-video pipeline  
- avoid live-preview if vae-decode is running  
- allow xyz-grid with multi-axis s&r  
- fix xyz-grid with lora  
- fix api script callbacks  
- fix gpu memory monitoring  
- simplify img2img/inpaint/sketch canvas handling  
- fix prompt caching  
- fix xyz grid skip final pass  
- fix sd upscale script  
- fix cogvideox-i2v  
- lora auto-apply tags remove duplicates  
- control load model on-demand if not already loaded  
- taesd limit render to 2024px  
- taesd downscale preview to 1024px max: configurable in settings -> live preview  
- uninstall conflicting `wandb` package  
- dont skip diffusers version check if quick is specified  
- notify on torch install  
- detect pipeline fro diffusers folder-style model  
- do not recast flux quants  
- fix xyz-grid with lora none  
- fix svd image2video  
- fix gallery display during generate  
- fix wildcards replacement to be unique  
- fix animatediff-xl  
- fix pag with batch count  

## Update for 2024-11-21

### Highlights for 2024-11-21

Three weeks is a long time in Generative AI world - and we're back with ~140 commits worth of updates!

*What's New?*

First, a massive update to docs including new UI top-level **info** tab with access to [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) and [wiki](https://github.com/vladmandic/automatic/wiki), many updates and new articles AND full **built-in documentation search** capabilities

#### New integrations

- [PuLID](https://github.com/ToTheBeginning/PuLID): Pure and Lightning ID Customization via Contrastive Alignment
- [InstantX InstantIR](https://github.com/instantX-research/InstantIR): Blind Image Restoration with Instant Generative Reference
- [nVidia Labs ConsiStory](https://github.com/NVlabs/consistory): Consistent Image Generation
- [MiaoshouAI PromptGen v2.0](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0) VQA captioning

#### Workflow Improvements

- Native **Docker** support
- **SD3x & Flux.1**: more ControlNets, all-in-one-safetensors, DPM samplers, skip-layer-guidance, etc.
- **XYZ grid**: benchmarking, video creation, etc.
- Enhanced **prompt** parsing
- **UI** improvements
- **Installer** self-healing `venv`

And quite a few more improvements and fixes since the last update!
For full list and details see changelog...

[README](https://github.com/vladmandic/automatic/blob/master/README.md) | [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2024-11-21

- Docs:  
  - new top-level **info** tab with access to [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) and [wiki](https://github.com/vladmandic/automatic/wiki)  
  - UI built-in [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) search  
    since changelog is the best up-to-date source of info  
    go to info -> changelog and search/highligh/navigate directly in UI!  
  - UI built-in [wiki](https://github.com/vladmandic/automatic/wiki)  
    go to info -> wiki and search wiki pages directly in UI!  
  - major [Wiki](https://github.com/vladmandic/automatic/wiki) and [Home](https://github.com/vladmandic/automatic) updates  
  - updated API swagger docs for at `/docs`  
- Integrations:  
  - [PuLID](https://github.com/ToTheBeginning/PuLID): Pure and Lightning ID Customization via Contrastive Alignment  
    - advanced method of face id transfer with better quality as well as control over identity and appearance  
      try it out, likely the best quality available for sdxl models  
    - select in *scripts -> pulid*  
    - compatible with *sdxl* for text-to-image, image-to-image, inpaint, refine, detailer workflows  
    - can be used in xyz grid  
    - *note*: this module contains several advanced features on top of original implementation  
  - [InstantIR](https://github.com/instantX-research/InstantIR): Blind Image Restoration with Instant Generative Reference  
    - alternative to traditional `img2img` with more control over restoration process  
    - select in *image -> scripts -> instantir*  
    - compatible with *sdxl*  
    - *note*: after used once it cannot be unloaded without reloading base model  
  - [ConsiStory](https://github.com/NVlabs/consistory): Consistent Image Generation  
    - create consistent anchor image and then generate images that are consistent with anchor  
    - select in *scripts -> consistory*  
    - compatible with *sdxl*  
    - *note*: very resource intensive and not compatible with model offloading  
    - *note*: changing default parameters can lead to unexpected results and/or failures  
    - *note*: after used once it cannot be unloaded without reloading base model  
  - [MiaoshouAI PromptGen v2.0](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v2.0) base and large:  
    - *in process -> visual query*  
    - caption modes:  
      `<GENERATE_TAGS>` generate tags  
      `<CAPTION>`, `<DETAILED_CAPTION>`, `<MORE_DETAILED_CAPTION>` caption image  
      `<ANALYZE>` image composition  
      `<MIXED_CAPTION>`, `<MIXED_CAPTION_PLUS>` detailed caption and tags with optional analyze  

- Model improvements:  
  - SD35: **ControlNets**:  
    - *InstantX Canny, Pose, Depth, Tile*  
    - *Alimama Inpainting, SoftEdge*  
    - *note*: that just like with FLUX.1 or any large model, ControlNet are also large and can push your system over the limit  
      e.g. SD3 controlnets vary from 1GB to over 4GB in size  
  - SD35: **All-in-one** safetensors  
    - *examples*: [large](https://civitai.com/models/882666/sd35-large-google-flan?modelVersionId=1003031), [medium](https://civitai.com/models/900327)  
    - *note*: enable *bnb* on-the-fly quantization for even bigger gains  
  - SD35: **skip-layer-guidance**  
    - enable in *scripts -> slg*
    - allows for granular strength/start/stop control of guidance for each layer of the model  
  - [NoobAI XL ControlNets](https://huggingface.co/collections/Eugeoter/controlnext-673161eae023f413e0432799), thanks @lbeltrame

- Workflow improvements:  
  - Native Docker support with pre-defined [Dockerfile](https://github.com/vladmandic/automatic/blob/dev/Dockerfile)
  - Samplers:
    - **FlowMatch samplers**:
      - Applicable to SD 3.x and Flux.1 models
      - Complete family: *DPM2, DPM2a, DPM2++, DPM2++ 2M, DPM2++ 2S, DPM2++ SDE, DPM2++ 2M SDE, DPM2++ 3M SDE*
    - **Beta and Exponential** sigma method enabled for all samplers
  - **XYZ grid**:  
    - optional time benchmark info to individual images  
    - optional add params to individual images  
    - create video from generated grid images  
      supports all standard video types and interpolation  
  - **Prompt parser**:  
    - support for prompt scheduling  
    - renamed parser options: `native`, `xhinker`, `compel`, `a1111`, `fixed`  
    - parser options are available in xyz grid  
    - improved caching  
  - **UI**:  
    - better gallery and networks sidebar sizing  
    - add additional [hotkeys](https://github.com/vladmandic/automatic/wiki/Hotkeys)  
    - add show networks on startup setting  
    - better mapping of networks previews  
    - optimize networks display load  
  - Image2image:  
    - integrated refine/upscale/hires workflow  
- Other:  
  - **Installer**:  
    - Log `venv` and package search paths  
    - Auto-remove invalid packages from `venv/site-packages`  
      e.g. packages starting with `~` which are left-over due to windows access violation  
    - Requirements: update  
  - Scripts:  
    - More verbose descriptions for all scripts  
  - Model loader:  
    - Report modules included in safetensors when attempting to load a model  
  - CLI:  
    - refactor command line params  
      run `webui.sh`/`webui.bat` with `--help` to see all options  
    - added `cli/model-metadata.py` to display metadata in any safetensors file  
    - added `cli/model-keys.py` to quicky display content of any safetensors file  
  - Internal:  
    - Auto pipeline switching coveres wrapper classes and nested pipelines  
    - Full settings validation on load of `config.json`  
    - Refactor of all params in main processing classes  
    - Improve API scripts usage resiliency  

- Fixes:  
  - custom watermark add alphablending  
  - fix xyz grid include images  
  - fix xyz skip on interrupted  
  - fix vqa models ignoring hfcache folder setting  
  - fix network height in standard vs modern ui  
  - fix k-diff enum on startup  
  - fix text2video scripts  
  - multiple xyz-grid fixes  
  - dont uninstall flash-attn  
  - ui css fixes  

## Update for 2024-11-01

Smaller release just 3 days after the last one, but with some important fixes and improvements.  
This release can be considered an LTS release before we kick off the next round of major updates.  

- Other:
  - Repo: move screenshots to GH pages
  - Update requirements
- Fixes:
  - detailer min/max size as fractions of image size  
  - ipadapter load on-demand  
  - ipadapter face use correct yolo model  
  - list diffusers remove duplicates  
  - fix legacy extensions access to shared objects  
  - fix diffusers load from folder  
  - fix lora enum logging on windows  
  - fix xyz grid with batch count  
  - move dowwloads of some auxillary models to hfcache instead of models folder  

## Update for 2024-10-29

### Highlights for 2024-10-29

- Support for **all SD3.x variants**  
  *SD3.0-Medium, SD3.5-Medium, SD3.5-Large, SD3.0-Large-Turbo*
- Allow quantization using `bitsandbytes` on-the-fly during models load
  Load any variant of SD3.x or FLUX.1 and apply quantization during load without the need for pre-quantized models  
- Allow for custom model URL in standard model selector  
  Can be used to specify any model from *HuggingFace* or *CivitAI*  
- Full support for `torch==2.5.1`
- New wiki articles: [Gated Access](https://github.com/vladmandic/automatic/wiki/Gated), [Quantization](https://github.com/vladmandic/automatic/wiki/Quantization), [Offloading](https://github.com/vladmandic/automatic/wiki/Offload)  

Plus tons of smaller improvements and cumulative fixes reported since last release  

[README](https://github.com/vladmandic/automatic/blob/master/README.md) | [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2024-10-29

- model selector:
  - change-in-behavior
  - when typing, it will auto-load model as soon as exactly one match is found
  - allows entering model that are not on the list which triggers huggingface search  
    e.g. `stabilityai/stable-diffusion-xl-base-1.0`  
    partial search hits are displayed in the log  
    if exact model is found, it will be auto-downloaded and loaded  
  - allows entering civitai direct download link which triggers model download  
    e.g. `https://civitai.com/api/download/models/72396?type=Model&format=SafeTensor&size=full&fp=fp16`  
  - auto-search-and-download can be disabled in settings -> models -> auto-download  
    this also disables reference models as they are auto-downloaded on first use as well  
- sd3 enhancements:  
  - allow on-the-fly bnb quantization during load
  - report when loading incomplete model  
  - handle missing model components during load  
  - handle component preloading  
  - native lora handler  
  - support for all sd35 variants: *medium/large/large-turbo*
  - gguf transformer loader (prototype)  
- flux.1 enhancements:  
  - allow on-the-fly bnb quantization during load
- samplers:
  - support for original k-diffusion samplers  
    select via *scripts -> k-diffusion -> sampler*  
- ipadapter:
  - list available adapters based on loaded model type
  - add adapter `ostris consistency` for sd15/sdxl
- detailer:
  - add `[prompt]` to refine/defailer prompts as placeholder referencing original prompt  
- torch
  - use `torch==2.5.1` by default on supported platforms
  - CUDA set device memory limit
    in *settings -> compute settings -> torch memory limit*  
    default=0 meaning no limit, if set torch will limit memory usage to specified fraction  
    *note*: this is not a hard limit, torch will try to stay under this value  
- compute backends:
  - OpenVINO: add accuracy option  
  - ZLUDA: guess GPU arch  
- major model load refactor
- wiki: new articles
  - [Gated Access Wiki](https://github.com/vladmandic/automatic/wiki/Gated)  
  - [Quantization Wiki](https://github.com/vladmandic/automatic/wiki/Quantization)  
  - [Offloading Wiki](https://github.com/vladmandic/automatic/wiki/Offload)  

fixes:  
- fix send-to-control  
- fix k-diffusion  
- fix sd3 img2img and hires  
- fix ipadapter supported model detection  
- fix t2iadapter auto-download
- fix omnigen dynamic attention  
- handle a1111 prompt scheduling  
- handle omnigen image placeholder in prompt  

## Update for 2024-10-23

### Highlights for 2024-10-23

A month later and with nearly 300 commits, here is the latest [SD.Next](https://github.com/vladmandic/automatic) update!  

#### Workflow highlights for 2024-10-23

- **Reprocess**: New workflow options that allow you to generate at lower quality and then  
  reprocess at higher quality for select images only or generate without hires/refine and then reprocess with hires/refine  
  and you can pick any previous latent from auto-captured history!  
- **Detailer** Fully built-in detailer workflow with support for all standard models  
- Built-in **model analyzer**  
  See all details of your currently loaded model, including components, parameter count, layer count, etc.  
- **Extract LoRA**: load any LoRA(s) and play with generate as usual  
  and once you like the results simply extract combined LoRA for future use!  

#### New models for 2024-10-23

- New fine-tuned [CLiP-ViT-L](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14) 1st stage **text-encoders** used by most models (SD15/SDXL/SD3/Flux/etc.) brings additional details to your images  
- New models:  
  [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)  
  [OmniGen](https://arxiv.org/pdf/2409.11340)  
  [CogView 3 Plus](https://huggingface.co/THUDM/CogView3-Plus-3B)  
  [Meissonic](https://github.com/viiika/Meissonic)  
- Additional integration:  
  [Ctrl+X](https://github.com/genforce/ctrl-x) which allows for control of **structure and appearance** without the need for extra models,  
  [APG: Adaptive Projected Guidance](https://arxiv.org/pdf/2410.02416) for optimal **guidance** control,  
  [LinFusion](https://github.com/Huage001/LinFusion) for on-the-fly **distillation** of any sd15/sdxl model  

#### What else for 2024-10-23

- Tons of work on **dynamic quantization** that can be applied *on-the-fly* during model load to any model type (*you do not need to use pre-quantized models*)  
  Supported quantization engines include `BitsAndBytes`, `TorchAO`, `Optimum.quanto`, `NNCF` compression, and more...  
- Auto-detection of best available **device/dtype** settings for your platform and GPU reduces neeed for manual configuration  
  *Note*: This is a breaking change to default settings and its recommended to check your preferred settings after upgrade  
- Full rewrite of **sampler options**, not far more streamlined with tons of new options to tweak scheduler behavior  
- Improved **LoRA** detection and handling for all supported models  
- Several of [Flux.1](https://huggingface.co/black-forest-labs/FLUX.1-dev) optimizations and new quantization types  

Oh, and we've compiled a full table with list of top-30 (*how many have you tried?*) popular text-to-image generative models,  
their respective parameters and architecture overview: [Models Overview](https://github.com/vladmandic/automatic/wiki/Models)  

And there are also other goodies like multiple *XYZ grid* improvements, additional *Flux ControlNets*, additional *Interrogate models*, better *LoRA tags* support, and more...  
[README](https://github.com/vladmandic/automatic/blob/master/README.md) | [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) | [WiKi](https://github.com/vladmandic/automatic/wiki) | [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867)

### Details for 2024-10-23

- **reprocess**
  - new top-level button: reprocess latent from your history of generated image(s)  
  - generate using full-quality:off and then reprocess using *full quality decode*  
  - generate without hires/refine and then *reprocess with hires/refine*  
    *note*: you can change hires/refine settings and run-reprocess again!  
  - reprocess using *detailer*  

- **history**
  - by default, **reprocess** will pick last latent, but you can select any latent from history!  
  - history is under *networks -> history*  
    each history item includes info on operations that were used, timestamp and metadata  
  - any latent operation during workflow automatically adds one or more items to history  
    e.g. generate base + upscale + hires + detailer  
  - history size: *settings -> execution -> latent history size*  
    memory usage is ~130kb of ram for 1mp image  
  - *note* list of latents in history is not auto-refreshed, use refresh button  

- **model analyzer**  
  - see all details of your currently loaded model, including components, parameter count, layer count, etc.  
  - in models -> current -> analyze  

- **text encoder**:  
  - allow loading different custom text encoders: *clip-vit-l, clip-vit-g, t5*  
    will automatically find appropriate encoder in the loaded model and replace it with loaded text encoder  
    download text encoders into folder set in settings -> system paths -> text encoders  
    default `models/Text-encoder` folder is used if no custom path is set  
    finetuned *clip-vit-l* models: [Detailed, Smooth](https://huggingface.co/zer0int/CLIP-GmP-ViT-L-14), [LongCLIP](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14)  
    reference *clip-vit-l* and *clip-vit-g* models: [OpenCLIP-Laion2b](https://huggingface.co/collections/laion/openclip-laion-2b-64fcade42d20ced4e9389b30)  
    *note* sd/sdxl contain heavily distilled versions of reference models, so switching to reference model produces vastly different results  
  - xyz grid support for text encoder  
  - full prompt parser now correctly works with different prompts in batch  

- **detailer**:  
  - replaced *face-hires* with *detailer* which can run any number of standard detailing models  
  - includes *face/hand/person/eyes* predefined detailer models plus support for manually downloaded models  
    set path in *settings -> system paths -> yolo*  
  - select one or more models in detailer menu and thats it!  
  - to avoid duplication of ui elements, detailer will use following values from **refiner**:  
    *sampler, steps, prompts*  
  - when using multiple detailers and prompt is *multi-line*, each line is applied to corresponding detailer  
  - adjustable settings:  
    *strength, max detected objects, edge padding, edge blur, min detection confidence, max detection overlap, min and max size of detected object*  
  - image metadata includes info on used detailer models  
  - *note* detailer defaults are not save in ui settings, they are saved in server settings  
    to apply your defaults, set ui values and apply via *system -> settings -> apply settings*  
  - if using models trained on multiple classes, you can specify which classes you want to detail  
    e.g. original yolo detection model is trained on coco dataset with 80 predefined classes  
    if you leave field blank, it will use any class found in the model  
    you can see classes defined in the model while model itself is loaded for the first time  

- **extract lora**: extract combined lora from current memory state, thanks @AI-Casanova  
  load any LoRA(s) and play with generate as usual and once you like the results simply extract combined LoRA for future use!  
  in *models -> extract lora*  

- **sampler options**: full rewrite  

  *sampler notes*:  
  - pick a sampler and then pick values, all values have "default" as a choice to make it simpler  
  - a lot of options are new, some are old but moved around  
    e.g. karras checkbox is replaced with a choice of different sigma methods  
  - not every combination of settings is valid  
  - some settings are specific to model types  
    e.g. sd15/sdxl typically use epsilon prediction  
  - quite a few well-known schedulers are just variations of settings, for example:  
    - *sampler sgm* is sampler with trailing spacing and sample prediction type  
    - *dpm 2m* or *3m* are *dpm 1s* with orders of 2 or 3  
    - *dpm 2m sde* is *dpm 2m* with *sde* as solver  
    - *sampler simple* is sampler with trailing spacing and linear beta schedule
  - xyz grid support for sampler options  
  - metadata updates for sampler options  
  - modernui updates for sampler options  
  - *note* sampler options defaults are not saved in ui settings, they are saved in server settings  
    to apply your defaults, set ui values and apply via *system -> settings -> apply settings*  

  *sampler options*:  
  - sigma method: *karas, beta, exponential*  
  - timesteps spacing: *linspace, leading, trailing*  
  - beta schedule: *linear, scaled, cosine*  
  - prediction type: *epsilon, sample, v-prediction*  
  - timesteps presents: *none, ays-sd15, ays-sdxl*  
  - timesteps override: <custom>  
  - sampler order: *0=default, 1-5*  
  - options: *dynamic, low order, rescale*  

- [Ctrl+X](https://github.com/genforce/ctrl-x):
  - control **structure** (*similar to controlnet*) and **appearance** (*similar to ipadapter*)  
    without the need for extra models, all via code feed-forwards!
  - can run in structure-only or appearance-only or both modes
  - when providing structure and appearance input images, its best to provide a short prompts describing them  
  - structure image can be *almost anything*: *actual photo, openpose-style stick man, 3d render, sketch, depth-map, etc.*  
    just describe what it is in a structure prompt so it can be de-structured and correctly applied  
  - supports sdxl in both txt2img and img2img, simply select from scripts

- [APG: Adaptive Projected Guidance](https://arxiv.org/pdf/2410.02416)
  - latest algo to provide better guidance for image generation, can be used instead of existing guidance rescale and/or PAG  
  - in addtion to stronger guidance and reduction of burn at high guidance values, it can also increase image details  
  - compatible with *sd15/sdxl/sc*  
  - select in scripts -> apg  
  - for low    cfg scale, use positive momentum: e.g. cfg=2 => momentum=0.6
  - for normal cfg scale, use negative momentum: e.g. cfg=6 => momentum=-0.3
  - for high   cfg scale, use neutral  momentum: e.g. cfg=10 => momentum=0.0

- [LinFusion](https://github.com/Huage001/LinFusion)  
  - apply liner distillation to during load to any sd15/sdxl model  
  - can reduce vram use for high resolutions and increase performance
  - *note*: use lower cfg scales as typical for distilled models  

- [Flux](https://huggingface.co/black-forest-labs/FLUX.1-dev)  
  - see [wiki](https://github.com/vladmandic/automatic/wiki/FLUX#quantization) for details on `gguf`  
  - support for `gguf` binary format for loading unet/transformer component  
  - support for `gguf` binary format for loading t5/text-encoder component: requires transformers pr  
  - additional controlnets: [JasperAI](https://huggingface.co/collections/jasperai/flux1-dev-controlnets-66f27f9459d760dcafa32e08) **Depth**, **Upscaler**, **Surface**, thanks @EnragedAntelope  
  - additional controlnets: [XLabs-AI](https://huggingface.co/XLabs-AI/flux-controlnet-hed-diffusers) **Canny**, **Depth**, **HED**  
  - mark specific unet as unavailable if load failed  
  - fix diffusers local model name parsing  
  - full prompt parser will auto-select `xhinker` for flux models  
  - controlnet support for img2img and inpaint (in addition to previous txt2img controlnet)  
  - allow separate vae load  
  - support for both kohya and onetrainer loras in native load mode for fp16/nf4/fp4, thanks @AI-Casanova  
  - support for differential diffusion  
  - added native load mode for qint8/qint4 models
  - avoid unet load if unchanged  

- [OmniGen](https://arxiv.org/pdf/2409.11340)  
  - Radical new model with pure LLM architecture based on Phi-3  
  - Select from *networks -> models -> reference*  
  - Can be used for text-to-image and image-to-image  
  - Image-to-image is *very* different, you need to specify in prompt what do you want to do  
    and add `|image|` placeholder where input image is used!  
    examples: `in |image| remove glasses from face`, `using depth map from |image|, create new image of a cute robot`  
  - Params used: prompt, steps, guidance scale for prompt guidance, refine guidance scale for image guidance  
    Recommended: guidance=3.0, refine-guidance=1.6  

- [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)  
  - New/improved variant of Stable Diffusion 3  
  - Select from *networks -> models -> reference*  
  - Available in standard and turbo variations  
  - *Note*: Access to to both variations of SD3.5 model is gated, you must accept the conditions and use HF login  

- [CogView 3 Plus](https://huggingface.co/THUDM/CogView3-Plus-3B)
  - Select from *networks -> models -> reference*  
  - resolution width and height can be from 512px to 2048px and must be divisible by 32  
  - precision: bf16 or fp32  
    fp16 is not supported due to internal model overflows  

- [Meissonic](https://github.com/viiika/Meissonic)  
  - Select from *networks -> models -> reference*  
  - Experimental as upstream implemenation code is unstable
  - Must set scheduler:default, generator:unset

- [SageAttention](https://github.com/thu-ml/SageAttention)  
  - new 8-bit attention implementation on top of SDP that can provide acceleration for some models, thanks @Disty0  
  - enable in *settings -> compute settings -> sdp options -> sage attention*
  - compatible with DiT-based models: e.g. *Flux.1, AuraFlow, CogVideoX*  
  - not compatible with UNet-based models, e.g. *SD15, SDXL*  

- **gpu**
  - previously `cuda_dtype` in settings defaulted to `fp16` if available  
  - now `cuda_type` defaults to **Auto** which executes `bf16` and `fp16` tests on startup and selects best available dtype  
    if you have specific requirements, you can still set to fp32/fp16/bf16 as desired  
    if you have gpu that incorrectly identifies bf16 or fp16 availablity, let us know so we can improve the auto-detection  
  - support for torch **expandable segments**  
    enable in *settings -> compute -> torch expandable segments*  
    can provide significant memory savings for some models  
    not enabled by default as its only supported on latest versions of torch and some gpus  

- **xyz grid** full refactor  
  - multi-mode: *selectable-script* and *alwayson-script*  
  - allow usage combined with other scripts  
  - allow **unet** selection  
  - allow passing **model args** directly:  
    allowed params will be checked against models call signature  
    example: `width=768; height=512, width=512; height=768`  
  - allow passing **processing args** directly:  
    params are set directly on main processing object and can be known or new params  
    example: `steps=10, steps=20; test=unknown`  
  - enable working with different resolutions  
    now you can adjust width/height in the grid just as any other param  
  - renamed options to include section name and adjusted cost of each option  
  - added additional metadata  

- **interrogate**  
  - add additional blip models: *blip-base, blip-large, blip-t5-xl, blip-t5-xxl, opt-2.7b, opt-6.7b*  
  - change default params for better memory utilization  
  - lock commits for miaoshouAI-promptgen  
  - add optional advanced params  
  - update logging  

- **lora** auto-apply tags to prompt  
  - controlled via *settings -> networks -> lora_apply_tags*  
    *0:disable, -1:all-tags, n:top-n-tags*  
  - uses tags from both model embedded data and civitai downloaded data  
  - if lora contains no tags, lora name itself will be used as a tag  
  - if prompt contains `_tags_` it will be used as placeholder for replacement, otherwise tags will be appended  
  - used tags are also logged and registered in image metadata  
  - loras are no longer filtered per detected type vs loaded model type as its unreliable  
  - loras display in networks now shows possible version in top-left corner  
  - correct using of `extra_networks_default_multiplier` if not scale is specified  
  - improve lora base model detection  
  - improve lora error handling and logging  
  - setting `lora_load_gpu` to load LoRA directly to GPU  
    *default*: true unless lovwram  

- **quantization**  
  - new top level settings group as we have quite a few quantization options now!  
    configure in *settings -> quantization*  
  - in addition to existing `optimum.quanto` and `nncf`, we now have `bitsandbytes` and `torchao`  
  - **bitsandbytes**: fp8, fp4, nf4  
    - quantization can be applied on-the-fly during model load  
    - currently supports `transformers` and `t5` in **sd3** and **flux**  
  - **torchao**: int8, int4, fp8, fp4, fpx  
    - configure in settings -> quantization  
    - can be applied to any model on-the-fly during load  

- **huggingface**:  
  - force logout/login on token change  
  - unified handling of cache folder: set via `HF_HUB` or `HF_HUB_CACHE` or via settings -> system paths  

- **cogvideox**:  
  - add support for *image2video* (in addition to previous *text2video* and *video2video*)  
  - *note*: *image2video* requires separate 5b model variant  

- **torch**  
  - due to numerous issues with torch 2.5.0 which was just released as stable, we are sticking with 2.4.1 for now  

- **backend=original** is now marked as in maintenance-only mode  
- **python 3.12** improved compatibility, automatically handle `setuptools`  
- **control**
  - persist/reapply units current state on server restart  
  - better handle size before/after metadata  
- **video** add option `gradio_skip_video` to avoid gradio issues with displaying generated videos  
- add support for manually downloaded diffusers models from huggingface  
- **ui**  
  - move checkboxes `full quality, tiling, hidiffusion` to advanced section  
  - hide token counter until tokens are known  
  - minor ui optimizations  
  - fix update infotext on image select  
  - fix imageviewer exif parser  
  - selectable info view in image viewer, thanks @ZeldaMaster501  
  - setting to enable browser autolaunch, thanks @brknsoul  
- **free-u** check if device/dtype are fft compatible and cast as necessary  
- **rocm**
  - additional gpu detection and auto-config code, thanks @lshqqytiger  
  - experimental triton backend for flash attention, thanks @lshqqytiger  
  - update to rocm 6.2, thanks @Disty0
- **directml**  
  - update `torch` to 2.4.1, thanks @lshqqytiger  
- **extensions**  
  - add mechanism to lock-down extension to specific working commit  
  - added `sd-webui-controlnet` and `adetailer` last-known working commits  
- **upscaling**  
  - interruptible operations
- **refactor**  
  - general lora apply/unapply process  
  - modularize main process loop  
  - massive log cleanup  
  - full lint pass  
  - improve inference mode handling  
  - unify quant lib loading  


## Update for 2024-09-13

### Highlights for 2024-09-13

Major refactor of [FLUX.1](https://blackforestlabs.ai/announcing-black-forest-labs/) support:  
- Full **ControlNet** support, better **LoRA** support, full **prompt attention** implementation  
- Faster execution, more flexible loading, additional quantization options, and more...  
- Added **image-to-image**, **inpaint**, **outpaint**, **hires** modes  
- Added workflow where FLUX can be used as **refiner** for other models  
- Since both *Optimum-Quanto* and *BitsAndBytes* libraries are limited in their platform support matrix,  
  try enabling **NNCF** for quantization/compression on-the-fly!  

Few image related goodies...  
- **Context-aware** resize that allows for *img2img/inpaint* even at massively different aspect ratios without distortions!
- **LUT Color grading** apply professional color grading to your images using industry-standard *.cube* LUTs!
- Auto **HDR** image create for SD and SDXL with both 16ch true-HDR and 8-ch HDR-effect images ;)  

And few video related goodies...  
- [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b) **2b** and **5b** variants  
  with support for *text-to-video* and *video-to-video*!  
- [AnimateDiff](https://github.com/guoyww/animatediff/) **prompt travel** and **long context windows**!  
  create video which travels between different prompts and at long video lengths!  

Plus tons of other items and fixes - see [changelog](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md) for details!  
Examples:
- Built-in prompt-enhancer, TAESD optimizations, new DC-Solver scheduler, global XYZ grid management, etc.  
- Updates to ZLUDA, IPEX, OpenVINO...

### Details for 2024-09-13

**Major refactor of FLUX.1 support:**
- allow configuration of individual FLUX.1 model components: *transformer, text-encoder, vae*  
  model load will load selected components first and then initialize model using pre-loaded components  
  components that were not pre-loaded will be downloaded and initialized as needed  
  as usual, components can also be loaded after initial model load  
  *note*: use of transformer/unet is recommended as those are flux.1 finetunes  
  *note*: manually selecting vae and text-encoder is not recommended  
  *note*: mix-and-match of different quantizations for different components can lead to unexpected errors  
  - transformer/unet is list of manually downloaded safetensors  
  - vae is list of manually downloaded safetensors  
  - text-encoder is list of predefined and manually downloaded text-encoders  
- **controlnet** support:
  support for **InstantX/Shakker-Labs** models including [Union-Pro](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union)
  note that flux controlnet models are large, up to 6.6GB on top of already large base model!  
  as such, you may need to use offloading:sequential which is not as fast, but uses far less memory  
  when using union model, you must also select control mode in the control unit  
  flux does not yet support *img2img* so to use controlnet, you need to set contronet input via control unit override  
- model support loading **all-in-one** safetensors  
  not recommended due to massive duplication of components, but added due to popular demand  
  each such model is 20-32GB in size vs ~11GB for typical unet fine-tune  
- improve logging, warn when attempting to load unet as base model  
- **refiner** support  
  FLUX.1 can be used as refiner for other models such as sd/sdxl  
  simply load sd/sdxl model as base and flux model as refiner and use as usual refiner workflow  
- **img2img**, **inpaint** and **outpaint** support  
  *note* flux may require higher denoising strength than typical sd/sdxl models  
  *note*: img2img is not yet supported with controlnet  
- transformer/unet support *fp8/fp4* quantization  
  this brings supported quants to: *nf4/fp8/fp4/qint8/qint4*
- vae support *fp16*  
- **lora** support additional training tools  
- **face-hires** support  
- support **fuse-qkv** projections  
  can speed up generate  
  enable via *settings -> compute -> fused projections*  

**Other improvements & Fixes:**
- [CogVideoX](https://huggingface.co/THUDM/CogVideoX-5b)  
  - support for both **2B** and **5B** variations  
  - support for both **text2video** and **video2video** modes
  - simply select in *scripts -> cogvideox*  
  - as with any video modules, includes additional frame interpolation using RIFE  
  - if init video is used, it will be automatically resized and interpolated to desired number of frames  
- **AnimateDiff**:  
  - **prompt travel**  
     create video which travels between different prompts at different steps!  
     example prompt:
      > 0: dog  
      > 5: cat  
      > 10: bird  
  - support for **v3** model (finally)  
  - support for **LCM** model  
  - support for **free-noise** rolling context window  
    allow for creation of much longer videos, automatically enabled if frames > 16  
- **Context-aware** image resize, thanks @AI-Casanova!  
  based on [seam-carving](https://github.com/li-plus/seam-carving)  
  allows for *img2img/inpaint* even at massively different aspect ratios without distortions!  
  simply select as resize method when using *img2img* or *control* tabs  
- **HDR** high-dynamic-range image create for SD and SDXL  
  create hdr images from in multiple exposures by latent-space modifications during generation  
  use via *scripts -> hdr*  
  option *save hdr images* creates images in standard 8bit/channel (hdr-effect) *and* 16bit/channel (full-hdr) PNG format  
  ui result is always 8bit/channel hdr-effect image plus grid of original images used to create hdr  
  grid image can be disabled via settings -> user interface -> show grid  
  actual full-hdr image is not displayed in ui, only optionally saved to disk  
- new scheduler: [DC Solver](https://github.com/wl-zhao/DC-Solver)  
- **color grading** apply professional color grading to your images  
  using industry-standard *.cube* LUTs!
  enable via *scripts -> color-grading*  
- **hires** workflow now allows for full resize options  
  not just limited width/height/scale  
- **xyz grid** is now availabe as both local and global script!
- **prompt enhance**: improve quality and/or verbosity of your prompts  
  simply select in *scripts -> prompt enhance*
  uses [gokaygokay/Flux-Prompt-Enhance](https://huggingface.co/gokaygokay/Flux-Prompt-Enhance) model  
- **decode**
  - auto-set upcast if first decode fails  
  - restore dtype on upcast  
- **taesd** configurable number of layers  
  can be used to speed-up taesd decoding by reducing number of ops  
  e.g. if generating 1024px image, reducing layers by 1 will result in preview being 512px  
  set via *settings -> live preview -> taesd decode layers*  
- **xhinker** prompt parser handle offloaded models  
- **control** better handle offloading  
- **upscale** will use resize-to if set to non-zero values over resize-by  
  applies to any upscale options, including refine workflow  
- **networks** add option to choose if mouse-over on network should attempt to fetch additional info  
  option:`extra_networks_fetch` enable/disable in *settings -> networks*  
- speed up some garbage collection ops  
- sampler settings add **dynamic shift**  
  used by flow-matching samplers to adjust between structure and details  
- sampler settings force base shift  
  improves quality of the flow-matching samplers  
- **t5** support manually downloaded models  
  applies to all models that use t5 transformer  
- **modern-ui** add override field  
- full **lint** updates  
- use `diffusers` from main branch, no longer tied to release  
- improve diffusers/transformers/huggingface_hub progress reporting  
- use unique identifiers for all ui components  
- **visual query** (a.ka vqa or vlm) added support for several models
  - [MiaoshouAI PromptGen 1.5 Base](https://huggingface.co/MiaoshouAI/Florence-2-base-PromptGen-v1.5)
  - [MiaoshouAI PromptGen 1.5 Large](https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5)
  - [CogFlorence 2.2 Large](https://huggingface.co/thwri/CogFlorence-2.2-Large)
- **modernui** update  
- **zluda** update to 3.8.4, thanks @lshqqytiger!
- **ipex** update to 2.3.110+xpu on linux, thanks @Disty0!
- **openvino** update to 2024.3.0, thanks @Disty0!
- update `requirements`
- fix **AuraFlow**  
- fix handling of model configs if offline config is not available  
- fix vae decode in backend original  
- fix model path typos  
- fix guidance end handler  
- fix script sorting  
- fix vae dtype during load  
- fix all ui labels are unique

## Update for 2024-08-31

### Highlights for 2024-08-31

Summer break is over and we are back with a massive update!  

Support for all of the new models:  
- [Black Forest Labs FLUX.1](https://blackforestlabs.ai/announcing-black-forest-labs/)  
- [AuraFlow 0.3](https://huggingface.co/fal/AuraFlow)  
- [AlphaVLLM Lumina-Next-SFT](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)  
- [Kwai Kolors](https://huggingface.co/Kwai-Kolors/Kolors)  
- [HunyuanDiT 1.2](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)  

What else? Just a bit... ;)  

New **fast-install** mode, new **Optimum Quanto** and **BitsAndBytes** based quantization modes, new **balanced offload** mode that dynamically offloads GPU<->CPU as needed, and more...  
And from previous service-pack: new **ControlNet-Union** *all-in-one* model, support for **DoRA** networks, additional **VLM** models, new **AuraSR** upscaler  

**Breaking Changes...**

Due to internal changes, you'll need to reset your **attention** and **offload** settings!  
But...For a good reason, new *balanced offload* is magic when it comes to memory utilization while sacrificing minimal performance!

### Details for 2024-08-31

**New Models...**

To use and of the new models, simply select model from *Networks -> Reference* and it will be auto-downloaded on first use  

- [Black Forest Labs FLUX.1](https://blackforestlabs.ai/announcing-black-forest-labs/)  
  FLUX.1 models are based on a hybrid architecture of multimodal and parallel diffusion transformer blocks, scaled to 12B parameters and builing on flow matching  
  This is a very large model at ~32GB in size, its recommended to use a) offloading, b) quantization  
  For more information on variations, requirements, options, and how to donwload and use FLUX.1, see [Wiki](https://github.com/vladmandic/automatic/wiki/FLUX)  
  SD.Next supports:  
  - [FLUX.1 Dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [FLUX.1 Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) original variations  
  - additional [qint8](https://huggingface.co/Disty0/FLUX.1-dev-qint8) and [qint4](https://huggingface.co/Disty0/FLUX.1-dev-qint4) quantized variations  
  - additional [nf4](https://huggingface.co/sayakpaul/flux.1-dev-nf4) quantized variation  
- [AuraFlow](https://huggingface.co/fal/AuraFlow)  
  AuraFlow v0.3 is the fully open-sourced largest flow-based text-to-image generation model  
  This is a very large model at 6.8B params and nearly 31GB in size, smaller variants are expected in the future  
  Use scheduler: Default or Euler FlowMatch or Heun FlowMatch  
- [AlphaVLLM Lumina-Next-SFT](https://huggingface.co/Alpha-VLLM/Lumina-Next-SFT-diffusers)  
  Lumina-Next-SFT is a Next-DiT model containing 2B parameters, enhanced through high-quality supervised fine-tuning (SFT)  
  This model uses T5 XXL variation of text encoder (previous version of Lumina used Gemma 2B as text encoder)  
  Use scheduler: Default or Euler FlowMatch or Heun FlowMatch  
- [Kwai Kolors](https://huggingface.co/Kwai-Kolors/Kolors)  
  Kolors is a large-scale text-to-image generation model based on latent diffusion  
  This is an SDXL style model that replaces standard CLiP-L and CLiP-G text encoders with a massive `chatglm3-6b` encoder supporting both English and Chinese prompting  
- [HunyuanDiT 1.2](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers)  
  Hunyuan-DiT is a powerful multi-resolution diffusion transformer (DiT) with fine-grained Chinese understanding  
- [AnimateDiff](https://github.com/guoyww/animatediff/)  
  support for additional models: **SD 1.5 v3** (Sparse), **SD Lightning** (4-step), **SDXL Beta**  

**New Features...**

- support for **Balanced Offload**, thanks @Disty0!  
  balanced offload will dynamically split and offload models from the GPU based on the max configured GPU and CPU memory size  
  model parts that dont fit in the GPU will be dynamically sliced and offloaded to the CPU  
  see *Settings -> Diffusers Settings -> Max GPU memory and Max CPU memory*  
  *note*: recommended value for max GPU memory is ~80% of your total GPU memory  
  *note*: balanced offload will force loading LoRA with Diffusers method  
  *note*: balanced offload is not compatible with Optimum Quanto  
- support for **Optimum Quanto** with 8 bit and 4 bit quantization options, thanks @Disty0 and @Trojaner!  
  to use, go to Settings -> Compute Settings and enable "Quantize Model weights with Optimum Quanto" option  
  *note*: Optimum Quanto requires PyTorch 2.4  
- new prompt attention mode: **xhinker** which brings support for prompt attention to new models such as FLUX.1 and SD3  
  to use, enable in *Settings -> Execution -> Prompt attention*
- use [PEFT](https://huggingface.co/docs/peft/main/en/index) for **LoRA** handling on all models other than SD15/SD21/SDXL  
  this improves LoRA compatibility for SC, SD3, AuraFlow, Flux, etc.  

**Changes & Fixes...**

- default resolution bumped from 512x512 to 1024x1024, time to move on ;)
- convert **Dynamic Attention SDP** into a global SDP option, thanks @Disty0!  
  *note*: requires reset of selected attention option
- update default **CUDA** version from 12.1 to 12.4
- update `requirements`
- samplers now prefers the model defaults over the diffusers defaults, thanks @Disty0!  
- improve xyz grid for lora handling and add lora strength option  
- don't enable Dynamic Attention by default on platforms that support Flash Attention, thanks @Disty0!  
- convert offload options into a single choice list, thanks @Disty0!  
  *note*: requires reset of selected offload option  
- control module allows reszing of indivudual process override images to match input image  
  for example: set size->before->method:nearest, mode:fixed or mode:fill  
- control tab includes superset of txt and img scripts
- automatically offload disabled controlnet units  
- prioritize specified backend if `--use-*` option is used, thanks @lshqqytiger
- ipadapter option to auto-crop input images to faces to improve efficiency of face-transfter ipadapters  
- update **IPEX** to 2.1.40+xpu on Linux, thanks @Disty0!  
- general **ROCm** fixes, thanks @lshqqytiger!  
- support for HIP SDK 6.1 on ZLUDA backend, thanks @lshqqytiger!
- fix full vae previews, thanks @Disty0!  
- fix default scheduler not being applied, thanks @Disty0!  
- fix Stable Cascade with custom schedulers, thanks @Disty0!  
- fix LoRA apply with force-diffusers
- fix LoRA scales with force-diffusers
- fix control API
- fix VAE load refrerencing incorrect configuration
- fix NVML gpu monitoring

## Update for 2024-07-08

This release is primary service release with cumulative fixes and several improvements, but no breaking changes.

**New features...**
- massive updates to [Wiki](https://github.com/vladmandic/automatic/wiki)  
  with over 20 new pages and articles, now includes guides for nearly all major features  
  *note*: this is work-in-progress, if you have any feedback or suggestions, please let us know!
  thanks @GenesisArtemis!  
- support for **DoRA** networks, thanks @AI-Casanova!
- support for [uv](https://pypi.org/project/uv/), extremely fast installer, thanks @Yoinky3000!  
  to use, simply add `--uv` to your command line params  
- [Xinsir ControlNet++ Union](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)  
  new SDXL *all-in-one* controlnet that can process any kind of preprocessors!
- [CogFlorence 2 Large](https://huggingface.co/thwri/CogFlorence-2-Large-Freeze) VLM model  
  to use, simply select in process -> visual query  
- [AuraSR](https://huggingface.co/fal/AuraSR) high-quality 4x GAN-style upscaling model  
  note: this is a large upscaler at 2.5GB  

**And fixes...**
- enable **Florence VLM**  for all platforms, thanks @lshqqytiger!  
- improve ROCm detection under WSL2, thanks @lshqqytiger!  
- add SD3 with FP16 T5 to list of detected models  
- fix executing extensions with zero params  
- add support for embeddings bundled in LoRA, thanks @AI-Casanova!  
- fix executing extensions with zero params  
- fix nncf for lora, thanks @Disty0!  
- fix diffusers version detection for SD3  
- fix current step for higher order samplers  
- fix control input type video  
- fix reset pipeline at the end of each iteration  
- fix faceswap when no faces detected  
- fix civitai search
- multiple ModernUI fixes

## Update for 2024-06-23

### Highlights for 2024-06-23

Following zero-day **SD3** release, a 10 days later heres a refresh with 10+ improvements  
including full prompt attention, support for compressed weights, additional text-encoder quantization modes.  

But theres more than SD3:  
- support for quantized **T5** text encoder *FP16/FP8/FP4/INT8* in all models that use T5: SD3, PixArt-Σ, etc.  
- support for **PixArt-Sigma** in small/medium/large variants  
- support for **HunyuanDiT 1.1**  
- additional **NNCF weights compression** support: SD3, PixArt, ControlNet, Lora  
- integration of **MS Florence** VLM/VQA *Base* and *Large* models  
- (finally) new release of **Torch-DirectML**  
- additional efficiencies for users with low VRAM GPUs  
- over 20 overall fixes  

### Model Improvements for 2024-06-23

- **SD3**: enable tiny-VAE (TAESD) preview and non-full quality mode  
- SD3: enable base LoRA support  
- SD3: add support for FP4 quantized T5 text encoder  
  simply select in *settings -> model -> text encoder*  
  *note* for SD3 with T5, set SD.Next to use FP16 precision, not BF16 precision  
- SD3: add support for INT8 quantized T5 text encoder, thanks @Disty0!  
- SD3: enable cpu-offloading for T5 text encoder, thanks @Disty0!  
- SD3: simplified loading of model in single-file safetensors format  
  model load can now be performed fully offline  
- SD3: full support for prompt parsing and attention, thanks @AI-Casanova!
- SD3: ability to target different prompts to each of text-encoders, thanks @AI-Casanova!  
  example: `dog TE2: cat TE3: bird`
- SD3: add support for sampler shift for Euler FlowMatch  
  see *settings -> samplers*, also available as param in xyz grid  
  higher shift means model will spend more time on structure and less on details  
- SD3: add support for selecting T5 text encoder variant in XYZ grid
- **Pixart-Σ**: Add *small* (512px) and *large* (2k) variations, in addition to existing *medium* (1k)  
- Pixart-Σ: Add support for 4/8bit quantized t5 text encoder  
  *note* by default pixart-Σ uses full fp16 t5 encoder with large memory footprint  
  simply select in *settings -> model -> text encoder* before or after model load  
- **HunyuanDiT**: support for model version 1.1  
- **MS Florence**: integration of Microsoft Florence VLM/VQA Base and Large models  
  simply select in *process -> visual query*!

### General Improvements for 2024-06-23

- support FP4 quantized T5 text encoder, in addition to existing FP8 and FP16
- support for T5 text-encoder loader in **all** models that use T5  
  *example*: load FP4 or FP8 quantized T5 text-encoder into PixArt Sigma!
- support for `torch-directml` **0.2.2**, thanks @lshqqytiger!  
  *note*: new directml is finally based on modern `torch` 2.3.1!  
- xyz grid: add support for LoRA selector
- vae load: store original vae so it can be restored when set to none
- extra networks: info display now contains link to source url if model if its known  
  works for civitai and huggingface models  
- force gc for lowvram users and improve gc logging
- improved google.colab support
- css tweaks for standardui
- css tweaks for modernui
- additional torch gc checks, thanks @Disty0!

**Improvements: NNCF**, thanks @Disty0!  
- SD3 and PixArt support  
- moved the first compression step to CPU  
- sequential cpu offload (lowvram) support  
- Lora support without reloading the model  
- ControlNet compression support  

### Fixes for 2024-06-23

- fix unsaturated outputs, force apply vae config on model load  
- fix hidiffusion handling of non-square aspect ratios, thanks @ShenZhang-Shin!
- fix control second pass resize  
- fix hunyuandit set attention processor
- fix civitai download without name
- fix compatibility with latest adetailer
- fix invalid sampler warning
- fix starting from non git repo
- fix control api negative prompt handling
- fix saving style without name provided
- fix t2i-color adapter
- fix sdxl "has been incorrectly initialized"
- fix api face-hires
- fix api ip-adapter
- fix memory exceptions with ROCm, thanks @Disty0!
- fix face-hires with lowvram, thanks @Disty0!
- fix pag incorrectly resetting pipeline
- cleanup image metadata
- restructure api examples: `cli/api-*`
- handle theme fallback when invalid theme is specified
- remove obsolete training code leftovers

## Update for 2024-06-13

### Highlights for 2024-06-13

First, yes, it is here and supported: [**StabilityAI Stable Diffusion 3 Medium**](https://stability.ai/news/stable-diffusion-3-medium)  
for details on how to download and use, see [Wiki](https://github.com/vladmandic/automatic/wiki/SD3)

#### What else 2024-06-13?

A lot of work on state-of-the-art multi-lingual models with both [Tenecent HunyuanDiT](https://github.com/Tencent/HunyuanDiT) and [MuLan](https://github.com/mulanai/MuLan)  
Plus tons of minor features such as optimized initial install experience, **T-Gate** and **ResAdapter**, additional ModernUI themes (both light and dark) and fixes since the last release which was only 2 weeks ago!

### Full Changelog for 2024-06-13

#### New Models for 2024-06-23

- [StabilityAI Stable Diffusion 3 Medium](https://stability.ai/news/stable-diffusion-3-medium)  
  yup, supported!  
  quote: *"Stable Diffusion 3 Medium is a multimodal diffusion transformer (MMDiT) model that features improved performance in image quality, typography, complex prompt understanding, and resource-efficiency"*  
  sdnext also supports switching optional T5 text encoder on-the-fly as well as loading model from either diffusers repo or safetensors single-file  
  for details, see [Wiki](https://github.com/vladmandic/automatic/wiki/SD3)
- [Tenecent HunyuanDiT](https://github.com/Tencent/HunyuanDiT) bilingual english/chinese diffusion transformer model  
  note: this is a very large model at ~17GB, but can be used with less VRAM using model offloading  
  simply select from networks -> models -> reference, model will be auto-downloaded on first use  

#### New Functionality for 2024-06-23

- [MuLan](https://github.com/mulanai/MuLan) Multi-language prompts
  write your prompts in ~110 auto-detected languages!  
  compatible with *SD15* and *SDXL*  
  enable in scripts -> MuLan and set encoder to `InternVL-14B-224px` encoder  
  *note*: right now this is more of a proof-of-concept before smaller and/or quantized models are released  
  model will be auto-downloaded on first use: note its huge size of 27GB  
  even executing it in FP16 will require ~16GB of VRAM for text encoder alone  
  examples:  
  - English: photo of a beautiful woman wearing a white bikini on a beach with a city skyline in the background
  - Croatian: fotografija lijepe žene u bijelom bikiniju na plaži s gradskim obzorom u pozadini
  - Italian: Foto di una bella donna che indossa un bikini bianco su una spiaggia con lo skyline di una città sullo sfondo
  - Spanish: Foto de una hermosa mujer con un bikini blanco en una playa con un horizonte de la ciudad en el fondo
  - German: Foto einer schönen Frau in einem weißen Bikini an einem Strand mit einer Skyline der Stadt im Hintergrund
  - Arabic: صورة لامرأة جميلة ترتدي بيكيني أبيض على شاطئ مع أفق المدينة في الخلفية
  - Japanese: 街のスカイラインを背景にビーチで白いビキニを着た美しい女性の写真
  - Chinese: 一个美丽的女人在海滩上穿着白色比基尼的照片, 背景是城市天际线
  - Korean: 도시의 스카이라인을 배경으로 해변에서 흰색 비키니를 입은 아름 다운 여성의 사진
- [T-Gate](https://github.com/HaozheLiu-ST/T-GATE) Speed up generations by gating at which step cross-attention is no longer needed  
  enable via scripts -> t-gate  
  compatible with *SD15*  
- **PCM LoRAs** allow for fast denoising using less steps with standard *SD15* and *SDXL* models  
  download from <https://huggingface.co/Kijai/converted_pcm_loras_fp16/tree/main>
- [ByteDance ResAdapter](https://github.com/bytedance/res-adapter) resolution-free model adapter  
  allows to use resolutions from 0.5 to 2.0 of original model resolution, compatible with *SD15* and *SDXL*
  enable via scripts -> resadapter and select desired model
- **Kohya HiRes Fix** allows for higher resolution generation using standard *SD15* models  
  enable via scripts -> kohya-hires-fix  
  *note*: alternative to regular hidiffusion method, but with different approach to scaling  
- additional built-in 4 great custom trained **ControlNet SDXL** models from Xinsir: OpenPose, Canny, Scribble, AnimePainter  
  thanks @lbeltrame
- add torch **full deterministic mode**
  enable in settings -> compute -> use deterministic mode  
  typical differences are not large and its disabled by default as it does have some performance impact  
- new sampler: **Euler FlowMatch**  

#### Improvements Fixes 2024-06-13

- additional modernui themes
- reintroduce prompt attention normalization, disabled by default, enable in settings -> execution  
  this can drastically help with unbalanced prompts  
- further work on improving python 3.12 functionality and remove experimental flag  
  note: recommended version remains python 3.11 for all users, except if you are using directml in which case its python 3.10  
- improved **installer** for initial installs  
  initial install will do single-pass install of all required packages with correct versions  
  subsequent runs will check package versions as necessary  
- add env variable `SD_PIP_DEBUG` to write `pip.log` for all pip operations  
  also improved installer logging  
- add python version check for `torch-directml`  
- do not install `tensorflow` by default  
- improve metadata/infotext parser  
  add `cli/image-exif.py` that can be used to view/extract metadata from images  
- lower overhead on generate calls  
- auto-synchronize modernui and core branches  
- add option to pad prompt with zeros, thanks @Disty

#### Fixes 2024-06-13

- cumulative fixes since the last release  
- fix apply/unapply hidiffusion for sd15  
- fix controlnet reference enabled check  
- fix face-hires with control batch count  
- install pynvml on-demand  
- apply rollback-vae option to latest torch versions, thanks @Iaotle  
- face hires skip if strength is 0  
- restore all sampler configuration on sampler change  

## Update for 2024-06-02

- fix textual inversion loading
- fix gallery mtime display
- fix extra network scrollable area when using modernui
- fix control prompts list handling
- fix restore variation seed and strength
- fix negative prompt parsing from metadata
- fix stable cascade progress monitoring
- fix variation seed with hires pass
- fix loading models trained with onetrainer
- add variation seed info to metadata
- workaround for scale-by when using modernui
- lock torch-directml version
- improve xformers installer
- improve ultralytics installer (face-hires)
- improve triton installer (compile)
- improve insightface installer (faceip)
- improve mim installer (dwpose)
- add dpm++ 1s and dpm++ 3m aliases for dpm++ 2m scheduler with different orders

## Update for 2024-05-28

### Highlights for 2024-05-28

New [SD.Next](https://github.com/vladmandic/automatic) release has been baking in `dev` for a longer than usual, but changes are massive - about 350 commits for core and 300 for UI...

Starting with the new UI - yup, this version ships with a *preview* of the new [ModernUI](https://github.com/BinaryQuantumSoul/sdnext-modernui)  
For details on how to enable and use it, see [Home](https://github.com/BinaryQuantumSoul/sdnext-modernui) and [WiKi](https://github.com/vladmandic/automatic/wiki/Themes)  
**ModernUI** is still in early development and not all features are available yet, please report [issues and feedback](https://github.com/BinaryQuantumSoul/sdnext-modernui/issues)  
Thanks to @BinaryQuantumSoul for his hard work on this project!  

*What else?*

#### New built-in features

- [PWA](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps) SD.Next is now installable as a web-app
- **Gallery**: extremely fast built-in gallery viewer  
  List, preview, search through all your images and videos!  
- **HiDiffusion** allows generating very-high resolution images out-of-the-box using standard models  
- **Perturbed-Attention Guidance** (PAG) enhances sample quality in addition to standard CFG scale  
- **LayerDiffuse** simply create transparent (foreground-only) images  
- **IP adapter masking** allows to use multiple input images for each segment of the input image  
- IP adapter **InstantStyle** implementation  
- **Token Downsampling** (ToDo) provides significant speedups with minimal-to-none quality loss  
- **Samplers optimizations** that allow normal samplers to complete work in 1/3 of the steps!  
  Yup, even popular DPM++2M can now run in 10 steps with quality equaling 30 steps using **AYS** presets  
- Native **wildcards** support  
- Improved built-in **Face HiRes**  
- Better **outpainting**  
- And much more...  
  For details of above features and full list, see [Changelog](https://github.com/vladmandic/automatic/blob/dev/CHANGELOG.md)

#### New models

While still waiting for *Stable Diffusion 3.0*, there have been some significant models released in the meantime:

- [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/), high end diffusion transformer model (*DiT*) capable of directly generating images at 4K resolution  
- [SDXS](https://github.com/IDKiro/sdxs), extremely fast 1-step generation consistency model  
- [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD), 1-step, 2-step, 4-step and 8-step optimized models  

*Note*  
[SD.Next](https://github.com/vladmandic/automatic) is no longer marked as a fork of [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) and github project has been fully detached  
Given huge number of changes with *+3443/-3342* commits diff (at the time of fork detach) over the past year,  
a completely different backend/engine and a change of focus, it is time to give credit to original [author](https://github.com/auTOMATIC1111),  and move on!  

### Full ChangeLog for 2024-05-28

- **Features**:
  - **ModernUI** preview of the new [ModernUI](https://github.com/BinaryQuantumSoul/sdnext-modernui)  
  - [PWA](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps) SD.Next is now installable as a web-app and includes verified manifest  
  - **Gallery
  - **Gallery**: list, preview, search through all your images and videos!  
    Implemented as infinite-scroll with client-side-caching and lazy-loading while being fully async and non-blocking  
    Search or sort by path, name, size, width, height, mtime or any image metadata item, also with extended syntax like *width > 1000*  
    *Settings*: optional additional user-defined folders, thumbnails in fixed or variable aspect-ratio  
  - [HiDiffusion](https://github.com/megvii-research/HiDiffusion):  
    Generate high-resolution images using your standard models without duplicates/distorsions AND improved performance  
    For example, *SD15* can now go up to *2024x2048* and *SDXL* up to *4k* natively
    Simply enable checkbox in advanced menu and set desired resolution  
    Additional settings are available in *settings -> inference settings -> hidiffusion*  
    And can also be set and used via *xyz grid*  
    *Note*: HiDiffusion resolution sensitive, so if you get error, set resolution to be multiples of 128  
  - [Perturbed-Attention Guidance](https://github.com/KU-CVLAB/Perturbed-Attention-Guidance)  
    PAG enhances sample quality by utilizing self-attention in formation of latent in addition to standard CFG scale  
    Simply set *advanced -> attention guidance* and *advanced -> adaptive scaling*  
    Additional options are available in *settings -> inference settings -> pag*  
    *Note*: PAG has replaced SAG as attention guidance method in SD.Next  
  - [LayerDiffuse](https://github.com/rootonchair/diffuser_layerdiffuse)
    Create transparent images with foreground-only being generated  
    Simply select from scripts -> apply to current model  
    All necessary files will be auto-downloaded on first use  
  - **IP Adapter Masking**:  
    Powerful method of using masking with ip-adapters  
    When combined with multiple ip-adapters, it allows for different inputs guidance for each segment of the input image  
    *Hint*: to create masks, you can use manually created masks or control->mask module with auto-segment to create masks and later upload them  
  - **IP Adapter advanced layer configuration**:  
    Allows for more control over how each layer of ip-adapter is applied, requires a valid dict to be passed as input  
    See [InstantStyle](https://github.com/InstantStyle/InstantStyle) for details  
  - **OneDiff**: new optimization/compile engine, thanks @aifartist  
    As with all other compile engines, enable via *settings -> compute settings -> compile*  
  - [ToDo](https://arxiv.org/html/2402.13573v2) Token Downsampling for Efficient Generation of High-Resolution Images  
    Newer alternative method to [ToMe](https://github.com/dbolya/tomesd) that can provide speed-up with minimal quality loss  
    Enable in *settings -> inference settings -> token merging*  
    Also available in XYZ grid  
  - **Outpaint**:  
    New method of outpainting that uses a combination of auto-masking and edge generation to create seamless transitions between original and generated image  
    Use on control tab:
    - *input -> denoising strength: 0.5 or higher*
    - *select image -> outpaint -> expand edges or zoom out to desired size*
    - *size -> mode: outpaint, method: nearest*
    - *mask -> inpaint masked only (if you want to keep original image)*
  - **Wildcards**:
    - native support of standard file-based wildcards in prompt  
    - enabled by default, can be disabled in *settings -> extra networks* if you want to use 3rd party extension  
    - wildcards folder is set in *settings -> system paths* and can be flat-file list or complex folder structure  
    - matches strings `"__*__"` in positive and negative prompts  
    - supports filename and path-based wildcards  
    - supports nested wildcards (wildcard can refer to another wildcard, etc.)  
    - supports wildcards files in one-choice per line or multiple choices per line separated by `|` format  
    - *note*: this is in addition to previously released style-based wildcards  
- **Models**:
  - **Load UNET**: ability to override/load external UNET to a selected model  
    Works similar to how VAE is selected and loaded: Set UNet folder and UNet model in settings  
    Can be replaced on-the-fly, not just during initial model load  
    Enables usage of fine-tunes such as [DPO-SD15](https://huggingface.co/mhdang/dpo-sd1.5-text2image-v1) or [DPO-SDXL](https://huggingface.co/mhdang/dpo-sdxl-text2image-v1)  
    *Note*: if there is a `JSON` file with the same name as the model it will be used as Unet config, otherwise Unet config from currently loaded model will be used  
  - [PixArt-Σ](https://pixart-alpha.github.io/PixArt-sigma-project/)
    pixart-Σ is a high end diffusion Transformer model (DiT) with a T5 encoder/decoder capable of directly generating images at 4K resolution  
    to use, simply select from *networks -> models -> reference -> PixArt-Σ*  
    *note*: this is a very large model at ~22GB  
    set parameters: *sampler: Default*  
  - [SDXS](https://github.com/IDKiro/sdxs)
    sdxs is an extremely fast 1-step generation consistency model that also uses TAESD as quick VAE out-of-the-box  
    to use, simply select from *networks -> models -> reference -> SDXS*  
    set parameters: *sampler: CMSI, steps: 1, cfg_scale: 0.0*
  - [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)  
    sd15 and sdxl 1-step, 2-step, 4-step and 8-step optimized models using lora  
    set parameters: *sampler: TCD or LCM, steps: 1/2/4/8, cfg_scale: 0.0*  
- **UI**:
  - Faster **UI** load times
  - Theme types:  
    **Standard** (built-in themes), **Modern** (experimental nextgen ui), **None** (used for Gradio and Huggingface 3rd party themes)  
    Specifying a theme type updates list of available themes  
    For example, *Gradio* themes will not appear as available if theme type is set to *Standard*  
  - Redesign of base txt2img interface  
  - Minor tweaks to styles: refresh/apply/save
  - See details in [WiKi](https://github.com/vladmandic/automatic/wiki/Themes)
- **API**:
  - Add API endpoint `/sdapi/v1/control` and CLI util `cli/simple-control.py`  
    (in addition to previously added `/sdapi/v1/preprocessors` and `/sdapi/v1/masking`)  
    example:
    > simple-control.py --prompt 'woman in the city' --sampler UniPC --steps 20  
    > --input \~/generative/Samples/cutie-512.png --output /tmp/test.png --processed /tmp/proc.png  
    > --control 'Canny:Canny FP16:0.7, OpenPose:OpenPose FP16:0.8' --type controlnet  
    > --ipadapter 'Plus:~/generative/Samples/cutie-512.png:0.5'  
  - Add API endpoint `/sdapi/v1/vqa` and CLI util `cli/simple-vqa.py`
- **Changes**:
  - Due to change in Diffusers model loading  
    initial model load will now fetch config files required for the model  
    from the Huggingface site instead of using predefined YAML files
  - Removed built-in extensions: *ControlNet* and *Image-Browser*  
    as both *image-browser* and *controlnet* have native built-in equivalents  
    both can still be installed by user if desired  
  - Different defaults depending on available GPU, thanks @Disty0
    - 4GB and below: *lowvram*
    - 8GB and below: *medvram*
    - Cross-attention: Dynamic Attention SDP with *medvram* or *lowvram*, otherwise SDP  
    - VAE Tiling enabled with *medvram* and *lowvram*
    - Disable Extract EMA by default
    - Disable forced VAE Slicing for *lowvram*
  - Upscaler compile disabled by default with OpenVINO backend  
  - Hypernetwork support disabled by default, can be enabled in settings  
- **Improvements**:
  - Faster server startup  
  - Styles apply wildcards to params
  - Face HiRes fully configurable and higher quality when using high-resolution models  
  - Extra networks persistent sort order in settings  
  - Add option to make batch generations use fully random seed vs sequential  
  - Make metadata in full screen viewer optional
  - Add VAE civitai scan metadata/preview
  - More efficient in-browser callbacks
  - Updated all system requirements  
  - UI log monitor will auto-reconnect to server on server restart  
  - UI styles includes indicator for active styles  
  - UI reduce load on browser  
  - Secondary sampler add option "same as primary"  
  - Change attention mechanism on-the-fly without model reload, thanks @Disty0  
  - Update stable-fast with support for torch 2.2.2 and 2.3.0, thanks @Aptronymist
  - Add torch *cudaMallocAsync* in compute options  
    Can improve memory utilization on compatible GPUs (RTX and newer)  
  - Torch dynamic profiling  
    You can enable/disable full torch profiling in settings top menu on-the-fly  
  - Prompt caching - if you use the same prompt multiple times, no need to re-parse and encode it  
    Useful for batches as prompt processing is ~0.1sec on each pass  
  - Enhance `SD_PROMPT_DEBUG` to show actual tokens used
  - Support controlnet manually downloads models in both standalone and diffusers format  
    For standalone, simply copy safetensors file to `models/control/controlnet` folder  
    For diffusers format, create folder with model name in `models/control/controlnet/`  
    and copy `model.json` and `diffusion_pytorch_model.safetensors` to that folder  
- **Samplers**
  - Add *Euler SGM* variation (e.g. SGM Uniform), optimized for SDXL-Lightning models  
    *note*: you can use other samplers as well with SDXL-Lightning models  
  - Add *CMSI* sampler, optimized for consistency models  
  - Add option *timestep spacing* to sampler settings and sampler section in main ui
    Note: changing timestep spacing changes behavior of sampler and can help to make any sampler turbo/lightning compatibile
  - Add option *timesteps* to manually set timesteps instead of relying on steps+spacing  
    Additionally, presets from nVidias align-you-steps reasearch are provided  
    Result is that perfectly aligned steps can drastically reduce number of steps needed!  
    For example, **AYS** preset alows DPM++2M to run in ~10 steps with quality equallying ~30 steps!  
- **IPEX**, thanks @Disty0
  - Update to *IPEX 2.1.20* on Linux  
    requires removing the venv folder to update properly  
  - Removed 1024x1024 workaround  
  - Disable ipexrun by default, set `IPEXRUN=True` if you want to use `ipexrun`  
- **ROCm**, thanks @Disty0  
  - Add support for ROCm 6.1 nighthly builds  
  - Switch to stable branch of PyTorch  
  - Compatibility improvenments  
  - Add **MIGraphX** torch compile engine  
- **ZLUDA**, thanks @lshqqytiger
  - Rewrite ZLUDA installer
  - ZLUDA **v3.8** updates: Runtime API support
  - Add `--reinstall-zluda` (to download the latest ZLUDA)
- **Fixes**:
  - Update requirements
  - Installer automatically handle detached git states  
  - Prompt params parser
  - Allowing forcing LoRA loading method for some or all models
  - Image save without metadata
  - API generate save metadata
  - Face/InstantID faults
  - CivitAI update model info for all models
  - FP16/BF16 test on model load
  - Variation seed possible NaNs
  - Enumerate diffusers model with multiple variants
  - Diffusers skip non-models on enum
  - Face-HiRes compatibility with control modules
  - Face-HiRes avoid doule save in some scenarios
  - Loading safetensors embeddings
  - CSS fixes
  - Check if attention processor is compatible with model
  - SD Upscale when used with control module
  - Noise sampler seed, thanks @leppie
  - Control module with ADetailer and active ControlNet
  - Control module restore button full functionality
  - Control improved handling with multiple control units and different init images
  - Control add correct metadata to image
  - Time embeddings load part of model load
  - A1111 update OptionInfo properties
  - MOTD exception handling
  - Notifications not triggering
  - Prompt cropping on copy

## Update for 2024-03-19

### Highlights 2024-03-19

New models:
- [Stable Cascade](https://github.com/Stability-AI/StableCascade) *Full* and *Lite*
- [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
- [KOALA 700M](https://github.com/youngwanLEE/sdxl-koala)
- [Stable Video Diffusion XT 1.1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)
- [VGen](https://huggingface.co/ali-vilab/i2vgen-xl)  

New pipelines and features:
- Img2img using [LEdit++](https://leditsplusplus-project.static.hf.space/index.html), context aware method with image analysis and positive/negative prompt handling
- Trajectory Consistency Distillation [TCD](https://mhh0318.github.io/tcd) for processing in even less steps
- Visual Query & Answer using [moondream2](https://github.com/vikhyat/moondream) as an addition to standard interrogate methods
- **Face-HiRes**: simple built-in detailer for face refinements
- Even simpler outpaint: when resizing image, simply pick outpaint method and if image has different aspect ratio, blank areas will be outpainted!
- UI aspect-ratio controls and other UI improvements
- User controllable invisibile and visible watermarking
- Native composable LoRA

What else?

- **Reference models**: *Networks -> Models -> Reference*: All reference models now come with recommended settings that can be auto-applied if desired  
- **Styles**: Not just for prompts! Styles can apply *generate parameters* as templates and can be used to *apply wildcards* to prompts  
improvements, Additional API endpoints  
- Given the high interest in [ZLUDA](https://github.com/vosen/ZLUDA) engine introduced in last release weve updated much more flexible/automatic install procedure (see [wiki](https://github.com/vladmandic/automatic/wiki/ZLUDA) for details)  
- Plus Additional Improvements such as: Smooth tiling, Refine/HiRes workflow improvements, Control workflow  

Further details:  
- For basic instructions, see [README](https://github.com/vladmandic/automatic/blob/master/README.md)  
- For more details on all new features see full [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
- For documentation, see [WiKi](https://github.com/vladmandic/automatic/wiki)
- [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) server  

### Full Changelog 2024-03-19

- [Stable Cascade](https://github.com/Stability-AI/StableCascade) *Full* and *Lite*
  - large multi-stage high-quality model from warp-ai/wuerstchen team and released by stabilityai  
  - download using networks -> reference
  - see [wiki](https://github.com/vladmandic/automatic/wiki/Stable-Cascade) for details
- [Playground v2.5](https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic)
  - new model version from Playground: based on SDXL, but with some cool new concepts
  - download using networks -> reference
  - set sampler to *DPM++ 2M EDM* or *Euler EDM*
- [KOALA 700M](https://github.com/youngwanLEE/sdxl-koala)
  - another very fast & light sdxl model where original unet was compressed and distilled to 54% of original size  
  - download using networks -> reference
  - *note* to download fp16 variant (recommended), set settings -> diffusers -> preferred model variant  
- [LEdit++](https://leditsplusplus-project.static.hf.space/index.html)
  - context aware img2img method with image analysis and positive/negative prompt handling  
  - enable via img2img -> scripts -> ledit
  - uses following params from standard img2img: cfg scale (recommended ~3), steps (recommended ~50), denoise strength (recommended ~0.7)
  - can use postive and/or negative prompt to guide editing process
    - positive prompt: what to enhance, strength and threshold for auto-masking
    - negative prompt: what to remove, strength and threshold for auto-masking  
  - *note*: not compatible with model offloading
- **Second Pass / Refine**
  - independent upscale and hires options: run hires without upscale or upscale without hires or both
  - upscale can now run 0.1-8.0 scale and will also run if enabled at 1.0 to allow for upscalers that simply improve image quality
  - update ui section to reflect changes
  - *note*: behavior using backend:original is unchanged for backwards compatibilty
- **Visual Query** visual query & answer in process tab  
  - go to process -> visual query  
  - ask your questions, e.g. "describe the image", "what is behind the subject", "what are predominant colors of the image?"
  - primary model is [moondream2](https://github.com/vikhyat/moondream), a *tiny* 1.86B vision language model  
    *note*: its still 3.7GB in size, so not really tiny  
  - additional support for multiple variations of several base models: *GIT, BLIP, ViLT, PIX*, sizes range from 0.3 to 1.7GB  
- **Video**
  - **Image2Video**
    - new module for creating videos from images  
    - simply enable from *img2img -> scripts -> image2video*  
    - model is auto-downloaded on first use
    - based on [VGen](https://huggingface.co/ali-vilab/i2vgen-xl)  
  - **Stable Video Diffusion**
    - updated with *SVD 1.0, SVD XT 1.0 and SVD XT 1.1*
    - models are auto-downloaded on first use
    - simply enable from *img2img -> scripts -> stable video diffusion*  
    - for svd 1.0, use frames=~14, for xt models use frames=~25
- **Composable LoRA**, thanks @AI-Casanova
  - control lora strength for each step
    for example: `<xxx:0.1@0,0.9@1>` means strength=0.1 for step at 0% and intepolate towards strength=0.9 for step at 100%
  - *note*: this is a very experimental feature and may not work as expected
- **Control**
  - added *refiner/hires* workflows
  - added resize methods to before/after/mask: fixed, crop, fill
- **Styles**: styles are not just for prompts!
  - new styles editor: *networks -> styles -> edit*
  - styles can apply generate parameters, for example to have a style that enables and configures hires:  
    parameters=`enable_hr: True, hr_scale: 2, hr_upscaler: Latent Bilinear antialias, hr_sampler_name: DEIS, hr_second_pass_steps: 20, denoising_strength: 0.5`
  - styles can apply wildcards to prompts, for example:  
    wildcards=`movie=mad max, dune, star wars, star trek; intricate=realistic, color sketch, pencil sketch, intricate`
  - as usual, you can apply any number of styles so you can choose which settings are applied and in which order and which wildcards are used
- **UI**
  - *aspect-ratio** add selector and lock to width/height control  
    allowed aspect ration can be configured via *settings -> user interface*  
  - *interrogate* tab is now merged into *process* tab  
  - *image viewer* now displays image metadata
  - *themes* improve on-the-fly switching
  - *log monitor* flag server warnings/errors and overall improve display
  - *control* separate processor settings from unit settings
- **Face HiRes**
  - new *face restore* option, works similar to well-known *adetailer* by running an inpaint on detected faces but with just a checkbox to enable/disable  
  - set as default face restorer in settings -> postprocessing  
  - disabled by default, to enable simply check *face restore* in your generate advanced settings  
  - strength, steps and sampler are set using by hires section in refine menu  
  - strength can be overriden in settings -> postprocessing  
  - will use secondary prompt and secondary negative prompt if present in refine  
- **Watermarking**
  - SD.Next disables all known watermarks in models, but does allow user to set custom watermark  
  - see *settings -> image options -> watermarking*  
  - invisible watermark: using steganogephy  
  - image watermark: overlaid on top of image  
- **Reference models**
  - additional reference models available for single-click download & run:
    *Stable Cascade, Stable Cascade lite, Stable Video Diffusion XT 1.1*  
  - reference models will now download *fp16* variation by default  
  - reference models will print recommended settings to log if present
  - new setting in extra network: *use reference values when available*  
    disabled by default, if enabled will force use of reference settings for models that have them
- **Samplers**
  - [TCD](https://mhh0318.github.io/tcd/): Trajectory Consistency Distillation  
    new sampler that produces consistent results in a very low number of steps (comparable to LCM but without reliance on LoRA)  
    for best results, use with TCD LoRA: <https://huggingface.co/h1t/TCD-SDXL-LoRA>
  - *DPM++ 2M EDM* and *Euler EDM*  
    EDM is a new solver algorithm currently available for DPM++2M and Euler samplers  
    Note that using EDM samplers with non-EDM optimized models will provide just noise and vice-versa  
- **Improvements**
  - **FaceID** extend support for LoRA, HyperTile and FreeU, thanks @Trojaner
  - **Tiling** now extends to both Unet and VAE producing smoother outputs, thanks @AI-Casanova
  - new setting in image options: *include mask in output*
  - improved params parsing from from prompt string and styles
  - default theme updates and additional built-in theme *black-gray*
  - support models with their own YAML model config files
  - support models with their own JSON per-component config files, for example: `playground-v2.5_vae.config`
  - prompt can have comments enclosed with `/*` and `*/`  
    comments are extracted from prompt and added to image metadata  
- **ROCm**  
  - add **ROCm** 6.0 nightly option to installer, thanks @jicka
  - add *flash attention* support for rdna3, thanks @Disty0  
    install flash_attn package for rdna3 manually and enable *flash attention* from *compute settings*  
    to install flash_attn, activate the venv and run `pip install -U git+https://github.com/ROCm/flash-attention@howiejay/navi_support`  
- **IPEX**
  - disabled IPEX Optimize by default  
- **API**
  - add preprocessor api endpoints  
    GET:`/sdapi/v1/preprocessors`, POST:`/sdapi/v1/preprocess`, sample script:`cli/simple-preprocess.py`
  - add masking api endpoints  
    GET:`/sdapi/v1/masking`, POST:`/sdapi/v1/mask`, sample script:`cli/simple-mask.py`
- **Internal**
  - improved vram efficiency for model compile, thanks @Disty0
  - **stable-fast** compatibility with torch 2.2.1  
  - remove obsolete textual inversion training code
  - remove obsolete hypernetworks training code
- **Refiner** validated workflows:
  - Fully functional: SD15 + SD15, SDXL + SDXL, SDXL + SDXL-R
  - Functional, but result is not as good: SD15 + SDXL, SDXL + SD15, SD15 + SDXL-R
- **SDXL Lightning** models just-work, just makes sure to set CFG Scale to 0  
    and choose a best-suited sampler, it may not be the one youre used to (e.g. maybe even basic Euler)  
- **Fixes**
  - improve *model cpu offload* compatibility
  - improve *model sequential offload* compatibility
  - improve *bfloat16* compatibility
  - improve *xformers* installer to match cuda version and install triton
  - fix extra networks refresh
  - fix *sdp memory attention* in backend original
  - fix autodetect sd21 models
  - fix api info endpoint
  - fix *sampler eta* in xyz grid, thanks @AI-Casanova
  - fix *requires_aesthetics_score* errors
  - fix t2i-canny
  - fix *differenital diffusion* for manual mask, thanks @23pennies
  - fix ipadapter apply/unapply on batch runs
  - fix control with multiple units and override images
  - fix control with hires
  - fix control-lllite
  - fix font fallback, thanks @NetroScript
  - update civitai downloader to handler new metadata
  - improve control error handling
  - use default model variant if specified variant doesnt exist
  - use diffusers lora load override for *lcm/tcd/turbo loras*
  - exception handler around vram memory stats gather
  - improve ZLUDA installer with `--use-zluda` cli param, thanks @lshqqytiger

## Update for 2024-02-22

Only 3 weeks since last release, but heres another feature-packed one!
This time release schedule was shorter as we wanted to get some of the fixes out faster.

### Highlights 2024-02-22

- **IP-Adapters** & **FaceID**: multi-adapter and multi-image suport  
- New optimization engines: [DeepCache](https://github.com/horseee/DeepCache), [ZLUDA](https://github.com/vosen/ZLUDA) and **Dynamic Attention Slicing**  
- New built-in pipelines: [Differential diffusion](https://github.com/exx8/differential-diffusion) and [Regional prompting](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline)  
- Big updates to: **Outpainting** (noised-edge-extend), **Clip-skip** (interpolate with non-integrer values!), **CFG end** (prevent overburn on high CFG scales), **Control** module masking functionality  
- All reported issues since the last release are addressed and included in this release  

Further details:  
- For basic instructions, see [README](https://github.com/vladmandic/automatic/blob/master/README.md)  
- For more details on all new features see full [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
- For documentation, see [WiKi](https://github.com/vladmandic/automatic/wiki)
- [Discord](https://discord.com/invite/sd-next-federal-batch-inspectors-1101998836328697867) server  

### Full ChangeLog for 2024-02-22

- **Improvements**:
  - **IP Adapter** major refactor  
    - support for **multiple input images** per each ip adapter  
    - support for **multiple concurrent ip adapters**  
      *note*: you cannot mix & match ip adapters that use different *CLiP* models, for example `Base` and `Base ViT-G`  
    - add **adapter start/end** to settings, thanks @AI-Casanova  
      having adapter start late can help with better control over composition and prompt adherence  
      having adapter end early can help with overal quality and performance  
    - unified interface in txt2img, img2img and control  
    - enhanced xyz grid support  
  - **FaceID** now also works with multiple input images!  
  - [Differential diffusion](https://github.com/exx8/differential-diffusion)  
    img2img generation where you control strength of each pixel or image area  
    can be used with manually created masks or with auto-generated depth-maps
    uses general denoising strength value  
    simply enable from *img2img -> scripts -> differential diffusion*  
    *note*: supports sd15 and sdxl models  
  - [Regional prompting](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline) as a built-in solution  
    usage is same as original implementation from @hako-mikan  
    click on title to open docs and see examples of full syntax on how to use it  
    simply enable from *scripts -> regional prompting*  
    *note*: supports sd15 models only  
  - [DeepCache](https://github.com/horseee/DeepCache) model acceleration  
    it can produce massive speedups (2x-5x) with no overhead, but with some loss of quality  
    *settings -> compute -> model compile -> deep-cache* and *settings -> compute -> model compile -> cache interval*  
  - [ZLUDA](https://github.com/vosen/ZLUDA) experimental support, thanks @lshqqytiger  
    - ZLUDA is CUDA wrapper that can be used for GPUs without native support
    - best use case is *AMD GPUs on Windows*, see [wiki](https://github.com/vladmandic/automatic/wiki/ZLUDA) for details  
  - **Outpaint** control outpaint now uses new alghorithm: noised-edge-extend  
    new method allows for much larger outpaint areas in a single pass, even outpaint 512->1024 works well  
    note that denoise strength should be increased for larger the outpaint areas, for example outpainting 512->1024 works well with denoise 0.75  
    outpaint can run in *img2img* mode (default) and *inpaint* mode where original image is masked (if inpaint masked only is selected)  
  - **Clip-skip** reworked completely, thanks @AI-Casanova & @Disty0  
    now clip-skip range is 0-12 where previously lowest value was 1 (default is still 1)  
    values can also be decimal to interpolate between different layers, for example `clip-skip: 1.5`, thanks @AI-Casanova  
  - **CFG End** new param to control image generation guidance, thanks @AI-Casanova  
    sometimes you want strong control over composition, but you want it to stop at some point  
    for example, when used with ip-adapters or controlnet, high cfg scale can overpower the guided image  
  - **Control**
    - when performing inpainting, you can specify processing resolution using **size->mask**  
    - units now have extra option to re-use current preview image as processor input  
  - **Cross-attention** refactored cross-attention methods, thanks @Disty0  
    - for backend:original, its unchanged: SDP, xFormers, Doggettxs, InvokeAI, Sub-quadratic, Split attention  
    - for backend:diffuers, list is now: SDP, xFormers, Batch matrix-matrix, Split attention, Dynamic Attention BMM, Dynamic Attention SDP  
      note: you may need to update your settings! Attention Slicing is renamed to Split attention  
    - for ROCm, updated default cross-attention to Scaled Dot Product  
  - **Dynamic Attention Slicing**, thanks @Disty0  
    - dynamically slices attention queries in order to keep them under the slice rate  
      slicing gets only triggered if the query size is larger than the slice rate to gain performance  
      *Dynamic Attention Slicing BMM* uses *Batch matrix-matrix*  
      *Dynamic Attention Slicing SDP* uses *Scaled Dot Product*  
    - *settings -> compute settings -> attention -> dynamic attention slicing*  
  - **ONNX**:  
    - allow specify onnx default provider and cpu fallback  
      *settings -> diffusers*  
    - allow manual install of specific onnx flavor  
      *settings -> onnx*  
    - better handling of `fp16` models/vae, thanks @lshqqytiger  
  - **OpenVINO** update to `torch 2.2.0`, thanks @Disty0  
  - **HyperTile** additional options thanks @Disty0  
    - add swap size option  
    - add use only for hires pass option  
  - add `--theme` cli param to force theme on startup  
  - add `--allow-paths` cli param to add additional paths that are allowed to be accessed via web, thanks @OuticNZ  
- **Wiki**:
  - added benchmark notes for IPEX, OpenVINO and Olive  
  - added ZLUDA wiki page  
- **Internal**
  - update dependencies  
  - refactor txt2img/img2img api  
  - enhanced theme loader  
  - add additional debug env variables  
  - enhanced sdp cross-optimization control  
    see *settings -> compute settings*  
  - experimental support for *python 3.12*  
- **Fixes**:  
  - add variation seed to diffusers txt2img, thanks @AI-Casanova  
  - add cmd param `--skip-env` to skip setting of environment parameters during sdnext load  
  - handle extensions that install conflicting versions of packages  
    `onnxruntime`, `opencv2-python`  
  - installer refresh package cache on any install  
  - fix embeddings registration on server startup, thanks @AI-Casanova  
  - ipex handle dependencies, thanks @Disty0  
  - insightface handle dependencies  
  - img2img mask blur and padding  
  - xyz grid handle ip adapter name and scale  
  - lazy loading of image may prevent metadata from being loaded on time  
  - allow startup without valid models folder  
  - fix interrogate api endpoint  
  - control fix resize causing runtime errors  
  - control fix processor override image after processor change  
  - control fix display grid with batch  
  - control restore pipeline before running scripts/extensions  
  - handle pipelines that return dict instead of object  
  - lora use strict name matching if preferred option is by-filename  
  - fix inpaint mask only for diffusers  
  - fix vae dtype mismatch, thanks @Disty0  
  - fix controlnet inpaint mask  
  - fix theme list refresh  
  - fix extensions update information in ui  
  - fix taesd with bfloat16
  - fix model merge manual merge settings, thanks @AI-Casanova  
  - fix gradio instant update issues for textboxes in quicksettings  
  - fix rembg missing dependency  
  - bind controlnet extension to last known working commit, thanks @Aptronymist  
  - prompts-from-file fix resizable prompt area  

## Update for 2024-02-07

Another big release just hit the shelves!

### Highlights 2024-02-07  

- A lot more functionality in the **Control** module:
  - Inpaint and outpaint support, flexible resizing options, optional hires  
  - Built-in support for many new processors and models, all auto-downloaded on first use  
  - Full support for scripts and extensions  
- Complete **Face** module  
  implements all variations of **FaceID**, **FaceSwap** and latest **PhotoMaker** and **InstantID**  
- Much enhanced **IPAdapter** modules  
- Brand new **Intelligent masking**, manual or automatic  
  Using ML models (*LAMA* object removal, *REMBG* background removal, *SAM* segmentation, etc.) and with live previews  
  With granular blur, erode and dilate controls  
- New models and pipelines:  
  **Segmind SegMoE**, **Mixture Tiling**, **InstaFlow**, **SAG**, **BlipDiffusion**  
- Massive work integrating latest advances with [OpenVINO](https://github.com/vladmandic/automatic/wiki/OpenVINO), [IPEX](https://github.com/vladmandic/automatic/wiki/Intel-ARC) and [ONNX Olive](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime-&-Olive)
- Full control over brightness, sharpness and color shifts and color grading during generate process directly in latent space  
- **Documentation**! This was a big one, with a lot of new content and updates in the [WiKi](https://github.com/vladmandic/automatic/wiki)  

Plus welcome additions to **UI performance, usability and accessibility** and flexibility of deployment as well as **API** improvements  
And it also includes fixes for all reported issues so far  

As of this release, default backend is set to **diffusers** as its more feature rich than **original** and supports many additional models (original backend does remain as fully supported)  

Also, previous versions of **SD.Next** were tuned for balance between performance and resource usage.  
With this release, focus is more on performance.  
See [Benchmark](https://github.com/vladmandic/automatic/wiki/Benchmark) notes for details, but as a highlight, we are now hitting **~110-150 it/s** on a standard nVidia RTX4090 in optimal scenarios!  

Further details:  
- For basic instructions, see [README](https://github.com/vladmandic/automatic/blob/master/README.md)  
- For more details on all new features see full [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
- For documentation, see [WiKi](https://github.com/vladmandic/automatic/wiki)

### Full ChangeLog 2024-02-07  

- Heavily updated [Wiki](https://github.com/vladmandic/automatic/wiki)  
- **Control**:  
  - new docs:
    - [Control overview](https://github.com/vladmandic/automatic/wiki/Control)  
    - [Control guide](https://github.com/vladmandic/automatic/wiki/Control-Guide), thanks @Aptronymist  
  - add **inpaint** support  
    applies to both *img2img* and *controlnet* workflows  
  - add **outpaint** support  
    applies to both *img2img* and *controlnet* workflows  
    *note*: increase denoising strength since outpainted area is blank by default  
  - new **mask** module  
    - granular blur (gaussian), erode (reduce or remove noise) and dilate (pad or expand)  
    - optional **live preview**  
    - optional **auto-segmentation** using ml models  
      auto-segmentation can be done using **segment-anything** models or **rembg** models  
      *note*: auto segmentation will automatically expand user-masked area to segments that include current user mask  
    - optional **auto-mask**  
      if you dont provide mask or mask is empty, you can instead use auto-mask to automatically generate mask  
      this is especially useful if you want to use advanced masking on batch or video inputs and dont want to manually mask each image  
      *note*: such auto-created mask is also subject to all other selected settings such as auto-segmentation, blur, erode and dilate  
    - optional **object removal** using LaMA model  
      remove selected objects from images with a single click  
      works best when combined with auto-segmentation to remove smaller objects  
    - masking can be combined with control processors in which case mask is applied before processor  
    - unmasked part of can is optionally applied to final image as overlay, see settings `mask_apply_overlay`  
  - support for many additional controlnet models  
    now built-in models include 30+ SD15 models and 15+ SDXL models  
  - allow **resize** both *before* and *after* generate operation  
    this allows for workflows such as: *image -> upscale or downscale -> generate -> upscale or downscale -> output*  
    providing more flexibility and than standard hires workflow  
    *note*: resizing before generate can be done using standard upscalers or latent
  - implicit **hires**  
    since hires is only used for txt2img, control reuses existing resize functionality
    any image size is used as txt2img target size  
    but if resize scale is also set its used to additionally upscale image after initial txt2img and for hires pass  
  - add support for **scripts** and **extensions**  
    you can now combine control workflow with your favorite script or extension  
    *note* extensions that are hard-coded for txt2img or img2img tabs may not work until they are updated  
  - add **depth-anything** depth map processor and trained controlnet  
  - add **marigold** depth map processor  
    this is state-of-the-art depth estimation model, but its quite heavy on resources  
  - add **openpose xl** controlnet  
  - add blip/booru **interrogate** functionality to both input and output images  
  - configurable output folder in settings  
  - auto-refresh available models on tab activate  
  - add image preview for override images set per-unit  
  - more compact unit layout  
  - reduce usage of temp files  
  - add context menu to action buttons  
  - move ip-adapter implementation to control tabs  
  - resize by now applies to input image or frame individually  
    allows for processing where input images are of different sizes  
  - support controlnets with non-default yaml config files  
  - implement resize modes for override images  
  - allow any selection of units  
  - dynamically install depenencies required by specific processors  
  - fix input image size  
  - fix video color mode  
  - fix correct image mode  
  - fix batch/folder/video modes  
  - fix processor switching within same unit  
  - fix pipeline switching between different modes  
- **Face** module  
  implements all variations of **FaceID**, **FaceSwap** and latest **PhotoMaker** and **InstantID**  
  simply select from scripts and choose your favorite method and model  
  *note*: all models are auto-downloaded on first use  
  - [FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)  
    - faceid guides image generation given the input image  
    - full implementation for *SD15* and *SD-XL*, to use simply select from *Scripts*  
      **Base** (93MB) uses *InsightFace* to generate face embeds and *OpenCLIP-ViT-H-14* (2.5GB) as image encoder  
      **Plus** (150MB) uses *InsightFace* to generate face embeds and *CLIP-ViT-H-14-laion2B* (3.8GB) as image encoder  
      **SDXL** (1022MB) uses *InsightFace* to generate face embeds and *OpenCLIP-ViT-bigG-14* (3.7GB) as image encoder  
  - [FaceSwap](https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md)  
    - face swap performs face swapping at the end of generation  
    - based on InsightFace in-swapper  
  - [PhotoMaker](https://github.com/TencentARC/PhotoMaker)  
    - for *SD-XL* only  
    - new model from TenencentARC using similar concept as IPAdapter, but with different implementation and  
      allowing full concept swaps between input images and generated images using trigger words  
    - note: trigger word must match exactly one term in prompt for model to work  
  - [InstantID](https://github.com/InstantID/InstantID)  
    - for *SD-XL* only  
    - based on custom trained ip-adapter and controlnet combined concepts  
    - note: controlnet appears to be heavily watermarked  
  - enable use via api, thanks @trojaner  
- [IPAdapter](https://huggingface.co/h94/IP-Adapter)  
  - additional models for *SD15* and *SD-XL*, to use simply select from *Scripts*:  
    **SD15**: Base, Base ViT-G, Light, Plus, Plus Face, Full Face  
    **SDXL**: Base SDXL, Base ViT-H SDXL, Plus ViT-H SDXL, Plus Face ViT-H SDXL  
  - enable use via api, thanks @trojaner  
- [Segmind SegMoE](https://github.com/segmind/segmoe)  
  - initial support for reference models  
    download&load via network -> models -> reference -> **SegMoE SD 4x2** (3.7GB), **SegMoE XL 2x1** (10GB), **SegMoE XL 4x2**  
  - note: since segmoe is basically sequential mix of unets from multiple models, it can get large  
    SD 4x2 is ~4GB, XL 2x1 is ~10GB and XL 4x2 is 18GB  
  - supports lora, thanks @AI-Casanova
  - support for create and load custom mixes will be added in the future  
- [Mixture Tiling](https://arxiv.org/abs/2302.02412)  
  - uses multiple prompts to guide different parts of the grid during diffusion process  
  - can be used ot create complex scenes with multiple subjects  
  - simply select from scripts  
- [Self-attention guidance](https://github.com/SusungHong/Self-Attention-Guidance)  
  - simply select scale in advanced menu  
  - can drastically improve image coherence as well as reduce artifacts  
  - note: only compatible with some schedulers  
- [FreeInit](https://tianxingwu.github.io/pages/FreeInit/) for **AnimateDiff**
  - greatly improves temporal consistency of generated outputs  
  - all options are available in animateddiff script  
- [SalesForce BlipDiffusion](https://huggingface.co/docs/diffusers/api/pipelines/blip_diffusion)  
  - model can be used to place subject in a different context  
  - requires input image  
  - last word in prompt and negative prompt will be used as source and target subjects  
  - sampler must be set to default before loading the model  
- [InstaFlow](https://github.com/gnobitab/InstaFlow)  
  - another take on super-fast image generation in a single step  
  - set *sampler:default, steps:1, cfg-scale:0*  
  - load from networks -> models -> reference  
- **Improvements**  
  - **ui**  
    - check version and **update** SD.Next via UI  
      simply go to: settings -> update
    - globally configurable **font size**  
      will dynamically rescale ui depending on settings -> user interface  
    - built-in **themes** can be changed on-the-fly  
      this does not work with gradio-default themes as css is created by gradio itself  
    - two new **themes**: *simple-dark* and *simple-light*  
    - modularized blip/booru interrogate  
      now appears as toolbuttons on image/gallery output  
    - faster browser page load  
    - update hints, thanks @brknsoul  
    - cleanup settings  
  - **server**
    - all move/offload options are disable by default for optimal performance  
      enable manually if low on vram  
  - **server startup**: performance  
    - reduced module imports  
      ldm support is now only loaded when running in backend=original  
    - faster extension load  
    - faster json parsing  
    - faster lora indexing  
    - lazy load optional imports  
    - batch embedding load, thanks @midcoastal and @AI-Casanova  
      10x+ faster embeddings load for large number of embeddings, now works for 1000+ embeddings  
    - file and folder list caching, thanks @midcoastal
      if you have a lot of files and and/or are using slower or non-local storage, this speeds up file access a lot  
    - add `SD_INSTALL_DEBUG` env variable to trace all `git` and `pip` operations
  - **extra networks**  
    - 4x faster civitai metadata and previews lookup  
    - better display and selection of tags & trigger words  
      if hashes are calculated, trigger words will only be displayed for actual model version  
    - better matching of previews  
    - better search, including searching for multiple keywords or using full regex  
      see wiki page for more details on syntax  
      thanks @NetroScript  
    - reduce html overhead  
  - **model compression**, thanks @Disty0  
    - using built-in NNCF model compression, you can reduce the size of your models significantly  
      example: up to 3.4GB of VRAM saved for SD-XL model!  
    - see [wiki](https://github.com/vladmandic/automatic/wiki/Model-Compression-with-NNCF) for details  
  - **embeddings**  
    you can now use sd 1.5 embeddings with your sd-xl models!, thanks @AI-Casanova  
    conversion is done on-the-fly, is completely transparent and result is an approximation of embedding  
    to enable: settings->extra networks->auto-convert embeddings  
  - **offline deployment**: allow deployment without git clone  
    for example, you can now deploy a zip of the sdnext folder  
  - **latent upscale**: updated latent upscalers (some are new)  
    *nearest, nearest-exact, area, bilinear, bicubic, bilinear-antialias, bicubic-antialias*
  - **scheduler**: added `SA Solver`  
  - **model load to gpu**  
    new option in settings->diffusers allowing models to be loaded directly to GPU while keeping RAM free  
    this option is not compatible with any kind of model offloading as model is expected to stay in GPU  
    additionally, all model-moves can now be traced with env variable `SD_MOVE_DEBUG`  
  - **xyz grid**
    - range control  
      example: `5.0-6.0:3` will generate 3 images with values `5.0,5.5,6.0`  
      example: `10-20:4` will generate 4 images with values `10,13,16,20`  
    - continue on error  
      now you can use xyz grid with different params and test which ones work and which dont  
    - correct font scaling, thanks @nCoderGit  
  - **hypertile**  
    - enable vae tiling  
    - add autodetect optimial value  
      set tile size to 0 to use autodetected value  
  - **cli**  
    - `sdapi.py` allow manual api invoke  
      example: `python cli/sdapi.py /sdapi/v1/sd-models`  
    - `image-exif.py` improve metadata parsing  
    - `install-sf` helper script to automatically find best available stable-fast package for the platform  
  - **memory**: add ram usage monitoring in addition to gpu memory usage monitoring  
  - **vae**: enable taesd batch decode  
    enable/disable with settings -> diffusers > vae slicing  
- **compile**
  - new option: **fused projections**  
    pretty much free 5% performance boost for compatible models  
    enable in settings -> compute settings  
  - new option: **dynamic quantization** (experimental)  
    reduces memory usage and increases performance  
    enable in settings -> compute settings  
    best used together with torch compile: *inductor*  
    this feature is highly experimental and will evolve over time  
    requires nightly versions of `torch` and `torchao`  
    > `pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121`  
    > `pip install -U git+https://github.com/pytorch-labs/ao`  
  - new option: **compile text encoder** (experimental)  
- **correction**  
  - new section in generate, allows for image corrections during generataion directly in latent space  
  - adds *brightness*, *sharpness* and *color* controls, thanks @AI-Casanova
  - adds *color grading* controls, thanks @AI-Casanova
  - replaces old **hdr** section
- **IPEX**, thanks @disty0  
  - see [wiki](https://github.com/vladmandic/automatic/wiki/Intel-ARC) for details  
  - rewrite ipex hijacks without CondFunc  
    improves compatibilty and performance  
    fixes random memory leaks  
  - out of the box support for Intel Data Center GPU Max Series  
  - remove IPEX / Torch 2.0 specific hijacks  
  - add `IPEX_SDPA_SLICE_TRIGGER_RATE`, `IPEX_ATTENTION_SLICE_RATE` and `IPEX_FORCE_ATTENTION_SLICE` env variables  
  - disable 1024x1024 workaround if the GPU supports 64 bit  
  - fix lock-ups at very high resolutions  
- **OpenVINO**, thanks @disty0  
  - see [wiki](https://github.com/vladmandic/automatic/wiki/OpenVINO) for details  
  - **quantization support with NNCF**  
    run 8 bit directly without autocast  
    enable *OpenVINO Quantize Models with NNCF* from *Compute Settings*  
  - **4-bit support with NNCF**  
    enable *Compress Model weights with NNCF* from *Compute Settings* and set a 4-bit NNCF mode  
    select both CPU and GPU from the device selection if you want to use 4-bit or 8-bit modes on GPU  
  - experimental support for *Text Encoder* compiling  
    OpenVINO is faster than IPEX now  
  - update to OpenVINO 2023.3.0  
  - add device selection to `Compute Settings`  
    selecting multiple devices will use `HETERO` device  
  - remove `OPENVINO_TORCH_BACKEND_DEVICE` env variable  
  - reduce system memory usage after compile  
  - fix cache loading with multiple models  
- **Olive** support, thanks @lshqqytiger
  - fully merged in in [wiki](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime-&-Olive), see wiki for details  
  - as a highlight, 4-5 it/s using DirectML on AMD GPU translates to 23-25 it/s using ONNX/Olive!  
- **fixes**  
  - civitai model download: enable downloads of embeddings
  - ipadapter: allow changing of model/image on-the-fly  
  - ipadapter: fix fallback of cross-attention on unload  
  - rebasin iterations, thanks @AI-Casanova
  - prompt scheduler, thanks @AI-Casanova
  - python: fix python 3.9 compatibility  
  - sdxl: fix positive prompt embeds
  - img2img: clip and blip interrogate  
  - img2img: sampler selection offset  
  - img2img: support variable aspect ratio without explicit resize  
  - cli: add `simple-upscale.py` script  
  - cli: fix cmd args parsing  
  - cli: add `run-benchmark.py` script  
  - api: add `/sdapi/v1/version` endpoint
  - api: add `/sdapi/v1/platform` endpoint
  - api: return current image in progress api if requested  
  - api: sanitize response object  
  - api: cleanup error logging  
  - api: fix api-only errors  
  - api: fix image to base64
  - api: fix upscale  
  - refiner: fix use of sd15 model as refiners in second pass  
  - refiner: enable none as option in xyz grid  
  - sampler: add sampler options info to metadata
  - sampler: guard against invalid sampler index  
  - sampler: add img2img_extra_noise option
  - config: reset default cfg scale to 6.0  
  - hdr: fix math, thanks @AI-Casanova
  - processing: correct display metadata  
  - processing: fix batch file names  
  - live preview: fix when using `bfloat16`  
  - live preview: add thread locking  
  - upscale: fix ldsr
  - huggingface: handle fallback model variant on load  
  - reference: fix links to models and use safetensors where possible  
  - model merge: unbalanced models where not all keys are present, thanks @AI-Casanova
  - better sdxl model detection
  - global crlf->lf switch  
  - model type switch if there is loaded submodels  
  - cleanup samplers use of compute devices, thanks @Disty0  
- **other**  
  - extensions `sd-webui-controlnet` is locked to commit `ecd33eb` due to breaking changes  
  - extension `stable-diffusion-webui-images-browser` is locked to commit `27fe4a7` due to breaking changes  
  - updated core requirements  
  - fully dynamic pipelines  
    pipeline switch is now done on-the-fly and does not require manual initialization of individual components  
    this allows for quick implementation of new pipelines  
    see `modules/sd_models.py:switch_pipe` for details  
  - major internal ui module refactoring  
    this may cause compatibility issues if an extension is doing a direct import from `ui.py`  
    in which case, report it so we can add a compatibility layer  
  - major public api refactoring  
    this may cause compatibility issues if an extension is doing a direct import from `api.py` or `models.py`  
    in which case, report it so we can add a compatibility layer  

## Update for 2023-12-29

To wrap up this amazing year, were releasing a new version of [SD.Next](https://github.com/vladmandic/automatic), this one is absolutely massive!  

### Highlights 2023-12-29

- Brand new Control module for *text, image, batch and video* processing  
  Native implementation of all control methods for both *SD15* and *SD-XL*  
  ▹ **ControlNet | ControlNet XS | Control LLLite | T2I Adapters | IP Adapters**  
  For details, see [Wiki](https://github.com/vladmandic/automatic/wiki/Control) documentation:  
- Support for new models types out-of-the-box  
  This brings number of supported t2i/i2i model families to 13!  
  ▹ **Stable Diffusion 1.5/2.1 | SD-XL | LCM | Segmind | Kandinsky | Pixart-α | Würstchen | aMUSEd | DeepFloyd IF | UniDiffusion | SD-Distilled | BLiP Diffusion | etc.**  
- New video capabilities:  
  ▹ **AnimateDiff | SVD | ModelScope | ZeroScope**  
- Enhanced platform support  
  ▹ **Windows | Linux | MacOS** with **nVidia | AMD | IntelArc | DirectML | OpenVINO | ONNX+Olive** backends  
- Better onboarding experience (first install)  
  with all model types available for single click download & load (networks -> reference)  
- Performance optimizations!
  For comparisment of different processing options and compile backends, see [Wiki](https://github.com/vladmandic/automatic/wiki/Benchmark)  
  As a highlight, were reaching **~100 it/s** (no tricks, this is with full features enabled and end-to-end on a standard nVidia RTX4090)  
- New [custom pipelines](https://github.com/vladmandic/automatic/blob/dev/scripts/example.py) framework for quickly porting any new pipeline  

And others improvements in areas such as: Upscaling (up to 8x now with 40+ available upscalers), Inpainting (better quality), Prompt scheduling, new Sampler options, new LoRA types, additional UI themes, better HDR processing, built-in Video interpolation, parallel Batch processing, etc.  

Plus some nifty new modules such as **FaceID** automatic face guidance using embeds during generation and **Depth 3D** image to 3D scene

### Full ChangeLog 2023-12-29

- **Control**  
  - native implementation of all image control methods:  
    **ControlNet**, **ControlNet XS**, **Control LLLite**, **T2I Adapters** and **IP Adapters**  
  - top-level **Control** next to **Text** and **Image** generate  
  - supports all variations of **SD15** and **SD-XL** models  
  - supports *Text*, *Image*, *Batch* and *Video* processing  
  - for details and list of supported models and workflows, see Wiki documentation:  
    <https://github.com/vladmandic/automatic/wiki/Control>  
- **Diffusers**  
  - [Segmind Vega](https://huggingface.co/segmind/Segmind-Vega) model support  
    - small and fast version of **SDXL**, only 3.1GB in size!  
    - select from *networks -> reference*  
  - [aMUSEd 256](https://huggingface.co/amused/amused-256) and [aMUSEd 512](https://huggingface.co/amused/amused-512) model support  
    - lightweigt models that excel at fast image generation  
    - *note*: must select: settings -> diffusers -> generator device: unset
    - select from *networks -> reference*
  - [Playground v1](https://huggingface.co/playgroundai/playground-v1), [Playground v2 256](https://huggingface.co/playgroundai/playground-v2-256px-base), [Playground v2 512](https://huggingface.co/playgroundai/playground-v2-512px-base), [Playground v2 1024](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic) model support  
    - comparable to SD15 and SD-XL, trained from scratch for highly aesthetic images  
    - simply select from *networks -> reference* and use as usual  
  - [BLIP-Diffusion](https://dxli94.github.io/BLIP-Diffusion-website/)  
    - img2img model that can replace subjects in images using prompt keywords  
    - download and load by selecting from *networks -> reference -> blip diffusion*
    - in image tab, select `blip diffusion` script
  - [DemoFusion](https://github.com/PRIS-CV/DemoFusion) run your SDXL generations at any resolution!  
    - in **Text** tab select *script* -> *demofusion*  
    - *note*: GPU VRAM limits do not automatically go away so be careful when using it with large resolutions  
      in the future, expect more optimizations, especially related to offloading/slicing/tiling,  
      but at the moment this is pretty much experimental-only  
  - [AnimateDiff](https://github.com/guoyww/animatediff/)  
    - overall improved quality  
    - can now be used with *second pass* - enhance, upscale and hires your videos!  
  - [IP Adapter](https://github.com/tencent-ailab/IP-Adapter)  
    - add support for **ip-adapter-plus_sd15, ip-adapter-plus-face_sd15 and ip-adapter-full-face_sd15**  
    - can now be used in *xyz-grid*  
  - **Text-to-Video**  
    - in text tab, select `text-to-video` script  
    - supported models: **ModelScope v1.7b, ZeroScope v1, ZeroScope v1.1, ZeroScope v2, ZeroScope v2 Dark, Potat v1**  
      *if you know of any other t2v models youd like to see supported, let me know!*  
    - models are auto-downloaded on first use  
    - *note*: current base model will be unloaded to free up resources  
  - **Prompt scheduling** now implemented for Diffusers backend, thanks @AI-Casanova
  - **Custom pipelines** contribute by adding your own custom pipelines!  
    - for details, see fully documented example:  
      <https://github.com/vladmandic/automatic/blob/dev/scripts/example.py>  
  - **Schedulers**  
    - add timesteps range, changing it will make scheduler to be over-complete or under-complete  
    - add rescale betas with zero SNR option (applicable to Euler, Euler a and DDIM, allows for higher dynamic range)  
  - **Inpaint**  
    - improved quality when using mask blur and padding  
  - **UI**  
    - 3 new native UI themes: **orchid-dreams**, **emerald-paradise** and **timeless-beige**, thanks @illu_Zn
    - more dynamic controls depending on the backend (original or diffusers)  
      controls that are not applicable in current mode are now hidden  
    - allow setting of resize method directly in image tab  
      (previously via settings -> upscaler_for_img2img)  
- **Optional**
  - **FaceID** face guidance during generation  
    - also based on IP adapters, but with additional face detection and external embeddings calculation  
    - calculates face embeds based on input image and uses it to guide generation  
    - simply select from *scripts -> faceid*  
    - *experimental module*: requirements must be installed manually:  
        > pip install insightface ip_adapter  
  - **Depth 3D** image to 3D scene
    - delivered as an extension, install from extensions tab  
      <https://github.com/vladmandic/sd-extension-depth3d>  
    - creates fully compatible 3D scene from any image by using depth estimation  
      and creating a fully populated mesh  
    - scene can be freely viewed in 3D in the UI itself or downloaded for use in other applications  
  - [ONNX/Olive](https://github.com/vladmandic/automatic/wiki/ONNX-Olive)  
    - major work continues in olive branch, see wiki for details, thanks @lshqqytiger  
      as a highlight, 4-5 it/s using DirectML on AMD GPU translates to 23-25 it/s using ONNX/Olive!  
- **General**  
  - new **onboarding**  
    - if no models are found during startup, app will no longer ask to download default checkpoint  
      instead, it will show message in UI with options to change model path or download any of the reference checkpoints  
    - *extra networks -> models -> reference* section is now enabled for both original and diffusers backend  
  - support for **Torch 2.1.2** (release) and **Torch 2.3** (dev)  
  - **Process** create videos from batch or folder processing  
      supports *GIF*, *PNG* and *MP4* with full interpolation, scene change detection, etc.  
  - **LoRA**  
    - add support for block weights, thanks @AI-Casanova  
      example `<lora:SDXL_LCM_LoRA:1.0:in=0:mid=1:out=0>`  
    - add support for LyCORIS GLora networks  
    - add support for LoRA PEFT (*Diffusers*) networks  
    - add support for Lora-OFT (*Kohya*) and Lyco-OFT (*Kohaku*) networks  
    - reintroduce alternative loading method in settings: `lora_force_diffusers`  
    - add support for `lora_fuse_diffusers` if using alternative method  
      use if you have multiple complex loras that may be causing performance degradation  
      as it fuses lora with model during load instead of interpreting lora on-the-fly  
  - **CivitAI downloader** allow usage of access tokens for download of gated or private models  
  - **Extra networks** new *settting -> extra networks -> build info on first access*  
    indexes all networks on first access instead of server startup  
  - **IPEX**, thanks @disty0  
    - update to **Torch 2.1**  
      if you get file not found errors, set `DISABLE_IPEXRUN=1` and run the webui with `--reinstall`  
    - built-in *MKL* and *DPCPP* for IPEX, no need to install OneAPI anymore  
    - **StableVideoDiffusion** is now supported with IPEX  
    - **8 bit support with NNCF** on Diffusers backend  
    - fix IPEX Optimize not applying with Diffusers backend  
    - disable 32bit workarounds if the GPU supports 64bit  
    - add `DISABLE_IPEXRUN` and `DISABLE_IPEX_1024_WA` environment variables  
    - performance and compatibility improvements  
  - **OpenVINO**, thanks @disty0  
    - **8 bit support for CPUs**  
    - reduce System RAM usage  
    - update to Torch 2.1.2  
    - add *Directory for OpenVINO cache* option to *System Paths*  
    - remove Intel ARC specific 1024x1024 workaround  
  - **HDR controls**  
    - batch-aware for enhancement of multiple images or video frames  
    - available in image tab  
  - **Logging**
    - additional *TRACE* logging enabled via specific env variables  
      see <https://github.com/vladmandic/automatic/wiki/Debug> for details  
    - improved profiling  
      use with `--debug --profile`  
    - log output file sizes  
  - **Other**  
    - **API** several minor but breaking changes to API behavior to better align response fields, thanks @Trojaner
    - **Inpaint** add option `apply_overlay` to control if inpaint result should be applied as overlay or as-is  
      can remove artifacts and hard edges of inpaint area but also remove some details from original  
    - **chaiNNer** fix `NaN` issues due to autocast  
    - **Upscale** increase limit from 4x to 8x given the quality of some upscalers  
    - **Networks** fix sort  
    - reduced default **CFG scale** from 6 to 4 to be more out-of-the-box compatibile with LCM/Turbo models
    - disable google fonts check on server startup  
    - fix torchvision/basicsr compatibility  
    - fix styles quick save  
    - add hdr settings to metadata  
    - improve handling of long filenames and filenames during batch processing  
    - do not set preview samples when using via api  
    - avoid unnecessary resizes in img2img and inpaint  
    - safe handling of config updates avoid file corruption on I/O errors  
    - updated `cli/simple-txt2img.py` and `cli/simple-img2img.py` scripts  
    - save `params.txt` regardless of image save status  
    - update built-in log monitor in ui, thanks @midcoastal  
    - major CHANGELOG doc cleanup, thanks @JetVarimax  
    - major INSTALL doc cleanup, thanks JetVarimax  

## Update for 2023-12-04

Whats new? Native video in SD.Next via both **AnimateDiff** and **Stable-Video-Diffusion** - and including native MP4 encoding and smooth video outputs out-of-the-box, not just animated-GIFs.  
Also new is support for **SDXL-Turbo** as well as new **Kandinsky 3** models and cool latent correction via **HDR controls** for any *txt2img* workflows, best-of-class **SDXL model merge** using full ReBasin methods and further mobile UI optimizations.  

- **Diffusers**
  - **IP adapter**
    - lightweight native implementation of T2I adapters which can guide generation towards specific image style  
    - supports most T2I models, not limited to SD 1.5  
    - models are auto-downloaded on first use
    - for IP adapter support in *Original* backend, use standard *ControlNet* extension  
  - **AnimateDiff**
    - lightweight native implementation of AnimateDiff models:  
      *AnimateDiff 1.4, 1.5 v1, 1.5 v2, AnimateFace*
    - supports SD 1.5 only  
    - models are auto-downloaded on first use  
    - for video saving support, see video support section
    - can be combined with IP-Adapter for even better results!  
    - for AnimateDiff support in *Original* backend, use standard *AnimateDiff* extension  
  - **HDR latent control**, based on [article](https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space#long-prompts-at-high-guidance-scales-becoming-possible)  
    - in *Advanced* params
    - allows control of *latent clamping*, *color centering* and *range maximization*  
    - supported by *XYZ grid*  
  - [SD21 Turbo](https://huggingface.co/stabilityai/sd-turbo) and [SDXL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) support  
    - just set CFG scale (0.0-1.0) and steps (1-3) to a very low value  
    - compatible with original StabilityAI SDXL-Turbo or any of the newer merges
    - download safetensors or select from networks -> reference
  - [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) and [Stable Video Diffusion XT](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) support  
    - download using built-in model downloader or simply select from *networks -> reference*  
      support for manually downloaded safetensors models will be added later  
    - for video saving support, see video support section
    - go to *image* tab, enter input image and select *script* -> *stable video diffusion*
  - [Kandinsky 3](https://huggingface.co/kandinsky-community/kandinsky-3) support  
    - download using built-in model downloader or simply select from *networks -> reference*  
    - this model is absolutely massive at 27.5GB at fp16, so be patient  
    - model params count is at 11.9B (compared to SD-XL at 3.3B) and its trained on mixed resolutions from 256px to 1024px  
    - use either model offload or sequential cpu offload to be able to use it  
  - better autodetection of *inpaint* and *instruct* pipelines  
  - support long seconary prompt for refiner  
- **Video support**
  - applies to any model that supports video generation, e.g. AnimateDiff and StableVideoDiffusion  
  - support for **animated-GIF**, **animated-PNG** and **MP4**  
  - GIF and PNG can be looped  
  - MP4 can have additional padding at the start/end as well as motion-aware interpolated frames for smooth playback  
    interpolation is done using [RIFE](https://arxiv.org/abs/2011.06294) with native implementation in SD.Next  
    And its fast - interpolation from 16 frames with 10x frames to target 160 frames results takes 2-3sec
  - output folder for videos is in *settings -> image paths -> video*  
- **General**  
  - redesigned built-in profiler  
    - now includes both `python` and `torch` and traces individual functions  
    - use with `--debug --profile`  
  - **model merge** add **SD-XL ReBasin** support, thanks @AI-Casanova  
  - further UI optimizations for **mobile devices**, thanks @iDeNoh  
  - log level defaults to info for console and debug for log file  
  - better prompt display in process tab  
  - increase maximum lora cache values  
  - fix extra networks sorting
  - fix controlnet compatibility issues in original backend  
  - fix img2img/inpaint paste params  
  - fix save text file for manually saved images  
  - fix python 3.9 compatibility issues  

## Update for 2023-11-23

New release, primarily focused around three major new features: full **LCM** support, completely new **Model Merge** functionality and **Stable-fast** compile support  
Also included are several other improvements and large number of hotfixes - see full changelog for details  

- **Diffusers**  
  - **LCM** support for any *SD 1.5* or *SD-XL* model!  
    - download [lcm-lora-sd15](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/tree/main) and/or [lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl/tree/main)  
    - load for favorite *SD 1.5* or *SD-XL* model *(original LCM was SD 1.5 only, this is both)*  
    - load **lcm lora** *(note: lcm lora is processed differently than any other lora)*  
    - set **sampler** to **LCM**  
    - set number of steps to some low number, for SD-XL 6-7 steps is normally sufficient  
      note: LCM scheduler does not support steps higher than 50  
    - set CFG to between 1 and 2  
  - Add `cli/lcm-convert.py` script to convert any SD 1.5 or SD-XL model to LCM model  
    by baking in LORA and uploading to Huggingface, thanks @Disty0  
  - Support for [Stable Fast](https://github.com/chengzeyi/stable-fast) model compile on *Windows/Linux/WSL2* with *CUDA*  
    See [Wiki:Benchmark](https://github.com/vladmandic/automatic/wiki/Benchmark) for details and comparison  
    of different backends, precision modes, advanced settings and compile modes  
    *Hint*: **70+ it/s** is possible on *RTX4090* with no special tweaks  
  - Add additional pipeline types for manual model loads when loading from `safetensors`  
  - Updated logic for calculating **steps** when using base/hires/refiner workflows  
  - Improve **model offloading** for both model and sequential cpu offload when dealing with meta tensors
  - Safe model offloading for non-standard models  
  - Fix **DPM SDE** scheduler  
  - Better support for SD 1.5 **inpainting** models  
  - Add support for **OpenAI Consistency decoder VAE**
  - Enhance prompt parsing with long prompts and support for *BREAK* keyword  
    Change-in-behavior: new line in prompt now means *BREAK*  
  - Add alternative Lora loading algorithm, triggered if `SD_LORA_DIFFUSERS` is set  
- **Models**
  - **Model merge**
    - completely redesigned, now based on best-of-class `meh` by @s1dlx  
      and heavily modified for additional functionality and fully integrated by @AI-Casanova (thanks!)  
    - merge SD or SD-XL models using *simple merge* (12 methods),  
      using one of *presets* (20 built-in presets) or custom block merge values  
    - merge with ReBasin permutations and/or clipping protection  
    - fully multithreaded for fastest merge possible  
  - **Model update**  
    - under UI -> Models - Update  
    - scan existing models for updated metadata on CivitAI and  
      provide download functionality for models with available  
- **Extra networks**  
  - Use multi-threading for 5x load speedup  
  - Better Lora trigger words support  
  - Auto refresh styles on change  
- **General**  
  - Many **mobile UI** optimizations, thanks @iDeNoh
  - Support for **Torch 2.1.1** with CUDA 12.1 or CUDA 11.8  
  - Configurable location for HF cache folder  
    Default is standard `~/.cache/huggingface/hub`  
  - Reworked parser when pasting previously generated images/prompts  
    includes all `txt2img`, `img2img` and `override` params  
  - Reworked **model compile**
  - Support custom upscalers in subfolders  
  - Add additional image info when loading image in process tab  
  - Better file locking when sharing config and/or models between multiple instances  
  - Handle custom API endpoints when using auth  
  - Show logged in user in log when accessing via UI and/or API  
  - Support `--ckpt none` to skip loading a model  
- **XYZ grid**
  - Add refiner options to XYZ Grid  
  - Add option to create only subgrids in XYZ grid, thanks @midcoastal
  - Allow custom font, background and text color in settings
- **Fixes**  
  - Fix `params.txt` saved before actual image
  - Fix inpaint  
  - Fix manual grid image save  
  - Fix img2img init image save  
  - Fix upscale in txt2img for batch counts when no hires is used  
  - More uniform models paths  
  - Safe scripts callback execution  
  - Improved extension compatibility  
  - Improved BF16 support  
  - Match previews for reference models with downloaded models

## Update for 2023-11-06

Another pretty big release, this time with focus on new models (3 new model types), new backends and optimizations
Plus quite a few fixes  

Also, [Wiki](https://github.com/vladmandic/automatic/wiki) has been updated with new content, so check it out!  
Some highlights: [OpenVINO](https://github.com/vladmandic/automatic/wiki/OpenVINO), [IntelArc](https://github.com/vladmandic/automatic/wiki/Intel-ARC), [DirectML](https://github.com/vladmandic/automatic/wiki/DirectML), [ONNX/Olive](https://github.com/vladmandic/automatic/wiki/ONNX-Olive)

- **Diffusers**
  - since now **SD.Next** supports **12** different model types, weve added reference model for each type in  
    *Extra networks -> Reference* for easier select & auto-download  
    Models can still be downloaded manually, this is just a convenience feature & a showcase for supported models  
  - new model type: [Segmind SSD-1B](https://huggingface.co/segmind/SSD-1B)  
    its a *distilled* model trained at 1024px, this time 50% smaller and faster version of SD-XL!  
    (and quality does not suffer, its just more optimized)  
    test shows batch-size:4 with 1k images at full quality used less than 6.5GB of VRAM  
    and for further optimization, you can use built-in **TAESD** decoder,  
    which results in batch-size:16 with 1k images using 7.9GB of VRAM
    select from extra networks -> reference or download using built-in **Huggingface** downloader: `segmind/SSD-1B`  
  - new model type: [Pixart-α XL 2](https://github.com/PixArt-alpha/PixArt-alpha)  
    in medium/512px and large/1024px variations  
    comparable in quality to SD 1.5 and SD-XL, but with better text encoder and highly optimized training pipeline  
    so finetunes can be done in as little as 10% compared to SD/SD-XL (note that due to much larger text encoder, it is a large model)  
    select from extra networks -> reference or download using built-in **Huggingface** downloader: `PixArt-alpha/PixArt-XL-2-1024-MS`  
  - new model type: [LCM: Latent Consistency Models](https://github.com/openai/consistency_models)  
    trained at 512px, but with near-instant generate in a as little as 3 steps!  
    combined with OpenVINO, generate on CPU takes less than 5-10 seconds: <https://www.youtube.com/watch?v=b90ESUTLsRo>  
    and absolute beast when combined with **HyperTile** and **TAESD** decoder resulting in **28 FPS**  
    (on RTX4090 for batch 16x16 at 512px)  
    note: set sampler to **Default** before loading model as LCM comes with its own *LCMScheduler* sampler  
    select from extra networks -> reference or download using built-in **Huggingface** downloader: `SimianLuo/LCM_Dreamshaper_v7`  
  - support for **Custom pipelines**, thanks @disty0  
    download using built-in **Huggingface** downloader  
    think of them as plugins for diffusers not unlike original extensions that modify behavior of `ldm` backend  
    list of community pipelines: <https://github.com/huggingface/diffusers/blob/main/examples/community/README.md>  
  - new custom pipeline: `Disty0/zero123plus-pipeline`, thanks @disty0  
    generate 4 output images with different camera positions: front, side, top, back!  
    for more details, see <https://github.com/vladmandic/automatic/discussions/2421>  
  - new backend: **ONNX/Olive** *(experimental)*, thanks @lshqqytiger  
    for details, see [WiKi](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime)
  - extend support for [Free-U](https://github.com/ChenyangSi/FreeU)  
    improve generations quality at no cost (other than finding params that work for you)  
- **General**  
  - attempt to auto-fix invalid samples which occur due to math errors in lower precision  
    example: `RuntimeWarning: invalid value encountered in cast: sample = sample.astype(np.uint8)`  
    begone **black images** *(note: if it proves as working, this solution will need to be expanded to cover all scenarios)*  
  - add **Lora OFT** support, thanks @antis0007 and @ai-casanova  
  - **Upscalers**  
    - **compile** option, thanks @disty0  
    - **chaiNNer** add high quality models from [Helaman](https://openmodeldb.info/users/helaman)  
  - redesigned **Progress bar** with full details on current operation  
  - new option: *settings -> images -> keep incomplete*  
    can be used to skip vae decode on aborted/skipped/interrupted image generations  
  - new option: *settings -> system paths -> models*  
    can be used to set custom base path for *all* models (previously only as cli option)  
  - remove external clone of items in `/repositories`  
  - **Interrogator** module has been removed from `extensions-builtin`  
    and fully implemented (and improved) natively  
- **UI**  
  - UI tweaks for default themes  
  - UI switch core font in default theme to **noto-sans**  
    previously default font was simply *system-ui*, but it lead to too much variations between browsers and platforms  
  - UI tweaks for mobile devices, thanks @iDeNoh  
  - updated **Context menu**  
    right-click on any button in action menu (e.g. generate button)  
- **Extra networks**  
  - sort by name, size, date, etc.  
  - switch between *gallery* and *list* views  
  - add tags from user metadata (in addition to tags in model metadata) for **lora**  
  - added **Reference** models for diffusers backend  
  - faster enumeration of all networks on server startup  
- **Packages**
  - updated `diffusers` to 0.22.0, `transformers` to 4.34.1  
  - update **openvino**, thanks @disty0  
  - update **directml**, @lshqqytiger  
- **Compute**  
  - **OpenVINO**:  
    - updated to mainstream `torch` *2.1.0*  
    - support for **ESRGAN** upscalers  
- **Fixes**  
  - fix **freeu** for backend original and add it to xyz grid  
  - fix loading diffuser models in huggingface format from non-standard location  
  - fix default styles looking in wrong location  
  - fix missing upscaler folder on initial startup  
  - fix handling of relative path for models  
  - fix simple live preview device mismatch  
  - fix batch img2img  
  - fix diffusers samplers: dpm++ 2m, dpm++ 1s, deis  
  - fix new style filename template  
  - fix image name template using model name  
  - fix image name sequence  
  - fix model path using relative path  
  - fix safari/webkit layour, thanks @eadnams22
  - fix `torch-rocm` and `tensorflow-rocm` version detection, thanks @xangelix  
  - fix **chainner** upscalers color clipping  
  - fix for base+refiner workflow in diffusers mode: number of steps, diffuser pipe mode  
  - fix for prompt encoder with refiner in diffusers mode  
  - fix prompts-from-file saving incorrect metadata  
  - fix add/remove extra networks to prompt
  - fix before-hires step  
  - fix diffusers switch from invalid model  
  - force second requirements check on startup  
  - remove **lyco**, multiple_tqdm  
  - enhance extension compatibility for extensions directly importing codeformers  
  - enhance extension compatibility for extensions directly accessing processing params  
  - **css** fixes  
  - clearly mark external themes in ui  
  - update `typing-extensions`  

## Update for 2023-10-17

This is a major release, with many changes and new functionality...  

Changelog is massive, but do read through or youll be missing on some very cool new functionality  
or even free speedups and quality improvements (regardless of which workflows youre using)!  

Note that for this release its recommended to perform a clean install (e.g. fresh `git clone`)  
Upgrades are still possible and supported, but clean install is recommended for best experience  

- **UI**  
  - added **change log** to UI  
    see *System -> Changelog*  
  - converted submenus from checkboxes to accordion elements  
    any ui state including state of open/closed menus can be saved as default!  
    see *System -> User interface -> Set menu states*  
  - new built-in theme **invoked**  
    thanks @BinaryQuantumSoul  
  - add **compact view** option in settings -> user interface  
  - small visual indicator bottom right of page showing internal server job state  
- **Extra networks**:  
  - **Details**  
    - new details interface to view and save data about extra networks  
      main ui now has a single button on each en to trigger details view  
    - details view includes model/lora metadata parser!  
    - details view includes civitai model metadata!  
  - **Metadata**:  
    - you can scan [civitai](https://civitai.com/)  
      for missing metadata and previews directly from extra networks  
      simply click on button in top-right corner of extra networks page  
  - **Styles**  
    - save/apply icons moved to extra networks  
    - can be edited in details view  
    - support for single or multiple styles per json  
    - support for embedded previews  
    - large database of art styles included by default  
      can be disabled in *settings -> extra networks -> show built-in*  
    - styles can also be used in a prompt directly: `<style:style_name>`  
      if style if an exact match, it will be used  
      otherwise it will rotate between styles that match the start of the name  
      that way you can use different styles as wildcards when processing batches  
    - styles can have **extra** fields, not just prompt and negative prompt  
      for example: *"Extra: sampler: Euler a, width: 480, height: 640, steps: 30, cfg scale: 10, clip skip: 2"*
  - **VAE**  
    - VAEs are now also listed as part of extra networks  
    - Image preview methods have been redesigned: simple, approximate, taesd, full  
      please set desired preview method in settings  
    - both original and diffusers backend now support "full quality" setting  
      if you desired model or platform does not support FP16 and/or you have a low-end hardware and cannot use FP32  
      you can disable "full quality" in advanced params and it will likely reduce decode errors (infamous black images)  
  - **LoRA**  
    - LoRAs are now automatically filtered based on compatibility with currently loaded model  
      note that if lora type cannot be auto-determined, it will be left in the list  
  - **Refiner**  
    - you can load model from extra networks as base model or as refiner  
      simply select button in top-right of models page  
  - **General**  
    - faster search, ability to show/hide/sort networks  
    - refactored subfolder handling  
      *note*: this will trigger model hash recalculation on first model use  
- **Diffusers**:  
  - better pipeline **auto-detect** when loading from safetensors  
  - **SDXL Inpaint**  
    - although any model can be used for inpainiting, there is a case to be made for  
      dedicated inpainting models as they are tuned to inpaint and not generate  
    - model can be used as base model for **img2img** or refiner model for **txt2img**  
      To download go to *Models -> Huggingface*:  
      - `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` *(6.7GB)*  
  - **SDXL Instruct-Pix2Pix**  
    - model can be used as base model for **img2img** or refiner model for **txt2img**  
      this model is massive and requires a lot of resources!  
      to download go to *Models -> Huggingface*:  
      - `diffusers/sdxl-instructpix2pix-768` *(11.9GB)*  
  - **SD Latent Upscale**  
    - you can use *SD Latent Upscale* models as **refiner models**  
      this is a bit experimental, but it works quite well!  
      to download go to *Models -> Huggingface*:  
      - `stabilityai/sd-x2-latent-upscaler` *(2.2GB)*  
      - `stabilityai/stable-diffusion-x4-upscaler` *(1.7GB)*  
  - better **Prompt attention**  
    should better handle more complex prompts  
    for sdxl, choose which part of prompt goes to second text encoder - just add `TE2:` separator in the prompt  
    for hires and refiner, second pass prompt is used if present, otherwise primary prompt is used  
    new option in *settings -> diffusers -> sdxl pooled embeds*  
    thanks @AI-Casanova  
  - better **Hires** support for SD and SDXL  
  - better **TI embeddings** support for SD and SDXL  
    faster loading, wider compatibility and support for embeddings with multiple vectors  
    information about used embedding is now also added to image metadata  
    thanks @AI-Casanova  
  - better **Lora** handling  
    thanks @AI-Casanova  
  - better **SDXL preview** quality (approx method)  
    thanks @BlueAmulet
  - new setting: *settings -> diffusers -> force inpaint*  
    as some models behave better when in *inpaint* mode even for normal *img2img* tasks  
- **Upscalers**:
  - pretty much a rewrite and tons of new upscalers - built-in list is now at **42**  
  - fix long outstanding memory leak in legacy code, amazing this went undetected for so long  
  - more high quality upscalers available by default  
    **SwinIR** (2), **ESRGAN** (12), **RealESRGAN** (6), **SCUNet** (2)  
  - if that is not enough, there is new **chaiNNer** integration:  
    adds 15 more upscalers from different families out-of-the-box:  
    **HAT** (6), **RealHAT** (2), **DAT** (1), **RRDBNet** (1), **SPSRNet** (1), **SRFormer** (2), **SwiftSR** (2)  
    and yes, you can download and add your own, just place them in `models/chaiNNer`  
  - two additional latent upscalers based on SD upscale models when using Diffusers backend  
    **SD Upscale 2x**, **SD Upscale 4x***  
    note: Recommended usage for *SD Upscale* is by using second pass instead of upscaler  
    as it allows for tuning of prompt, seed, sampler settings which are used to guide upscaler  
  - upscalers are available in **xyz grid**  
  - simplified *settings->postprocessing->upscalers*  
    e.g. all upsamplers share same settings for tiling  
  - allow upscale-only as part of **txt2img** and **img2img** workflows  
    simply set *denoising strength* to 0 so hires does not get triggered  
  - unified init/download/execute/progress code  
  - easier installation  
- **Samplers**:  
  - moved ui options to submenu  
  - default list for new installs is now all samplers, list can be modified in settings  
  - simplified samplers configuration in settings  
    plus added few new ones like sigma min/max which can highly impact sampler behavior  
  - note that list of samplers is now *different* since keeping a flat-list of all possible  
    combinations results in 50+ samplers which is not practical  
    items such as algorithm (e.g. karras) is actually a sampler option, not a sampler itself  
- **CivitAI**:
  - civitai model download is now multithreaded and resumable  
    meaning that you can download multiple models in parallel  
    as well as resume aborted/incomplete downloads  
  - civitai integration in *models -> civitai* can now find most  
    previews AND metadata for most models (checkpoints, loras, embeddings)  
    metadata is now parsed and saved in *[model].json*  
    typical hit rate is >95% for models, loras and embeddings  
  - description from parsed model metadata is used as model description if there is no manual  
    description file present in format of *[model].txt*  
  - to enable search for models, make sure all models have set hash values  
    *Models -> Valida -> Calculate hashes*  
- **LoRA**
  - new unified LoRA handler for all LoRA types (lora, lyco, loha, lokr, locon, ia3, etc.)  
    applies to both original and diffusers backend  
    thanks @AI-Casanova for diffusers port  
  - for *backend:original*, separate lyco handler has been removed  
- **Compute**  
  - **CUDA**:  
    - default updated to `torch` *2.1.0* with cuda *12.1*  
    - testing moved to `torch` *2.2.0-dev/cu122*  
    - check out *generate context menu -> show nvml* for live gpu stats (memory, power, temp, clock, etc.)
  - **Intel Arc/IPEX**:  
    - tons of optimizations, built-in binary wheels for Windows  
      i have to say, intel arc/ipex is getting to be quite a player, especially with openvino  
      thanks @Disty0 @Nuullll  
  - **AMD ROCm**:  
    - updated installer to support detect `ROCm` *5.4/5.5/5.6/5.7*  
    - support for `torch-rocm-5.7`
  - **xFormers**:
    - default updated to *0.0.23*  
    - note that latest xformers are still not compatible with cuda 12.1  
      recommended to use torch 2.1.0 with cuda 11.8  
      if you attempt to use xformers with cuda 12.1, it will force a full xformers rebuild on install  
      which can take a very long time and may/may-not work  
    - added cmd param `--use-xformers` to force usage of exformers  
  - **GC**:  
    - custom garbage collect threshold to reduce vram memory usage, thanks @Disty0  
      see *settings -> compute -> gc*  
- **Inference**  
  - new section in **settings**  
    - [HyperTile](https://github.com/tfernd/HyperTile): new!  
      available for *diffusers* and *original* backends  
      massive (up to 2x) speed-up your generations for free :)  
      *note: hypertile is not compatible with any extension that modifies processing parameters such as resolution*  
      thanks @tfernd
    - [Free-U](https://github.com/ChenyangSi/FreeU): new!  
      available for *diffusers* and *original* backends  
      improve generations quality at no cost (other than finding params that work for you)  
      *note: temporarily disabled for diffusers pending release of diffusers==0.22*  
      thanks @ljleb  
    - [Token Merging](https://github.com/dbolya/tomesd): not new, but updated  
      available for *diffusers* and *original* backends  
      speed-up your generations by merging redundant tokens  
      speed up will depend on how aggressive you want to be with token merging  
    - **Batch mode**  
      new option *settings -> inference -> batch mode*  
      when using img2img process batch, optionally process multiple images in batch in parallel  
      thanks @Symbiomatrix
- **NSFW Detection/Censor**  
  - install extension: [NudeNet](https://github.com/vladmandic/sd-extension-nudenet)  
    body part detection, image metadata, advanced censoring, etc...  
    works for *text*, *image* and *process* workflows  
    more in the extension notes  
- **Extensions**
  - automatic discovery of new extensions on github  
    no more waiting for them to appear in index!
  - new framework for extension validation  
    extensions ui now shows actual status of extensions for reviewed extensions  
    if you want to contribute/flag/update extension status, reach out on github or discord  
  - better overall compatibility with A1111 extensions (up to a point)  
  - [MultiDiffusion](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)  
    has been removed from list of built-in extensions  
    you can still install it manually if desired  
  - [LyCORIS]<https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris>  
    has been removed from list of built-in extensions  
    it is considered obsolete given that all functionality is now built-in  
- **General**  
  - **Startup**  
    - all main CLI parameters can now be set as environment variable as well  
      for example `--data-dir <path>` can be specified as `SD_DATADIR=<path>` before starting SD.Next  
  - **XYZ Grid**
    - more flexibility to use selection or strings  
  - **Logging**  
    - get browser session info in server log  
    - allow custom log file destination  
      see `webui --log`  
    - when running with `--debug` flag, log is force-rotated  
      so each `sdnext.log.*` represents exactly one server run  
    - internal server job state tracking  
  - **Launcher**  
    - new `webui.ps1` powershell launcher for windows (old `webui.bat` is still valid)  
      thanks @em411  
  - **API**
    - add end-to-end example how to use API: `cli/simple-txt2img.js`  
      covers txt2img, upscale, hires, refiner  
  - **train.py**
    - wrapper script around built-in **kohyas lora** training script  
      see `cli/train.py --help`  
      new support for sd and sdxl, thanks @evshiron  
      new support for full offline mode (without sdnext server running)  
- **Themes**
  - all built-in themes are fully supported:  
    - *black-teal (default), light-teal, black-orange, invoked, amethyst-nightfall, midnight-barbie*  
  - if youre using any **gradio default** themes or a **3rd party** theme or  that are not optimized for SD.Next, you may experience issues  
    default minimal style has been updated for compatibility, but actual styling is completely outside of SD.Next control  

## Update for 2023-09-13

Started as a mostly a service release with quite a few fixes, but then...  
Major changes how **hires** works as well as support for a very interesting new model [Wuerstchen](https://huggingface.co/blog/wuertschen)  

- tons of fixes  
- changes to **hires**  
  - enable non-latent upscale modes (standard upscalers)  
  - when using latent upscale, hires pass is run automatically  
  - when using non-latent upscalers, hires pass is skipped by default  
    enabled using **force hires** option in ui  
    hires was not designed to work with standard upscalers, but i understand this is a common workflow  
  - when using refiner, upscale/hires runs before refiner pass  
  - second pass can now also utilize full/quick vae quality  
  - note that when combining non-latent upscale, hires and refiner output quality is maximum,  
    but operations are really resource intensive as it includes: *base->decode->upscale->encode->hires->refine*
  - all combinations of: decode full/quick + upscale none/latent/non-latent + hires on/off + refiner on/off  
    should be supported, but given the number of combinations, issues are possible  
  - all operations are captured in image metadata
- diffusers:
  - allow loading of sd/sdxl models from safetensors without online connectivity
  - support for new model: [wuerstchen](https://huggingface.co/warp-ai/wuerstchen)  
    its a high-resolution model (1024px+) thats ~40% faster than sd-xl with a bit lower resource requirements  
    go to *models -> huggingface -> search "warp-ai/wuerstchen" -> download*  
    its nearly 12gb in size, so be patient :)
- minor re-layout of the main ui  
- updated **ui hints**  
- updated **models -> civitai**  
  - search and download loras  
  - find previews for already downloaded models or loras  
- new option **inference mode**  
  - default is standard `torch.no_grad`  
    new option is `torch.inference_only` which is slightly faster and uses less vram, but only works on some gpus  
- new cmdline param `--no-metadata`  
  skips reading metadata from models that are not already cached  
- updated **gradio**  
- **styles** support for subfolders  
- **css** optimizations
- clean-up **logging**  
  - capture system info in startup log  
  - better diagnostic output  
  - capture extension output  
  - capture ldm output  
  - cleaner server restart  
  - custom exception handling

## Update for 2023-09-06

One week later, another large update!

- system:  
  - full **python 3.11** support  
    note that changing python version does require reinstall  
    and if youre already on python 3.10, really no need to upgrade  
- themes:  
  - new default theme: **black-teal**  
  - new light theme: **light-teal**  
  - new additional theme: **midnight-barbie**, thanks @nyxia  
- extra networks:  
  - support for **tags**  
    show tags on hover, search by tag, list tags, add to prompt, etc.  
  - **styles** are now also listed as part of extra networks  
    existing `styles.csv` is converted upon startup to individual styles inside `models/style`  
    this is stage one of new styles functionality  
    old styles interface is still available, but will be removed in future  
  - cache file lists for much faster startup  
    speedups are 50+% for large number of extra networks  
  - ui refresh button now refreshes selected page, not all pages  
  - simplified handling of **descriptions**  
    now shows on-mouse-over without the need for user interaction  
  - **metadata** and **info** buttons only show if there is actual content  
- diffusers:  
  - add full support for **textual inversions** (embeddings)  
    this applies to both sd15 and sdxl  
    thanks @ai-casanova for porting compel/sdxl code  
  - mix&match **base** and **refiner** models (*experimental*):  
    most of those are "because why not" and can result in corrupt images, but some are actually useful  
    also note that if youre not using actual refiner model, you need to bump refiner steps  
    as normal models are not designed to work with low step count  
    and if youre having issues, try setting prompt parser to "fixed attention" as majority of problems  
    are due to token mismatches when using prompt attention  
    - any sd15 + any sd15  
    - any sd15 + sdxl-refiner  
    - any sdxl-base + sdxl-refiner  
    - any sdxl-base + any sd15  
    - any sdxl-base + any sdxl-base  
  - ability to **interrupt** (stop/skip) model generate  
  - added **aesthetics score** setting (for sdxl)  
    used to automatically guide unet towards higher pleasing images  
    highly recommended for simple prompts  
  - added **force zeros** setting  
    create zero-tensor for prompt if prompt is empty (positive or negative)  
- general:  
  - `rembg` remove backgrounds support for **is-net** model  
  - **settings** now show markers for all items set to non-default values  
  - **metadata** refactored how/what/when metadata is added to images  
    should result in much cleaner and more complete metadata  
  - pre-create all system folders on startup  
  - handle model load errors gracefully  
  - improved vram reporting in ui  
  - improved script profiling (when running in debug mode)  

## Update for 2023-08-30

Time for a quite a large update that has been leaking bit-by-bit over the past week or so...  
*Note*: due to large changes, it is recommended to reset (delete) your `ui-config.json`  

- diffusers:  
  - support for **distilled** sd models  
    just go to models/huggingface and download a model, for example:  
    `segmind/tiny-sd`, `segmind/small-sd`, `segmind/portrait-finetuned`  
    those are lower quality, but extremely small and fast  
    up to 50% faster than sd 1.5 and execute in as little as 2.1gb of vram  
- general:  
  - redesigned **settings**  
    - new layout with separated sections:  
      *settings, ui config, licenses, system info, benchmark, models*  
    - **system info** tab is now part of settings  
      when running outside of sdnext, system info is shown in main ui  
    - all system and image paths are now relative by default  
    - add settings validation when performing load/save  
    - settings tab in ui now shows settings that are changed from default values  
    - settings tab switch to compact view  
  - update **gradio** major version  
    this may result in some smaller layout changes since its a major version change  
    however, browser page load is now much faster  
  - optimizations:
    - optimize model hashing  
    - add cli param `--skip-all` that skips all installer checks  
      use at personal discretion, but it can be useful for bulk deployments  
    - add model **precompile** option (when model compile is enabled)  
    - **extra network** folder info caching  
      results in much faster startup when you have large number of extra networks  
    - faster **xyz grid** switching  
      especially when using different checkpoints  
  - update **second pass** options for clarity
  - models:
    - civitai download missing model previews
  - add **openvino** (experimental) cpu optimized model compile and inference  
    enable with `--use-openvino`  
    thanks @disty0  
  - enable batch **img2img** scale-by workflows  
    now you can batch process with rescaling based on each individual original image size  
  - fixes:
    - fix extra networks previews  
    - css fixes  
    - improved extensions compatibility (e.g. *sd-cn-animation*)  
    - allow changing **vae** on-the-fly for both original and diffusers backend

## Update for 2023-08-20

Another release thats been baking in dev branch for a while...

- general:
  - caching of extra network information to enable much faster create/refresh operations  
    thanks @midcoastal
- diffusers:
  - add **hires** support (*experimental*)  
    applies to all model types that support img2img, including **sd** and **sd-xl**  
    also supports all hires upscaler types as well as standard params like steps and denoising strength  
    when used with **sd-xl**, it can be used with or without refiner loaded  
    how to enable - there are no explicit checkboxes other than second pass itself:
    - hires: upscaler is set and target resolution is not at default  
    - refiner: if refiner model is loaded  
  - images save options: *before hires*, *before refiner*
  - redo `move model to cpu` logic in settings -> diffusers to be more reliable  
    note that system defaults have also changed, so you may need to tweak to your liking  
  - update dependencies

## Update for 2023-08-17

Smaller update, but with some breaking changes (to prepare for future larger functionality)...

- general:
  - update all metadata saved with images  
    see <https://github.com/vladmandic/automatic/wiki/Metadata> for details  
  - improved **amd** installer with support for **navi 2x & 3x** and **rocm 5.4/5.5/5.6**  
    thanks @evshiron  
  - fix **img2img** resizing (applies to *original, diffusers, hires*)  
  - config change: main `config.json` no longer contains entire configuration  
    but only differences from defaults (similar to recent change performed to `ui-config.json`)  
- diffusers:
  - enable **batch img2img** workflows  
- original:  
  - new samplers: **dpm++ 3M sde** (standard and karras variations)  
    enable in *settings -> samplers -> show samplers*
  - expose always/never discard penultimate sigma  
    enable in *settings -> samplers*  

## Update for 2023-08-11

This is a big one thats been cooking in `dev` for a while now, but finally ready for release...

- diffusers:
  - **pipeline autodetect**
    if pipeline is set to autodetect (default for new installs), app will try to autodetect pipeline based on selected model  
    this should reduce user errors such as loading **sd-xl** model when **sd** pipeline is selected  
  - **quick vae decode** as alternative to full vae decode which is very resource intensive  
    quick decode is based on `taesd` and produces lower quality, but its great for tests or grids as it runs much faster and uses far less vram  
    disabled by default, selectable in *txt2img/img2img -> advanced -> full quality*  
  - **prompt attention** for sd and sd-xl  
    supports both `full parser` and native `compel`  
    thanks @ai-casanova  
  - advanced **lora load/apply** methods  
    in addition to standard lora loading that was recently added to sd-xl using diffusers, now we have  
    - **sequential apply** (load & apply multiple loras in sequential manner) and  
    - **merge and apply** (load multiple loras and merge before applying to model)  
    see *settings -> diffusers -> lora methods*  
    thanks @hameerabbasi and @ai-casanova  
  - **sd-xl vae** from safetensors now applies correct config  
    result is that 3rd party vaes can be used without washed out colors  
  - options for optimized memory handling for lower memory usage  
    see *settings -> diffusers*
- general:
  - new **civitai model search and download**  
    native support for civitai, integrated into ui as *models -> civitai*  
  - updated requirements  
    this time its a bigger change so upgrade may take longer to install new requirements
  - improved **extra networks** performance with large number of networks

## Update for 2023-08-05

Another minor update, but it unlocks some cool new items...

- diffusers:
  - vaesd live preview (sd and sd-xl)  
  - fix inpainting (sd and sd-xl)  
- general:
  - new torch 2.0 with ipex (intel arc)  
  - additional callbacks for extensions  
    enables latest comfyui extension  

## Update for 2023-07-30

Smaller release, but IMO worth a post...

- diffusers:
  - sd-xl loras are now supported!
  - memory optimizations: Enhanced sequential CPU offloading, model CPU offload, FP16 VAE
    - significant impact if running SD-XL (for example, but applies to any model) with only 8GB VRAM
  - update packages
- minor bugfixes

## Update for 2023-07-26

This is a big one, new models, new diffusers, new features and updated UI...

First, **SD-XL 1.0** is released and yes, SD.Next supports it out of the box!

- [SD-XL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors)
- [SD-XL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors)

Also fresh is new **Kandinsky 2.2** model that does look quite nice:

- [Kandinsky Decoder](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder)
- [Kandinsky Prior](https://huggingface.co/kandinsky-community/kandinsky-2-2-prior)

Actual changelog is:

- general:
  - new loading screens and artwork
  - major ui simplification for both txt2img and img2img  
    nothing is removed, but you can show/hide individual sections  
    default is very simple interface, but you can enable any sections and save it as default in settings  
  - themes: add additional built-in theme, `amethyst-nightfall`
  - extra networks: add add/remove tags to prompt (e.g. lora activation keywords)
  - extensions: fix couple of compatibility items
  - firefox compatibility improvements
  - minor image viewer improvements
  - add backend and operation info to metadata

- diffusers:
  - were out of experimental phase and diffusers backend is considered stable  
  - sd-xl: support for **sd-xl 1.0** official model
  - sd-xl: loading vae now applies to both base and refiner and saves a bit of vram  
  - sd-xl: denoising_start/denoising_end
  - sd-xl: enable dual prompts  
    dual prompt is used if set regardless if refiner is enabled/loaded  
    if refiner is loaded & enabled, refiner prompt will also be used for refiner pass  
    - primary prompt goes to [OpenAI CLIP-ViT/L-14](https://huggingface.co/openai/clip-vit-large-patch14)
    - refiner prompt goes to [OpenCLIP-ViT/bigG-14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
  - **kandinsky 2.2** support  
    note: kandinsky model must be downloaded using model downloader, not as safetensors due to specific model format  
  - refiner: fix batch processing
  - vae: enable loading of pure-safetensors vae files without config  
    also enable *automatic* selection to work with diffusers  
  - sd-xl: initial lora support  
    right now this applies to official lora released by **stability-ai**, support for **kohyas** lora is expected soon  
  - implement img2img and inpainting (experimental)  
    actual support and quality depends on model  
    it works as expected for sd 1.5, but not so much for sd-xl for now  
  - implement limited stop/interrupt for diffusers
    works between stages, not within steps  
  - add option to save image before refiner pass  
  - option to set vae upcast in settings  
  - enable fp16 vae decode when using optimized vae  
    this pretty much doubles performance of decode step (delay after generate is done)  

- original
  - fix hires secondary sampler  
    this now fully obsoletes `fallback_sampler` and `force_hr_sampler_name`  


## Update for 2023-07-18

While were waiting for official SD-XL release, heres another update with some fixes and enhancements...

- **global**
  - image save: option to add invisible image watermark to all your generated images  
    disabled by default, can be enabled in settings -> image options  
    watermark information will be shown when loading image such as in process image tab  
    also additional cli utility `/cli/image-watermark.py` to read/write/strip watermarks from images  
  - batch processing: fix metadata saving, also allow to drag&drop images for batch processing  
  - ui configuration: you can modify all ui default values from settings as usual,  
    but only values that are non-default will be written to `ui-config.json`  
  - startup: add cmd flag to skip all `torch` checks  
  - startup: force requirements check on each server start  
    there are too many misbehaving extensions that change system requirements  
  - internal: safe handling of all config file read/write operations  
    this allows sdnext to run in fully shared environments and prevents any possible configuration corruptions  
- **diffusers**:
  - sd-xl: remove image watermarks autocreated by 0.9 model  
  - vae: enable loading of external vae, documented in diffusers wiki  
    and mix&match continues, you can even use sd-xl vae with sd 1.5 models!  
  - samplers: add concept of *default* sampler to avoid needing to tweak settings for primary or second pass  
    note that sampler details will be printed in log when running in debug level  
  - samplers: allow overriding of sampler beta values in settings  
  - refiner: fix refiner applying only to first image in batch  
  - refiner: allow using direct latents or processed output in refiner  
  - model: basic support for one more model: [UniDiffuser](https://github.com/thu-ml/unidiffuser)  
    download using model downloader: `thu-ml/unidiffuser-v1`  
    and set resolution to 512x512  

## Update for 2023-07-14

Trying to unify settings for both original and diffusers backend without introducing duplicates...

- renamed **hires fix** to **second pass**  
  as that is what it actually is, name hires fix is misleading to start with  
- actual **hires fix** and **refiner** are now options inside **second pass** section  
- obsoleted settings -> sampler -> **force_hr_sampler_name**  
  it is now part of **second pass** options and it works the same for both original and diffusers backend  
  which means you can use different scheduler settings for txt2img and hires if you want  
- sd-xl refiner will run if its loaded and if second pass is enabled  
  so you can quickly enable/disable refiner by simply enabling/disabling second pass  
- you can mix&match **model** and **refiner**  
  for example, you can generate image using sd 1.5 and still use sd-xl refiner as second pass  
- reorganized settings -> samplers to show which section refers to which backend  
- added diffusers **lmsd** sampler  

## Update for 2023-07-13

Another big one, but now improvements to both **diffusers** and **original** backends as well plus ability to dynamically switch between them!

- swich backend between diffusers and original on-the-fly
  - you can still use `--backend <backend>` and now that only means in which mode app will start,
    but you can change it anytime in ui settings
  - for example, you can even do things like generate image using sd-xl,  
    then switch to original backend and perform inpaint using a different model  
- diffusers backend:
  - separate ui settings for refiner pass with sd-xl  
    you can specify: prompt, negative prompt, steps, denoise start  
  - fix loading from pure safetensors files  
    now you can load sd-xl from safetensors file or from huggingface folder format  
  - fix kandinsky model (2.1 working, 2.2 was just released and will be soon)  
- original backend:
  - improvements to vae/unet handling as well as cross-optimization heads  
    in non-technical terms, this means lower memory usage and higher performance  
    and you should be able to generate higher resolution images without any other changes
- other:
  - major refactoring of the javascript code  
    includes fixes for text selections and navigation  
  - system info tab now reports on nvidia driver version as well  
  - minor fixes in extra-networks  
  - installer handles origin changes for submodules  

big thanks to @huggingface team for great communication, support and fixing all the reported issues asap!


## Update for 2023-07-10

Service release with some fixes and enhancements:

- diffusers:
  - option to move base and/or refiner model to cpu to free up vram  
  - model downloader options to specify model variant / revision / mirror  
  - now you can download `fp16` variant directly for reduced memory footprint  
  - basic **img2img** workflow (*sketch* and *inpaint* are not supported yet)  
    note that **sd-xl** img2img workflows are architecturaly different so it will take longer to implement  
  - updated hints for settings  
- extra networks:
  - fix corrupt display on refesh when new extra network type found  
  - additional ui tweaks  
  - generate thumbnails from previews only if preview resolution is above 1k
- image viewer:
  - fixes for non-chromium browsers and mobile users and add option to download image  
  - option to download image directly from image viewer
- general
  - fix startup issue with incorrect config  
  - installer should always check requirements on upgrades

## Update for 2023-07-08

This is a massive update which has been baking in a `dev` branch for a while now

- merge experimental diffusers support  

*TL;DR*: Yes, you can run **SD-XL** model in **SD.Next** now  
For details, see Wiki page: [Diffusers](https://github.com/vladmandic/automatic/wiki/Diffusers)  
Note this is still experimental, so please follow Wiki  
Additional enhancements and fixes will be provided over the next few days  
*Thanks to @huggingface team for making this possible and our internal @team for all the early testing*

Release also contains number of smaller updates:

- add pan & zoom controls (touch and mouse) to image viewer (lightbox)  
- cache extra networks between tabs  
  this should result in neat 2x speedup on building extra networks  
- add settings -> extra networks -> do not automatically build extra network pages  
  speeds up app start if you have a lot of extra networks and you want to build them manually when needed  
- extra network ui tweaks  

## Update for 2023-07-01

Small quality-of-life updates and bugfixes:

- add option to disallow usage of ckpt checkpoints
- change lora and lyco dir without server restart
- additional filename template fields: `uuid`, `seq`, `image_hash`  
- image toolbar is now shown only when image is present
- image `Zip` button gone and its not optional setting that applies to standard `Save` button
- folder `Show` button is present only when working on localhost,  
  otherwise its replaced with `Copy` that places image URLs on clipboard so they can be used in other apps

## Update for 2023-06-30

A bit bigger update this time, but contained to specific areas...

- change in behavior  
  extensions no longer auto-update on startup  
  using `--upgrade` flag upgrades core app as well as all submodules and extensions  
- **live server log monitoring** in ui  
  configurable via settings -> live preview  
- new **extra networks interface**  
  *note: if youre using a 3rd party ui extension for extra networks, it will likely need to be updated to work with new interface*
  - display in front of main ui, inline with main ui or as a sidebar  
  - lazy load thumbnails  
    drastically reduces load times for large number of extra networks  
  - auto-create thumbnails from preview images in extra networks in a background thread  
    significant load time saving on subsequent restarts  
  - support for info files in addition to description files  
  - support for variable aspect-ratio thumbnails  
  - new folder view  
- **extensions sort** by trending  
- add requirements check for training  

## Update for 2023-06-26

- new training tab interface  
  - redesigned preprocess, train embedding, train hypernetwork  
- new models tab interface  
  - new model convert functionality, thanks @akegarasu  
  - new model verify functionality  
- lot of ipex specific fixes/optimizations, thanks @disty0  

## Update for 2023-06-20

This one is less relevant for standard users, but pretty major if youre running an actual server  
But even if not, it still includes bunch of cumulative fixes since last release - and going by number of new issues, this is probably the most stable release so far...
(next one is not going to be as stable, but it will be fun :) )

- minor improvements to extra networks ui  
- more hints/tooltips integrated into ui  
- new dedicated api server  
  - but highly promising for high throughput server  
- improve server logging and monitoring with  
  - server log file rotation  
  - ring buffer with api endpoint `/sdapi/v1/log`  
  - real-time status and load endpoint `/sdapi/v1/system-info/status`

## Update for 2023-06-14

Second stage of a jumbo merge from upstream plus few minor changes...

- simplify token merging  
- reorganize some settings  
- all updates from upstream: **A1111** v1.3.2 [df004be] *(latest release)*  
  pretty much nothing major that i havent released in previous versions, but its still a long list of tiny changes  
  - skipped/did-not-port:  
    add separate hires prompt: unnecessarily complicated and spread over large number of commits due to many regressions  
    allow external scripts to add cross-optimization methods: dangerous and i dont see a use case for it so far  
    load extension info in threads: unnecessary as other optimizations ive already put place perform equally good  
  - broken/reverted:  
    sub-quadratic optimization changes  

## Update for 2023-06-13

Just a day later and one *bigger update*...
Both some **new functionality** as well as **massive merges** from upstream  

- new cache for models/lora/lyco metadata: `metadata.json`  
  drastically reduces disk access on app startup  
- allow saving/resetting of **ui default values**  
  settings -> ui defaults
- ability to run server without loaded model  
  default is to auto-load model on startup, can be changed in settings -> stable diffusion  
  if disabled, model will be loaded on first request, e.g. when you click generate  
  useful when you want to start server to perform other tasks like upscaling which do not rely on model  
- updated `accelerate` and `xformers`
- huge nubmer of changes ported from **A1111** upstream  
  this was a massive merge, hopefully this does not cause any regressions  
  and still a bit more pending...

## Update for 2023-06-12

- updated ui labels and hints to improve clarity and provide some extra info  
  this is 1st stage of the process, more to come...  
  if you want to join the effort, see <https://github.com/vladmandic/automatic/discussions/1246>
- new localization and hints engine  
  how hints are displayed can be selected in settings -> ui  
- reworked **installer** sequence  
  as some extensions are loading packages directly from their preload sequence  
  which was preventing some optimizations to take effect  
- updated **settings** tab functionality, thanks @gegell  
  with real-time monitor for all new and/or updated settings  
- **launcher** will now warn if application owned files are modified  
  you are free to add any user files, but do not modify app files unless youre sure in what youre doing  
- add more profiling for scripts/extensions so you can see what takes time  
  this applies both to initial load as well as execution  
- experimental `sd_model_dict` setting which allows you to load model dictionary  
  from one model and apply weights from another model specified in `sd_model_checkpoint`  
  results? who am i to judge :)


## Update for 2023-06-05

Few new features and extra handling for broken extensions  
that caused my phone to go crazy with notifications over the weekend...

- added extra networks to **xyz grid** options  
  now you can have more fun with all your embeddings and loras :)  
- new **vae decode** method to help with larger batch sizes, thanks @bigdog  
- new setting -> lora -> **use lycoris to handle all lora types**  
  this is still experimental, but the goal is to obsolete old built-in lora module  
  as it doesnt understand many new loras and built-in lyco module can handle it all  
- somewhat optimize browser page loading  
  still slower than id want, but gradio is pretty bad at this  
- profiling of scripts/extensions callbacks  
  you can now see how much or pre/post processing is done, not just how long generate takes  
- additional exception handling so bad exception does not crash main app  
- additional background removal models  
- some work on bfloat16 which nobody really should be using, but why not 🙂


## Update for 2023-06-02

Some quality-of-life improvements while working on larger stuff in the background...

- redesign action box to be uniform across all themes  
- add **pause** option next to stop/skip  
- redesigned progress bar  
- add new built-in extension: **agent-scheduler**  
  very elegant way to getting full queueing capabilities, thank @artventurdev  
- enable more image formats  
  note: not all are understood by browser so previews and images may appear as blank  
  unless you have some browser extensions that can handle them  
  but they are saved correctly. and cant beat raw quality of 32-bit `tiff` or `psd` :)  
- change in behavior: `xformers` will be uninstalled on startup if they are not active  
  if you do have `xformers` selected as your desired cross-optimization method, then they will be used  
  reason is that a lot of libaries try to blindly import xformers even if they are not selected or not functional  

## Update for 2023-05-30

Another bigger one...And more to come in the next few days...

- new live preview mode: taesd  
  i really like this one, so its enabled as default for new installs  
- settings search feature  
- new sampler: dpm++ 2m sde  
- fully common save/zip/delete (new) options in all tabs  
  which (again) meant rework of process image tab  
- system info tab: live gpu utilization/memory graphs for nvidia gpus  
- updated controlnet interface  
- minor style changes  
- updated lora, swinir, scunet and ldsr code from upstream  
- start of merge from a1111 v1.3  

## Update for 2023-05-26

Some quality-of-life improvements...

- updated [README](https://github.com/vladmandic/automatic/blob/master/README.md)
- created [CHANGELOG](https://github.com/vladmandic/automatic/blob/master/CHANGELOG.md)  
  this will be the source for all info about new things moving forward  
  and cross-posted to [Discussions#99](https://github.com/vladmandic/automatic/discussions/99) as well as discord [announcements](https://discord.com/channels/1101998836328697867/1109953953396957286)
- optimize model loading on startup  
  this should reduce startup time significantly  
- set default cross-optimization method for each platform backend  
  applicable for new installs only  
  - `cuda` => Scaled-Dot-Product
  - `rocm` => Sub-quadratic
  - `directml` => Sub-quadratic
  - `ipex` => invokeais
  - `mps` => Doggettxs
  - `cpu` => Doggettxs
- optimize logging  
- optimize profiling  
  now includes startup profiling as well as `cuda` profiling during generate  
- minor lightbox improvements  
- bugfixes...i dont recall when was a release with at least several of those  

other than that - first stage of [Diffusers](https://github.com/huggingface/diffusers) integration is now in master branch  
i dont recommend anyone to try it (and dont even think reporting issues for it)  
but if anyone wants to contribute, take a look at [project page](https://github.com/users/vladmandic/projects/1/views/1)

## Update for 2023-05-23

Major internal work with perhaps not that much user-facing to show for it ;)

- update core repos: **stability-ai**, **taming-transformers**, **k-diffusion, blip**, **codeformer**  
  note: to avoid disruptions, this is applicable for new installs only
- tested with **torch 2.1**, **cuda 12.1**, **cudnn 8.9**  
  (production remains on torch2.0.1+cuda11.8+cudnn8.8)  
- fully extend support of `--data-dir`  
  allows multiple installations to share pretty much everything, not just models  
  especially useful if you want to run in a stateless container or cloud instance  
- redo api authentication  
  now api authentication will use same user/pwd (if specified) for ui and strictly enforce it using httpbasicauth  
  new authentication is also fully supported in combination with ssl for both sync and async calls  
  if you want to use api programatically, see examples in `cli/sdapi.py`  
- add dark/light theme mode toggle  
- redo some `clip-skip` functionality  
- better matching for vae vs model  
- update to `xyz grid` to allow creation of large number of images without creating grid itself  
- update `gradio` (again)  
- more prompt parser optimizations  
- better error handling when importing image settings which are not compatible with current install  
  for example, when upscaler or sampler originally used is not available  
- fixes...amazing how many issues were introduced by porting a1111 v1.20 code without adding almost no new functionality  
  next one is v1.30 (still in dev) which does bring a lot of new features  

## Update for 2023-05-17

This is a massive one due to huge number of changes,  
but hopefully it will go ok...

- new **prompt parsers**  
  select in UI -> Settings -> Stable Diffusion  
  - **Full**: my new implementation  
  - **A1111**: for backward compatibility  
  - **Compel**: as used in ComfyUI and InvokeAI (a.k.a *Temporal Weighting*)  
  - **Fixed**: for really old backward compatibility  
- monitor **extensions** install/startup and  
  log if they modify any packages/requirements  
  this is a *deep-experimental* python hack, but i think its worth it as extensions modifying requirements  
  is one of most common causes of issues
- added `--safe` command line flag mode which skips loading user extensions  
  please try to use it before opening new issue  
- reintroduce `--api-only` mode to start server without ui  
- port *all* upstream changes from [A1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)  
  up to today - commit hash `89f9faa`  

## Update for 2023-05-15

- major work on **prompt parsing**
  this can cause some differences in results compared to what youre used to, but its all about fixes & improvements
  - prompt parser was adding commas and spaces as separate words and tokens and/or prefixes
  - negative prompt weight using `[word:weight]` was ignored, it was always `0.909`
  - bracket matching was anything but correct. complex nested attention brackets are now working.
  - btw, if you run with `--debug` flag, youll now actually see parsed prompt & schedule
- updated all scripts in `/cli`  
- add option in settings to force different **latent sampler** instead of using primary only
- add **interrupt/skip** capabilities to process images

## Update for 2023-05-13

This is mostly about optimizations...

- improved `torch-directml` support  
  especially interesting for **amd** users on **windows**  where **torch+rocm** is not yet available  
  dont forget to run using `--use-directml` or default is **cpu**  
- improved compatibility with **nvidia** rtx 1xxx/2xxx series gpus  
- fully working `torch.compile` with **torch 2.0.1**  
  using `inductor` compile takes a while on first run, but does result in 5-10% performance increase  
- improved memory handling  
  for highest performance, you can also disable aggressive **gc** in settings  
- improved performance  
  especially *after* generate as image handling has been moved to separate thread  
- allow per-extension updates in extension manager  
- option to reset configuration in settings  

## Update for 2023-05-11

- brand new **extension manager**  
  this is pretty much a complete rewrite, so new issues are possible
- support for `torch` 2.0.1  
  note that if you are experiencing frequent hangs, this may be a worth a try  
- updated `gradio` to 3.29.0
- added `--reinstall` flag to force reinstall of all packages  
- auto-recover & re-attempt when `--upgrade` is requested but fails
- check for duplicate extensions  

## Update for 2023-05-08

Back online with few updates:

- bugfixes. yup, quite a lot of those  
- auto-detect some cpu/gpu capabilities on startup  
  this should reduce need to tweak and tune settings like no-half, no-half-vae, fp16 vs fp32, etc  
- configurable order of top level tabs  
- configurable order of scripts in txt2img and img2img  
  for both, see sections in ui-> settings -> user interface

## Update for 2023-05-04

Again, few days later...

- reviewed/ported **all** commits from **A1111** upstream  
  some a few are not applicable as i already have alternative implementations  
  and very few i choose not to implement (save/restore last-known-good-config is a bad hack)  
  otherwise, were fully up to date (it doesnt show on fork status as code merges were mostly manual due to conflicts)  
  but...due to sheer size of the updates, this may introduce some temporary issues  
- redesigned server restart function  
  now available and working in ui  
  actually, since server restart is now a true restart and not ui restart, it can be used much more flexibly  
- faster model load  
  plus support for slower devices via stream-load function (in ui settings)  
- better logging  
  this includes new `--debug` flag for more verbose logging when troubleshooting  

## Update for 2023-05-01

Been a bit quieter for last few days as changes were quite significant, but finally here we are...

- Updated core libraries: Gradio, Diffusers, Transformers
- Added support for **Intel ARC** GPUs via Intel OneAPI IPEX (auto-detected)
- Added support for **TorchML** (set by default when running on non-compatible GPU or on CPU)
- Enhanced support for AMD GPUs with **ROCm**
- Enhanced support for Apple **M1/M2**
- Redesigned command params: run `webui --help` for details
- Redesigned API and script processing
- Experimental support for multiple **Torch compile** options
- Improved sampler support
- Google Colab: <https://colab.research.google.com/drive/126cDNwHfifCyUpCCQF9IHpEdiXRfHrLN>  
  Maintained by <https://github.com/Linaqruf/sd-notebook-collection>
- Fixes, fixes, fixes...

To take advantage of new out-of-the-box tunings, its recommended to delete your `config.json` so new defaults are applied. its not necessary, but otherwise you may need to play with UI Settings to get the best of Intel ARC, TorchML, ROCm or Apple M1/M2.

## Update for 2023-04-27

a bit shorter list as:

- ive been busy with bugfixing  
  there are a lot of them, not going to list each here.  
  but seems like critical issues backlog is quieting down and soon i can focus on new features development.  
- ive started collaboration with couple of major projects,
  hopefully this will accelerate future development.

whats new:

- ability to view/add/edit model description shown in extra networks cards  
- add option to specify fallback sampler if primary sampler is not compatible with desired operation  
- make clip skip a local parameter  
- remove obsolete items from UI settings  
- set defaults for AMD ROCm  
  if you have issues, you may want to start with a fresh install so configuration can be created from scratch
- set defaults for Apple M1/M2  
  if you have issues, you may want to start with a fresh install so configuration can be created from scratch

## Update for 2023-04-25

- update process image -> info
- add VAE info to metadata
- update GPU utility search paths for better GPU type detection
- update git flags for wider compatibility
- update environment tuning
- update ti training defaults
- update VAE search paths
- add compatibility opts for some old extensions
- validate script args for always-on scripts  
  fixes: deforum with controlnet  

## Update for 2023-04-24

- identify race condition where generate locks up while fetching preview
- add pulldowns to x/y/z script
- add VAE rollback feature in case of NaNs
- use samples format for live preview
- add token merging
- use **Approx NN** for live preview
- create default `styles.csv`
- fix setup not installing `tensorflow` dependencies
- update default git flags to reduce number of warnings

## Update for 2023-04-23

- fix VAE dtype  
  should fix most issues with NaN or black images  
- add built-in Gradio themes  
- reduce requirements  
- more AMD specific work
- initial work on Apple platform support
- additional PR merges
- handle torch cuda crashing in setup
- fix setup race conditions
- fix ui lightbox
- mark tensorflow as optional
- add additional image name templates

## Update for 2023-04-22

- autodetect which system libs should be installed  
  this is a first pass of autoconfig for **nVidia** vs **AMD** environments  
- fix parse cmd line args from extensions  
- only install `xformers` if actually selected as desired cross-attention method
- do not attempt to use `xformers` or `sdp` if running on cpu
- merge tomesd token merging  
- merge 23 PRs pending from a1111 backlog (!!)

*expect shorter updates for the next few days as ill be partially ooo*

## Update for 2023-04-20

- full CUDA tuning section in UI Settings
- improve exif/pnginfo metadata parsing  
  it can now handle 3rd party images or images edited in external software
- optimized setup performance and logging
- improve compatibility with some 3rd party extensions
  for example handle extensions that install packages directly from github urls
- fix initial model download if no models found
- fix vae not found issues
- fix multiple git issues

note: if you previously had command line optimizations such as --no-half, those are now ignored and moved to ui settings

## Update for 2023-04-19

- fix live preview
- fix model merge
- fix handling of user-defined temp folders
- fix submit benchmark
- option to override `torch` and `xformers` installer
- separate benchmark data for system-info extension
- minor css fixes
- created initial merge backlog from pending prs on a1111 repo  
  see #258 for details

## Update for 2023-04-18

- reconnect ui to active session on browser restart  
  this is one of most frequently asked for items, finally figured it out  
  works for text and image generation, but not for process as there is no progress bar reported there to start with  
- force unload `xformers` when not used  
  improves compatibility with AMD/M1 platforms  
- add `styles.csv` to UI settings to allow customizing path  
- add `--skip-git` to cmd flags for power users that want  
  to skip all git checks and operations and perform manual updates
- add `--disable-queue` to cmd flags that disables Gradio queues (experimental)
  this forces it to use HTTP instead of WebSockets and can help on unreliable network connections  
- set scripts & extensions loading priority and allow custom priorities  
  fixes random extension issues:  
  `ScuNet` upscaler disappearing, `Additional Networks` not showing up on XYZ axis, etc.
- improve html loading order
- remove some `asserts` causing runtime errors and replace with user-friendly messages
- update README.md

## Update for 2023-04-17

- **themes** are now dynamic and discovered from list of available gradio themes on huggingface  
  its quite a list of 30+ supported themes so far  
- added option to see **theme preview** without the need to apply it or restart server
- integrated **image info** functionality into **process image** tab and removed separate **image info** tab
- more installer improvements
- fix urls
- updated github integration
- make model download as optional if no models found

## Update for 2023-04-16

- support for ui themes! to to *settings* -> *user interface* -> "ui theme*
  includes 12 predefined themes
- ability to restart server from ui
- updated requirements
- removed `styles.csv` from repo, its now fully under user control
- removed model-keyword extension as overly aggressive
- rewrite of the fastapi middleware handlers
- install bugfixes, hopefully new installer is now ok  \
  i really want to focus on features and not troubleshooting installer

## Update for 2023-04-15

- update default values
- remove `ui-config.json` from repo, its now fully under user control
- updated extensions manager
- updated locon/lycoris plugin
- enable quick launch by default
- add multidiffusion upscaler extensions
- add model keyword extension
- enable strong linting
- fix circular imports
- fix extensions updated
- fix git update issues
- update github templates

## Update for 2023-04-14

- handle duplicate extensions
- redo exception handler
- fix generate forever
- enable cmdflags compatibility
- change default css font
- fix ti previews on initial start
- enhance tracebacks
- pin transformers version to last known good version
- fix extension loader

## Update for 2023-04-12

This has been pending for a while, but finally uploaded some massive changes

- New launcher
  - `webui.bat` and `webui.sh`:  
    Platform specific wrapper scripts that starts `launch.py` in Python virtual environment  
    *Note*: Server can run without virtual environment, but it is recommended to use it  
    This is carry-over from original repo  
    **If youre unsure which launcher to use, this is the one you want**  
  - `launch.py`:  
    Main startup script  
    Can be used directly to start server in manually activated `venv` or to run it without `venv`  
  - `installer.py`:  
    Main installer, used by `launch.py`  
  - `webui.py`:  
    Main server script  
- New logger
- New exception handler
- Built-in performance profiler
- New requirements handling
- Move of most of command line flags into UI Settings
