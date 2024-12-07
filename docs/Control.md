# Control Overview

![screenshot-processors](https://github.com/user-attachments/assets/7bccb82b-366e-4bdb-ae57-cc53fac95d3c)

Native control module for SD.Next for Diffusers backend  
Can be used for Control generation as well as Image and Text workflows  

For a guide on the options and settings, as well as explanations for the controls themselves, see the [Control Guide](https://github.com/vladmandic/automatic/wiki/Control-Guide) page.

## Supported Control Models  

- [lllyasviel ControlNet](https://github.com/lllyasviel/ControlNet) for **SD 1.5** and **SD-XL** models  
  Includes ControlNets as well as Reference-only mode and any compatible 3rd party models  
  Original ControlNets for SD15 are 1.4GB each and for SDXL its at massive 4.9GB  
- [VisLearn ControlNet XS](https://vislearn.github.io/ControlNet-XS/) for **SD-XL** models  
  Lightweight ControlNet models for SDXL at 165MB only with near-identical results  
- [TencentARC T2I-Adapter](https://github.com/TencentARC/T2I-Adapter) for **SD 1.5** and **SD-XL** models  
  T2I-Adapters provide similar functionality at much lower resource cost at only 300MB each  
- [Kohya Control LLite](https://huggingface.co/kohya-ss/controlnet-lllite) for **SD-XL** models  
  LLLite models for SDXL at 46MB only provide lightweight image control  
- [TenecentAILab IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) for **SD 1.5** and **SD-XL** models  
  IP-Adapters provides great style transfer functionality at much lower resource cost at below 100MB for SD15 and 700MB for SDXL  
  IP-Adapters can be combined with ControlNet for more stable results, especially when doing batch/video processing  
- [CiaraRowles TemporalNet](https://huggingface.co/CiaraRowles/TemporalNet) for **SD 1.5** models  
  ControlNet model designed to enhance temporal consistency and reduce flickering for batch/video processing  

All built-in models are downloaded upon first use and stored stored in:  
  `/models/controlnet`, `/models/adapter`, `/models/xs`, `/models/lite`, `/models/processor`

Listed below are all models that are supported out-of-the-box:

### ControlNet  

- **SD15**:  
  Canny, Depth, IP2P, LineArt, LineArt Anime, MLDS, NormalBae, OpenPose,  
  Scribble, Segment, Shuffle, SoftEdge, TemporalNet, HED, Tile  
- **SDXL**:  
  Canny Small XL, Canny Mid XL, Canny XL, Depth Zoe XL, Depth Mid XL

Note: only models compatible with currently loaded base model are listed  
Additional ControlNet models in safetensors can be downloaded manually and placed into corresponding folder: `/models/control/controlnet`  

## ControlNet XS

- **SDXL**:  
  Canny, Depth  

## ControlNet LLLite

- **SDXL**:  
  Canny, Canny anime, Depth anime, Blur anime, Pose anime, Replicate anime

Note: control-lllite is implemented using unofficial implementation and its considered experimental  
Additional ControlNet models in safetensors can be downloaded manually and placed into corresponding folder: `/models/control/lite`  

### T2I-Adapter

    'Segment': 'TencentARC/t2iadapter_seg_sd14v1',
    'Zoe Depth': 'TencentARC/t2iadapter_zoedepth_sd15v1',
    'OpenPose': 'TencentARC/t2iadapter_openpose_sd14v1',
    'KeyPose': 'TencentARC/t2iadapter_keypose_sd14v1',
    'Color': 'TencentARC/t2iadapter_color_sd14v1',
    'Depth v1': 'TencentARC/t2iadapter_depth_sd14v1',
    'Depth v2': 'TencentARC/t2iadapter_depth_sd15v2',
    'Canny v1': 'TencentARC/t2iadapter_canny_sd14v1',
    'Canny v2': 'TencentARC/t2iadapter_canny_sd15v2',
    'Sketch v1': 'TencentARC/t2iadapter_sketch_sd14v1',
    'Sketch v2': 'TencentARC/t2iadapter_sketch_sd15v2',

- **SD15**:  
  Segment, Zoe Depth, OpenPose, KeyPose, Color, Depth v1, Depth v2, Canny v1, Canny v2, Sketch v1, Sketch v2  
- **SDXL**:  
  Canny XL, Depth Zoe XL, Depth Midas XL, LineArt XL, OpenPose XL, Sketch XL  

*Note*: Only models compatible with currently loaded base model are listed

### Processors

- **Pose style**: OpenPose, DWPose, MediaPipe Face
- **Outline style**: Canny, Edge, LineArt Realistic, LineArt Anime, HED, PidiNet
- **Depth style**: Midas Depth Hybrid, Zoe Depth, Leres Depth, Normal Bae
- **Segmentation style**: SegmentAnything
- **Other**: MLSD, Shuffle

*Note*: Processor sizes can vary from none for built-in ones to anywhere between 200MB up to 4.2GB for ZoeDepth-Large

### Segmentation Models

There are 8 Auto-segmentation models available:  
  
- Facebook SAM ViT Base (357MB)  
- Facebook SAM ViT Large (1.16GB)
- Facebook SAM ViT Huge (2.56GB)
- SlimSAM Uniform (106MB)
- SlimSAM Uniform Tiny (37MB)
- Rembg Silueta
- Rembg U2Net  
- Rembg ISNet

### Reference

Reference mode is its own pipeline, so it cannot have multiple units or processors  

## Workflows

### Inputs & Outputs

- Image -> Image
- Batch: list of images -> Gallery and/or Video
- Folder: folder with images -> Gallery and/or Video
- Video -> Gallery and/or Video

*Notes*:
- Input/Output/Preview panels can be minimized by clicking on them  
- For video output, make sure to set video options  

### Unit

- Unit is: **input** plus **process** plus **control**
- Pipeline consists of any number of configured units  
  If unit is using using control modules, all control modules inside pipeline must be of same type  
  e.g. **ControlNet**, **ControlNet-XS**, **T2I-Adapter** or **Reference**
- Each unit can use primary input or its own override input  
- Each unit can have no processor in which case it will run control on input directly  
  Use when you're using predefined input templates  
- Unit can have no control in which case it will run processor only  
- Any combination of input, processor and control is possible  
  For example, two enabled units with process only will produce compound processed image but without control  

### What-if?

- If no input is provided then pipeline will run in **txt2img** mode  
  Can be freely used instead of standard `txt2img`  
- If none of units have control or adapter, pipeline will run in **img2img** mode using input image  
  Can be freely used instead of standard `img2img`  
- If you have processor enabled, but no controlnet or adapter loaded,  
  pipeline will run in **img2img** mode using processed input
- If you have multiple processors enabled, but no controlnet or adapter loaded,  
  pipeline will run in **img2img** mode on *blended* processed image  
- Output resolution is by default set to input resolution,  
  Use resize settings to force any resolution  
- Resize operation can run before (on input image) or after processing (on output image)  
- Using video input will run pipeline on each frame unless **skip frames** is set  
  Video output is standard list of images (gallery) and can be optionally encoded into a video file  
  Video file can be interpolated using **RIFE** for smoother playback  

### Overrides

- Control can be based on main input or each individual unit can have its own override input
- By default, control runs in default control+txt2img mode
- If init image is provided, it runs in control+img2img mode  
  Init image can be same as control image or separate
- IP adapter can be applied to any workflow
- IP adapter can use same input as control input or separate

### Inpaint

- Inpaint workflow is triggered when input image is provided in **inpaint** mode
- Inpaint mode can be used with image-to-image or controlnet workflows
- Other unit types such as T2I, XS or Lite do not support inpaint mode

### Outpaint

- Outpaint workflow is triggered when input image is provided in **outpaint** mode
- Outpaint mode can be used with image-to-image or controlnet workflows
- Other unit types such as T2I, XS or Lite do not support outpaint mode
- Recommendation is to increase denoising strength to at least 0.8 since outpained area is blank and needs to be filled with noise
- Outpaint folloing input image can be controled by overlap setting - higher overlap and more of original image will be part of the outpaint process

## Logging  

To enable extra logging for troubleshooting purposes,  
set environment variables before running **SD.Next**

- Linux:
  > export SD_CONTROL_DEBUG=true  
  > export SD_PROCESS_DEBUG=true  
  > ./webui.sh --debug  

- Windows:
  > set SD_CONTROL_DEBUG=true  
  > set SD_PROCESS_DEBUG=true  
  > webui.bat --debug  

*Note*: Starting with debug info enabled also enables **Test** mode in Control module

## Limitations / TODO

### Known issues

- Using model offload can cause Control models to be on the wrong device at the time of the execution  
  Example error message:
  > Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same  

  Workaround: Disable **model offload** in settings -> diffusers and use **move model** option instead  

- Issues after trying to use DWPose and installation fails: `` error.  
  Example error message:
  > Control processor DWPose: DLL load failed while importing _ext  

  Workaround: Activate venv and run following commands to install dwpose dependencies manually:  
  `pip install --upgrade --no-deps --force-reinstall openmim==0.3.9 mmengine==0.10.4 mmcv==2.1.0 mmpose==1.3.1 mmdet==3.3.0`

## Future

- Pose editor
- Process multiple images in batch in parallel
- ControlLora <https://huggingface.co/stabilityai/control-lora>
- Multi-frame rendering <https://xanthius.itch.io/multi-frame-rendering-for-stablediffusion>
- Deflickering and deghosting
