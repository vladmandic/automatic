# [Stable Diffusion 3.x](https://stability.ai/news/stable-diffusion-3-medium)

StabilityAI's Stable Diffusion 3 family consists of:
- [Stable Diffusion 3.0 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium-diffusers)
- [Stable Diffusion 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [Stable Diffusion 3.5 Large Turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo)

![screenshot-modernui-sd3](https://github.com/user-attachments/assets/1ed02ecc-23e4-4fda-8ae5-2d7393dc530c)

!!! info

    Allow gated access
    This is a gated model, you need to accept the terms and conditions to use it  
    For more information see [Gated Access Wiki](https://github.com/vladmandic/automatic/wiki/Gated)

!!! info

    Set offloading
    Set appropriate offloading setting before loading the model to avoid out-of-memory errors  
    For more information see [Offloading Wiki](https://github.com/vladmandic/automatic/wiki/Offload)  

!!! info

    Choose quantization
    Check compatibility of different quantizations with your platform and GPU!  
    For more information see [Quantization Wiki](https://github.com/vladmandic/automatic/wiki/Quantization)  

    [!TIP] Use reference models
    Use of reference models is recommended over manually downloaded models!  
    Simply select it from *Networks -> Models -> Reference*  
  and model will be auto-downloaded on first use  

## Components

**SD3.x** model consists of:
- Unet/Transformer: MMDiT  
- Text encoder 1: [CLIP-ViT/L](https://huggingface.co/openai/clip-vit-large-patch14),
- Text encoder 2: [OpenCLIP-ViT/G](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k),
- Text encoder 3: [T5-XXL Version 1.1](https://huggingface.co/google/t5-v1_1-xxl)  
- VAE

When using reference models, all components will be loaded as needed.  
If using manually downloaded model, you need to ensure that all components are correctly configured and available.  
Note that majority of available downloads are not actually all-in-one models and are instead just a part of the full model with individual components.

!!! info

    Do not attempt to assemble a full model by loading all individual components  
    That may be how some other apps are designed to work, but its not how SD.Next works  
    Always load full model and then replace individual components as needed  

!!! warning

    If you're getting error message during model load: `file=xxx is not a complete model`  
    It means exactly that - you're trying to load a model component instead of full model  

!!! tip

    For convience, you can add setting that allow quick replacements of model components to your   
    **quicksettings** by adding *Settings -> User Interface -> Quicksettings  list -> sd_unet, sd_vae, sd_text_encoder*

![image](https://github.com/user-attachments/assets/37a6b28f-2b80-4981-bf98-29290352733e)

## Fine-tunes

### Diffusers

N/A: Currently there are no known diffusers fine-tunes of SD3.0 or SD3.5 models

### LoRAs

SD.Next includes support for SD3 LoRAs  

Since LoRA keys vary significantly between tools used to train LoRA as well as LoRA types,  
support for additional LoRAs will be added as needed - please report any non-functional LoRAs!

Also note that compatibility of LoRA depends on the quantization type!
If you have issues loading LoRA, try switching your FLUX.1 base model to different quantization type  

### All-in-one

Since text encoders and VAE are same between all FLUX.1 models, *using all-in-one safetensors is not recommended* due to large duplication of data  

### Unet/Transformer

Unet/Transformer component is a typical model fine-tune and is around 11GB in size  

To load a Unet/Transformer safetensors file:  
1. Download `safetensors` or `gguf` file from desired source and place it in `models/UNET` folder  
2. Load model as usual and then  
3. Replace transformer with one in desired safetensors file using:  
   *Settings -> Execution & Models -> UNet*  

### Text Encoder

SD.Next allows changing optional text encoder on-the-fly  

Go to *Settings -> Models -> Text encoder* and select the desired text encoder  
T5 enhances text rendering and some details, but its otherwise very lightly used and optional  
Loading lighter T5 will greatly decrease model resource usage, but may not be compatible with all offloading modes  

### VAE

SD.Next allows changing VAE model used by FLUX.1 on-the-fly  
There are no alternative VAE models released, so this setting is mostly for future use  

!!! tip

    To enable **image previews** during generate, set *Settings -> Live Preview -> Method to **TAESD***  

To further speed up generation, you can disable "full quality" which triggers use of TAESD instead of full VAE to decode final image  

### Scheduler

Model only supports only its native FlowMatch scheduler, additional schedulers will be added in the future  
Due to specifics of flow-matching methods, number of steps also has strong influence on the image composition, not just on the way how its resolved
