# Black Forest Labs FLUX.1

**FLUX.1** family consists of 3 variations:
- [Pro](https://blackforestlabs.ai/announcing-black-forest-labs/)  
  Model weights are NOT released, model is available only via Black Forest Labs  
- [Dev](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev)  
  Open-weight, guidance-distilled from Pro variation, available for non-commercial applications  
- [Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)  
  Open-weight, timestep-distilled from Dev variation, available under Apache2.0 license  

Additionally [SD.Next](https://github.com/vladmandic/automatic/) includes *pre-quantized* variations of **FLUX.1 Dev** variation: `qint8`, `qint4` and `nf4`  
Pick variant that uses less memory as model in original form has very high requirements  

![screenshot-modernui-f1](https://github.com/user-attachments/assets/b509a280-8d3b-48b5-8525-363bad8c1ed2)

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

!!! tip

    Use reference models
    Use of reference models is recommended over manually downloaded models!  
    Simply select it from *Networks -> Models -> Reference*  
  and model will be auto-downloaded on first use  

!!! info

    Do not attempt to assemble a full model by loading all individual components  
    That may be how some other apps are designed to work, but its not how SD.Next works  
    Always load full model and then replace individual components as needed  

!!! warning

    If you're getting error message during model load: `file=xxx is not a complete model`  
    It means exactly that - you're trying to load a model component instead of full model  

## Components

**FLUX.1** model consists of:
- Unet/Transformer: MMDiT  
- Text encoder 1: [CLIP-ViT/L](https://huggingface.co/openai/clip-vit-large-patch14),
- Text encoder 2: [T5-XXL Version 1.1](https://huggingface.co/google/t5-v1_1-xxl)  
- VAE

When using reference models, all components will be loaded as needed.  
If using manually downloaded model, you need to ensure that all components are correctly configured and available.  
Note that majority of available downloads are not actually all-in-one models and are instead just a part of the full model with individual components.

!!! tip

    For convience, you can add setting that allow quick replacements of model components  
    to your **quicksettings** by adding  
    *Settings -> User Interface -> Quicksettings  list -> sd_model_checkpoint, sd_unet, sd_vae, sd_text_encoder*

![image](https://github.com/user-attachments/assets/37a6b28f-2b80-4981-bf98-29290352733e)

## Fine-tunes

### Diffusers

There are already many **FLUX.1** unofficial variations available  
Any Diffuser-based variation can be downloaded and loaded into SD.Next using Models -> Huggingface -> Download  
For example, interesting variation is a merge of Dev and Schnell variations by sayakpaul: [sayakpaul/FLUX.1-merged](https://huggingface.co/sayakpaul/flux.1-merged)  

### LoRAs

SD.Next includes support for FLUX.1 LoRAs  

Since LoRA keys vary significantly between tools used to train LoRA as well as LoRA types,  
support for additional LoRAs will be added as needed - please report any non-functional LoRAs!

Also note that compatibility of LoRA depends on the quantization type!
If you have issues loading LoRA, try switching your FLUX.1 base model to different quantization type  

### All-in-one

Typical all-in-one safetensors file is over 20GB in size and contains full model with transformer, both text-encoders and VAE  
Since text encoders and VAE are same between all FLUX.1 models, *using all-in-one safetensors is not recommended* due to large duplication of data  

### Unet/Transformer

Unet/Transformer component of FLUX.1 is a typical model fine-tune and is around 11GB in size  

To load a Unet/Transformer safetensors file:  
1. Download `safetensors` or `gguf` file from desired source and place it in `models/UNET` folder  
   example: [FastFlux Unchained](https://civitai.com/models/671478?modelVersionId=751723)  
2. Load FLUX.1 model as usual and then  
3. Replace transformer with one in desired safetensors file using:  
   *Settings -> Execution & Models -> UNet*  

### Text Encoder

SD.Next allows changing optional text encoder on-the-fly  

Go to *Settings -> Models -> Text encoder* and select the desired text encoder  
T5 enhances text rendering and some details, but its otherwise very lightly used and optional  
Loading lighter T5 will greatly decrease model resource usage, but may not be compatible with all offloading modes  

!!! tip

    To use **prompt attention** syntax with FLUX.1, set  
*Settings -> Execution -> Prompt attention to **xhinker***

*Example image with different encoder quantization options*  
![flux-encoder](https://github.com/user-attachments/assets/afdf6ead-a591-48ae-9eef-f2dd778a645f)

### VAE

SD.Next allows changing VAE model used by FLUX.1 on-the-fly  
There are no alternative VAE models released, so this setting is mostly for future use  

!!! tip

    To enable **image previews** during generate, set *Settings -> Live Preview -> Method to **TAESD***  

To further speed up generation, you can disable "full quality" which triggers use of TAESD instead of full VAE to decode final image  

### Scheduler

FLUX.1 at the moment supports only Euler FlowMatch scheduler, additional schedulers will be added in the future  
Due to specifics of flow-matching methods, number of steps also has strong influence on the image composition, not just on the way how its resolved

*Example image at different steps*  
![flux-steps](https://github.com/user-attachments/assets/943b584d-40eb-4cde-abf3-0cbf1b7adb38)

Additionally, sampler can be tuned with *shift* parameter which roughly modifies how long does model spend on composition vs actual diffusion  

*Example image with different sampler shift values*
![flux-shift](https://github.com/user-attachments/assets/2c810977-a69a-4760-bad0-5e80abb3a1d6)

### ControlNet

Support for all **InstantX/Shakker-Labs** models including [Union-Pro](InstantX/FLUX.1-dev-Controlnet-Union)  

FLUX.1 ControlNets are large at over 6GB on top of already very large FLUX.1 model  
as such, you may need to use offloading:sequential which is not as fast, but uses far less memory  

When using union model, you must also select control mode in the control unit  

## Flux Tools

Link to [Flux Tools](https://blackforestlabs.ai/flux-1-tools/) announcement  
- Redux is actually a tool
- Fill is inpaint/outpaint optimized version of Flux-dev  
- Canny/Depth are optimized versions of Flux-dev for their respective tasks: they are *not* ControlNets that work on top of a model  

To use, go to image or control interface and select *Flux Tools* in scripts  
All models are auto-downloaded on first use  
*note*: All models are [gated](https://github.com/vladmandic/automatic/wiki/Gated) and require acceptance of terms and conditions via web page  
*recommended*: Enable on-the-fly [quantization](https://github.com/vladmandic/automatic/wiki/Quantization) or [compression](https://github.com/vladmandic/automatic/wiki/NNCF-Compression) to reduce resource usage  
- [Redux](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev): ~0.1GB  
  works together with existing model and basically uses input image to analyze it and use that instead of prompt  
  *recommended*: low denoise strength levels result in more variety  
- [Fill](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev): ~23.8GB, replaces currently loaded model  
  *note*: can be used in inpaint/outpaint mode only  
- [Canny](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev): ~23.8GB, replaces currently loaded model  
  *recommended*: guidance scale 30  
- [Depth](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev): ~23.8GB, replaces currently loaded model  
  *recommended*: guidance scale 10  

## Notes

### Performance

Performance and memory usage of different **FLUX.1** variations:

| dtype | time (sec) | performance | memory   | offload    | note |
|-------|------------|-------------|----------|------------|------|
| bf16  |            |             | >32 GB   | none       | *1   |
| bf16  | 50.47      | 0.40 it/s   |          | balanced   | *2   |
| bf16  | 94.28      | 0.21 it/s   |  1.89 GB | sequential |      |
| nf4   | 14.69      | 1.36 it/s   | 17.92 GB | none       |      |
| nf4   | 21.02      | 0.95 it/s   |          | balanced   | *2   |
| nf4   |            |             |          | sequential | err  |
| qint8 | 15.42      | 1.30 it/s   | 18.85 GB | none       |      |
| qint8 |            |             |          | balanced   | err  |
| qint8 |            |             |          | sequential | err  |
| qint4 | 18.37      | 1.09 it/s   | 11.38 GB | none       |      |
| qint4 |            |             |          | balanced   | err  |
| qint4 |            |             |          | sequential | err  |

**Notes**:
- *1: Memory usage exceeeds 32GB and is not recommended  
- *2: Balanced offload VRAM usage is not included since it depends on desired threshold  
