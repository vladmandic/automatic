# Stable-Casdade

Original repo: <https://github.com/Stability-AI/StableCascade>

## Use

1. Set your compute precision in Settings -> Compute -> Precision  
   to either **BF16** (if supported) or **FP32** (if not supported)  
    *Note*: **FP16** is not supported for this model  
2. Enable model offloading in Settings -> Diffusers -> Model CPU offload  
   without this, stable cascade will use >16GB of VRAM  
3. Recommended: Set sampler to *Default*  
4. Select model from *Networks -> Models -> Reference*  
   you can select either Full or Lite variation of the model
   and it will automatically be downloaded on first use and loaded into SD.Next  
   attempting to load a manually downloaded safetensors files is not supported as model requires special handling  
   SD.Next automatically chooses BF16 variation when downloading from networs -> reference  
   since its smaller and can be used with either BF16 or FP32 compute precision

### UNet models:

1. Put the UNet safetensors in `models/UNet` folder and put the text encoder (if you use one) in there too. Text encoder name should be the UNet Name + _text_encoder  
2. Load the Stable Cascade base (or a custom decoder) from Huggingface as the main model first, then load the UNet (prior) model as the UNet model from settings.  

Example UNet name: `sc_unet.safetensors`  
Example Text Encoder name: `sc_unet_text_encoder.safetensors`  

### Params

- **Prompt** & **Negative prompt**: as usual
- **Width** & **Height**: as usual
- **CFG scale**: used to condition the prior model, reference value is ~4
- **Secondary CFG scale**: used to condition decoder model, reference value is ~1
- **Steps**: used to control number of steps of the prior model
- **Refiner steps**: used to control number of steps of the decoder model
- **Sampler**: recommended to set to Default *before* loading a model  
  Stable Cascade has its own sampler and results with standard samplers will look suboptimal  
  Built-in sampler is *DDIM/DDPM* based, so if you want to experiment at least use similar sampler  

### Notes

- If model download fails, simply retry it, it will continue from where it left off
- Model consists out of 3 stages split into 2 pipelines which are exected as C -> B -> A:
- Full variation requires ~10GB VRAM and runs at ~3 it/s on RTX4090 at 1024px
- Lite variation requires ~4GB VRAM and runs at ~6 it/s on RTX4090 at 1024px

*Note*: performance numbers are for combined pipeline, both decoder and prior

## Variations

### Overview

Stable cascade is a 3-stage model split into two pipelines (so-called *prior* and *decoder*) and comes into two main variations: **Full** and **Lite**  
You can select which one to use from Networks -> Models -> Reference  

Additionally, each variation comes in 3 different precisions: **FP32**, **BF16**, and **FP16**  
*Note*: **FP16** is an unofficial version by [@KohakuBlueleaf](https://huggingface.co/KBlueLeaf/Stable-Cascade-FP16-fixed) of the model fixed to work with FP16 and may result in slightly different output  


Which precision is going to get loaded depends on:  
- your user preference in Settings -> Compute -> Precision  
- and GPU compatibility as not all GPUs support all precision types  

### Sizes

Stage A and auxiliary models sizes are fixed and noted above  
Stage B and Stage C models are dependent on the variation and precision used  

Variation | Precision | Stage B | Stage C
----------|-----------|---------|---------
Full      | FP32      | 6.2GB   | 14GB
Full      | BF16      | 3.1GB   | 7GB
Full      | FP16      | N/A     | 7GB
Lite      | FP32      | 2.8GB   | 4GB
Lite      | BF16      | 1.4GB   | 2GB
Lite      | FP16      | N/A     | N/A
