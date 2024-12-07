# NNCF Model Comporession

## Usage

0. Use Diffusers backend. `Execution & Models` -> `Execution backend`
1. Go into `Compute Settings`  
2. Enable `Compress Model weights with NNCF` options  
3. Reload the model.  

**Note:**
VAE Upcast (in Diffusers settings) has to be set to false if you use the VAE option.  
If you get black images with SDXL models, use the [FP16 Fixed VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors).  

### Features

* Uses INT8, halves the model size  
Saves 3.4 GB of VRAM with SDXL  

### Disadvantages

* It is Autocast, GPU will still use 16 Bit to run the model and will be slower  
* Not implemented in Original backend  
* Fused projections are not compatible with NNCF  
* Using Loras will make generations slower  


## Options

These results compares NNCF 8 bit to 16 bit.  

- Model:  
  Compresses UNet or Transformers part of the model.  
  This is where the most memory savings happens for Stable Diffusion.  

  SDXL: 2500 MB~ memory savings.  
  SD 1.5: 750 MB~ memory savings.  
  PixArt-XL-2: 600 MB~ memory savings.  

- Text Encoder:  
  Compresses Text Encoder parts of the model.  
  This is where the most memory savings happens for PixArt.  

  PixArt-XL-2: 4750 MB~ memory savings.  
  SDXL: 750 MB~ memory savings.  
  SD 1.5: 120 MB~ memory savings.  

- VAE:  
  Compresses VAE part of the model.  
  Memory savings from compressing VAE is pretty small.  

  SD 1.5 / SDXL / PixArt-XL-2: 75 MB~ memory savings.  
