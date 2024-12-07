# LCM: Latent Consistency Model

LCM (Latent Consistency Model) is a new feature that provides support for SD 1.5 and SD-XL models.

## Installation

Download the LCM LoRA models and place them in your LoRA folder (models/lora or custom):

- For SD 1.5: [lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors)
- For SD-XL: [lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors)

As they have the same name, we recommend doing them one at a time and then renaming it before downloading the next.  

## Usage

1. Make sure to use the **Diffusers** backend in SDNext, **Original** backend will **NOT WORK**
2. Load your preferred SD 1.5 or SD-XL model that you want to use LCM with
3. Load the correct **LCM lora** (**lcm-lora-sdv1-5 or lcm-lora-sdxl**) into your prompt, ex: `<lora:lcm-lora-sdv1-5:1>`
4. Set your **sampler** to **LCM** 
5. Set number of steps to a low number, e.g. **4-6 steps** for SD 1.5, **2-8 steps** for SD-XL
6. Set your **CFG Scale to 1 or 2** (or somewhere between, play with it for best quality)
7. Optionally, turning on **Hypertile and/or FreeU** will greatly increase speed and quality of output images
8. ???
9. Generate!

## Notes

- This also works with latent upscaling, as a second pass/hires fix.
- LCM scheduler does not support steps higher than 50
- The `cli/lcm-convert.py` script can convert any SD 1.5 or SD-XL model to an LCM model by baking in the LoRA and uploading to Huggingface