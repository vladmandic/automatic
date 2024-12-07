# StableDiffusion-XL

## Downloading SD-XL

You can simply download these two files from Huggingface and place them into your normal checkpoint directory, though we recommend a subfolder.

* [SD-XL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors)
* [SD-XL Refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/sd_xl_refiner_1.0.safetensors)

## Setup for SD-XL

To facilitate easy use of SD-XL and swapping between refiners, backends, and pipelines, we recommend selecting the  
following items in your Settings Tab, on the User Interface page:

![image](https://github.com/vladmandic/automatic/assets/108482020/59688e46-7056-42b6-a8bd-3a53008c9663)

Once you select them, hit Apply settings, and then Restart server.
When the server returns to being active and your browser page reloads, the Quicksettings  
at the top of your screen should look like this (assuming you were using SDXL):

![image](https://github.com/vladmandic/automatic/assets/108482020/82b78692-47e8-4604-bf66-df0d90409d65)

## VRAM Optimization

There are now 3 methods of memory optimization with the Diffusers backend, and consequently SDXL: Model Shuffle, Medvram, and Lowvram.  
Choose one based on your GPU, VRAM, and how large you want your batches to be.

**Note: `VAE Tiling` can be enabled to save additional VRAM if necessary, but it is recommended to use `VAE Slicing` if you do not have  
abundant VRAM.**  
**`Enable attention slicing`** should generally not be used, as the performance impact is significant.


### Option 1: Model Shuffle

"Model Shuffle" is a memory optimization feature that dynamically moves different parts of the model between the GPU and CPU to  
efficiently utilize VRAM. This is enabled when the following 3 options are Enabled in the Diffusers settings page:

* Move the base model to CPU when using the refiner.
* Move the refiner model to CPU when not in use.
* Move the UNet to CPU during VAE decoding.

To use `Model Shuffling` do not have `--medvram` or `--lowvram` active, then use the following settings:

![image](https://github.com/vladmandic/automatic/assets/108482020/cfe8daed-ec4d-4bb2-bc50-ec72a38e8c66)


The important parts are the 3 Move checkboxes.


Note that if you activate either `CPU model offload` or `Sequential CPU offload`, they will deactivate and ignore Model Shuffling.  
**VRAM Usage**: "Model Shuffle" will work in 8 GB of VRAM.

### Option 2: MEDVRAM

If you have a GPU with 6GB VRAM or require larger batches of SD-XL images without VRAM constraints, you can use the `--medvram` command line argument.  
This option significantly reduces VRAM requirements at the expense of inference speed.  
**Cannot be used with `--lowvram/Sequential CPU offloading`**  
**Note: Until some upstream fixes go in, this will not work with DML or MAC.**

Alternatively, you can enable the `Enable model CPU offload` checkbox in the `Settings` tab on the `Diffusers settings` page:

* Model CPU Offload (same as `--medvram`)
* VAE slicing (recommended)
* Attention slicing is NOT recommended.

![image](https://github.com/vladmandic/automatic/assets/108482020/9b33541c-d4c4-453f-a939-684b480f06a5)


**VRAM Usage**: "Model CPU Offload" can work in 6 GB of VRAM.

**Note**: `--medvram` supersedes the `Model Shuffle` option (e.g., Move base model, refiner model, UNet), and is mutually exclusive  
and cannot be used together with `--lowvram/Sequential CPU offload`


### Option 3: LOWVRAM

If your GPU has as low as 2GB of VRAM, start your SD.Next session with `--lowvram` as a command line argument to **vastly** reduce  
VRAM requirements at the cost of even more inference speed. This is essentially the `Enable Sequential CPU offload` setting.

![image](https://github.com/vladmandic/automatic/assets/108482020/ff56b38e-7aaf-4a04-92db-087aa3ebc63e)

Note: VAE slicing, VAE tiling, and Attention slicing are all enabled by `--lowvram` regardless of the checkboxes.

Using this setting with a GPU that has higher VRAM, your generations will take even longer, but you will be able to do *ridiculously* large  
batches of SD-XL images, up to and including 24 on a 12GB GPU.

**Note: Until some upstream fixes go in, this will not work with SDXL LoRA's and SD 1.5.**


We look forward to seeing how large your batches can get, do let us know on the Discord server, and we **HIGHLY RECOMMEND** that  
you continue down this guide and configure your SD.Next with the Fixed FP16 VAE!

## Fixed FP16 VAE

It is currently recommended to use a **Fixed FP16 VAE** rather than the ones built into the SD-XL base and refiner for  
significant reductions in VRAM (from 6GB of VRAM to <1GB VRAM) and a doubling of VAE processing speed.

Below are the instructions for installation and use:

- [Download Fixed FP16 VAE](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors) to your VAE folder.
- In your `Settings` tab, go to `Diffusers settings` and set `VAE Upcasting` to `False` and hit Apply.  
- Select the your VAE and simply `Reload Checkpoint` to reload the model or  hit `Restart server`.

You should be good to go, Enjoy the huge performance boost!

## Using SD-XL

* To use SD-XL, first SD.Next needs to be in Diffusers mode, not Original, select it from the Backend radio buttons.
* Then select Stable Diffusion XL from the Pipeline dropdown.
* Next select the sd_xl_base_1.0.safetensors file from the Checkpoint dropdown.
* (optional) Finally select the sd_xl_refiner_1.0.safetensors file from the Refiner dropdown.  

## Using SD-XL Refiner

To use **refiner**, it first needs to be loaded and then it can be enabled using `Second pass` option in the UI.
Note that use of **refiner** is not necessary as **base** model can produce very good results on its own.

Refiner can be used in two-modes: as in traditional workflow or with early handover from base to refiner.  
In either case, refiner will use calculated number of steps based on `Refiner steps`.

If `denoise start` is set to `0` or `1`, then traditional workflow is used:

* Base model runs from `0` -> `100%` using Sampling steps.
* Refiner model runs from `0` -> `100%` using Refiner steps.

However, in this mode, refiner may not produce much better result and will likely only smoothen the image as base model already reached 100% and there is insufficient remaining noise for refiner to do anything else.

If `refiner start` is set to any other value, then handover mode is used:

* Base model runs from `0%` -> `denoise_start%`    
  Exact number is calculated internally to be Sampling steps.
* Refiner model runs from `denoise_start%` -> `100%`    
  Exact number is calculated internally to be Refiner steps.

In this mode, using different ratio of steps for primary and refiner is allowed, but may result in unexpected results as base and refiner operations will not be perfectly aligned.

Note on steps vs timesteps. In all workflows (even with original backend and SD 1.5 models), steps do not refer directly do operations internally executed. Steps are used to calculate actual values at which operations will be executed. For example, steps=6 roughly means execute denoising at 0% -> 20%  -> 40%  -> 60%  -> 80%  -> 100%.
For that reason, specifying steps above 99 is meaningless.
