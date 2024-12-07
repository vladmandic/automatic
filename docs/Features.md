# Features

## Control

SDNext's Control tab is our long awaited effort to bring ControlNet, IP-Adapters, T2I Adapter, ControlNet XS, and ControlNet LLLite to our users.  

After doing that, we decided that we would add everything else under the sun that we could squeeze in there, and place it directly into your hands with greater options and flexibility than ever before, to allow you to **Control** your image and video generation with as little effort, as as much power, as possible.

**Note that this document is a work in progress, it's all quite complex and will take some time to write up a bit more as well as smooth out the rough edges and correct any issues and bugs that pop up, expect frequent updates!**  

This guide will attempt to explain how to use it so that anyone can understand it and put it to work for themselves.  

Be sure to also check out [the Control resource page](https://github.com/vladmandic/automatic/wiki/Control) which has more technical information as well as some general tips and suggestions. Its usage information will be merged into this page soon™️.

We'll start with the... Control Controls!

### Controls

#### Input

The Input control is exactly what it sounds like, it controls what input images (or videos) are contributing to your image generation, by default that is just the image in [Control input pane](https://github.com/vladmandic/automatic/wiki/control-guide#control-input), however if you select `Separate init image`, another image pane will appear below, allowing you to use that as well.  

**Note:** When using a Control input image as well as a Init input image, the Init input dominates. Adjusting denoise to >=0.9 is recommended, as that will allow the Control input to balance with the Init input. Higher values will increase the strength of Control input further, giving it dominance.  

![Input control](https://github.com/vladmandic/automatic/assets/108482020/abe4a404-e0ff-42c8-813a-2355b39c4f2e)

`Show Preview` is simple, it controls the visibility of the preview window in the far right of the middle row. You'll want this on if you're doing any kind of masking or manipulations that you would want to preview before generating.

There are 3 different Input types:

- `Control only`: This uses only the Control input below as a source for any ControlNet or IP Adapter type tasks based on any of our various options.  

- `Init image same as control`: This option will additionally treat any image placed into the `Control input` pane as a source for img2img type tasks, an image to modify for example.  

- `Separate init image`: This option creates an additional window next to `Control input` labeled `Init input`, so you can have a separate image for both Control operations and an init source.

`Denoising strength` is the same as if you were doing any img2img operation. The higher the value, the more denoising that will take place, and the greater any source image will be modified.

#### Size

This can be a little confusing at first because of the `Before` and `After` subtabs, however it's really quite simple and extremely powerful.  
The Control size menu allows you to manipulate the size of your input images before and after inference takes place.  

![Size control](https://github.com/vladmandic/automatic/assets/108482020/97102d4b-03d4-4aa8-85de-d6a8c2fa928d)

The `Before` subtab does 2 things:

- If you do not select any `Resize method`, it is only controlling the output image size width and height in pixels as it would in any text2img or img2img operation.

- However, if you do select a `Resize method`, Nearest for example, you can upscale *or* downscale the `Control input` image before any other operations take place. This will be the size of any image used in further operations. Second Pass is not entirely functional yet, but will be part of this.

For example, you might have a much larger image, such as 2048x3072, that you want to use with canny or depth map, but you do not want an image that large to manipulate or guide your generation, that would be prohibitive, slower, and possibly cause an OOM.  

This is where `Resize method` comes in, you would simply select a resize method, typically Nearest or Lanczos, and then either set the pixel width or height you want to resize to under Fixed, or switch over to Scale and select a number below 1. A setting of 0.5 would make your input image effectively 1024x1536 pixels, which would be used as input for later operations.  

The `After` subtab controls any upscaling or downscaling that would take place **at the end of your image generation process**, most commonly this would either be latent upscaling, and ESRGAN model such as 4x Ultrasharp, or one of the various chaiNNer models we provide. This is the same as it would be in a standard upscaling via text2img or img2img.

#### Mask

The Mask controls are where we start getting into the real meat of Control, not only does it allow a plethora of different options to mask, segment, and control the view of your masking with various preview types, but it comes with **22 different colormaps for your viewing pleasure!** (And I think vlad made some of those words up 🤫)  

![Mask controls](https://github.com/vladmandic/automatic/assets/108482020/e56bb906-02e3-4a5f-869d-6d0d274dc6a5)

- `Live update`: With this checked, your masking will update as you make changes to it, if this is off, you will need to hit the `Refresh` button to the right to have your preview pane update, making more changes to it while it is processing may lead to it being desynchronized, just hit the refresh button if it does not look correct.

- `Inpaint masked only`: Inpainting will apply only to areas you have masked if this is checked. You must actually inpaint something, otherwise it's just img2img.

- `Invert mask`: Inverts the masking, things you mark with the brush will be excluded from a full mask of the image.

- `Auto-mask`: There are three options here, Threshold, Edge, and Greyscale. Each provides a different method of auto-masking your images.

- `Auto-segment`: Just like Auto-mask, we have provided an extensive list of [Auto-segmentation models](https://github.com/vladmandic/automatic/wiki/control#segmentation-models), they don't require ControlNet to handle the process, but may take a few seconds to process, depending on your GPU.  

- `Preview`: You can select the preview type here, we have provided 5 modes, Masked, Binary, Greyscale, Color, and Composite, which is the default.

- `Colormap`: You can select the style/color scheme of the preview here. There are 22 fantastic color schemes!

- `Blur`: This blurs the edges of what you have masked, to allow some flexibility. Play with it.

- `Erode`: This slider controls the reduction of your auto-masking or auto-segmentation border.

- `Dilate`: This slider controls the expansion of your auto-masking or auto-segmentation border.

#### Video

The Video controls are quite exciting and fun to play with, with our tools now you can, if you wished, turn any video into an anime version for example, frame by frame. There are three output options, GIF, PNG, and MP4. You must select one of these to have video output.  
With these simple controls, you can tweak your video output with surprising flexibility.  
Some video output methods provide more controls, try them all.

![Video controls](https://github.com/vladmandic/automatic/assets/108482020/a2ad0a15-1284-4bbf-87f7-b109011448c3)

- `Skip input frames`: This setting controls how many frames are processed from input instead of every frame. Setting it to 0 would mean processing every frame, a setting of 1 would process every other frame, a setting of 2 would process every third frame, cutting the number of total frames by 2/3rds, and so on.

- `Video file`: You select the type of output you want here, animated GIF (not JIF!), animated PNG, or MP4 video, all provided via FFMPEG of course.  

- `Duration`: The length in seconds you want your output video to be.  

- `Pad frames`: Determine how many frames to add to the beginning and end of the video. This feature is particularly useful when used with interpolation.

- `Interpolate frames`: The number of frames you want interpolated (via RIFE) between existing frames (filtered by skip input frames) in a video sequence. This smoothens the video output, especially if you're skipping frames to avoid choppy motion or low frame rates.

- `Loop`: This is purely for animated GIF and PNG output, it enables the classic looping that you would expect.

When you're using interpolation, the software also detects scene changes. If the scene changes significantly, it will insert pad frames instead of interpolating between two unrelated frames. This ensures a seamless transition between scenes and maintains the overall quality of the video output.

### Extensions

These are some nice goodies that we have cooked up so that no actual installed extensions are necessary, you may even find that our version works better!

#### AnimateDiff

This is the new home of our Vlad-created implementation of the AnimateDiff extension. Now with **FREE** FreeInit!  

***I honestly don't know how to use this, so I'll update this when I do. My apologies! But if you already do, enjoy!***

![AnimateDiff controls](https://github.com/vladmandic/automatic/assets/108482020/233bc583-2449-4b6c-8ee5-9d4a8f1316c3)

#### IP-Adapter

This is our IP Adapter implementation, with 10 available models for your image or face cloning needs!

![IP-Adapter controls](https://github.com/vladmandic/automatic/assets/108482020/fa473529-228b-441b-811d-d5c4c8cda6a1)

### Image Panes

You may notice small icons above the image panes that look like pencils, these are Interrogate buttons. The left one is BLIP, and the right one is DeepBooru. Click one of the buttons to interrogate the image in the pane below it. The results will appear in your prompt area.

![Interrogate buttons](https://github.com/vladmandic/automatic/assets/108482020/6c391a1f-0999-44b6-8363-98c53d576c88)

#### Control Input

This is the heart of Control, you may put any image or even video here to be processed by our system, that means any and all scripts, extensions, even the various Controlnet variants below, though you can individually add guidance images to each of those. If an image is placed here, the system will assume you are performing an img2img process of some sort. If you upload a video to SDNext via the Control input pane, you will see that you can play the video, both input and resultant output. Batching and folders should work as expected.

Note below there are 2 other buttons, Inpainting and Outpainting, below.

![Control input](https://github.com/vladmandic/automatic/assets/108482020/6f40cbad-226b-492e-9ff3-2f81404a9ea8)

### ControlNet+

At the very bottom of the Control page, we have what you've all been waiting for, full ControlNet! I do mean full too, we have it all!
This includes SD and SD-XL. at last! You won't ever need the ControlNet extension ever again, much less to touch the original LDM backend.  

This will take a bit more work to document example workflows, but there are tooltips, and if you've used ControlNet before, you shouldn't have any problems! However if you do, hop on by our Discord server and we're happy to help.

<br>
<br>
<br>

## Process/Visual query

Visual query subsection of the Process tab contains tools to use Visual Question Answering interrogation of images using Vision Language Models.

Currently supported models:

- Moondream 2
- GiT Textcaps
- GIT VQA
  - Base
  - Large
- Blip
  - Base
  - Large
- ViLT Base
- Pix Textcaps
- MS Florence 2
  - Base
  - Large

<br>
<br>
<br>

# LCM

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

<br>
<br>
<br>

# LoRa

## Introduction

LoRA models are small Stable Diffusion models that apply tiny changes to standard checkpoint models. They are usually 10 to 100 times smaller than checkpoint models. That makes them very attractive to people who have an extensive collection of models.

This is a tutorial for beginners who haven’t used LoRA models before. You will learn what LoRA models are, where to find them, and how to use them in the SD.NEXT WebUI.

You can place your LoRA models in `./*Your SD.NEXT directory*/models/Lora`. (You can change the path of the LoRa directory in your settings in the system paths tab)

You can either access your LoRA models by clicking on Networks and then on Lora and select the lora model you want to add to your prompt or by typing: <br>
`<lora:*lora file name*:*preferred weight*>`

The weight indicates the amount of effect it has on your image generation.

## Notes

- Some LoRa's are for different diffusers pipelines, for example you have SD1.5 LoRa's and you have SDXL LoRa's, if you try to use one of these while using the wrong type diffusers pipeline it will give an error.

<br>
<br>
<br>

# HiDiffusion

## Introduction

Diffusion models are great for high-resolution image synthesis but struggle with object duplication and longer generation times at higher resolutions. HiDiffusion, a solution with two key components: Resolution-Aware U-Net (RAU-Net), which prevents object duplication by adjusting feature map sizes, and Modified Shifted Window Multi-head Self-Attention (MSW-MSA), which reduces computation time. HiDiffusion can be added to existing models to generate images up to 4096×4096 pixels at 1.5-6 times faster speeds. Experiments show that HiDiffusion effectively tackles these issues and sets new standards for high-resolution image synthesis.

### Reference

You can read more about HiDiffusion in the link below:

- <https://hidiffusion.github.io/>

### Benefits of using HiDiffusion

- Increases the resolution and speed of pretrained diffusion models.
- Supports txt2image, img2img, inpainting and more.

## How to enable

Check the HiDiffusion checkbox in the SD.NEXT webUI in either the Text, Image or Control tab.

<!-- Add image here -->

<br>
<br>
<br>

# Face restore

## Introduction

Face restore will try to detect a face or multiple faces in a generated image, then it will do a seperate pass over the face which makes the face have an higher resolution and more detailed.

## Models

SD.NEXT has 3 different choices for face restoration:

- Codeformer
- GFPGAN
- Face HiRes

### Codeformer

CodeFormer, created by sczhou, is a robust face restoration algorithm designed to work with both old photos and AI-generated faces. The underlying technology of CodeFormer is based on a Transformer-based prediction network, which models global composition and context for code prediction. This allows the model to discover natural faces that closely approximate the target faces, even when the inputs are severely degraded. A controllable feature transformation module is also included, which enables a flexible trade-off between fidelity and quality. More here: [Codeformer](https://shangchenzhou.com/projects/CodeFormer/).

The CodeFormer weight parameter:
<!-- Add image here -->
0 = Maximum effect; 1 = Minimum effect.

### GFPGAN

GFPGAN stands for "Generative Facial Prior Generative Adversarial Network". It is an artificial intelligence model developed for the purpose of real-world face restoration. The model is designed to repair and enhance faces in photos, particularly useful for restoring old or damaged photographs. GFPGAN leverages generative adversarial networks (GANs), specifically utilizing facial priors encapsulated in a pre-trained face GAN like StyleGAN2, to restore realistic and faithful facial details. More here: [GFPGAN](https://github.com/TencentARC/GFPGAN).

<!-- Add image here -->

### Face HiRes

Face Hires is a feature that aims to improve the details of faces in generated images. It draws inspiration from the popular Adetailer extension, but simplifies the workflow to a single checkbox, making it easy to enable or disable.

Here's what Face Hires does:

- Detection: Identifies and locates faces in the image.
- Cropping and Resizing: Crops each detected face and enlarges it to become a full image.
- Enhancement: Applies an image-to-image (img2img) process to enhance the face.
- Restoration: Resizes the enhanced face back to its original size and integrates it back into the original image.

This process addresses the common issue where models fail to perfectly resolve details in images, especially for faces that are not front and center.

### Parameters

Face hires will use secondary prompt for face restore step if its present. if its not, it will use normal primary prompt.

Face hires has number of tunable paramteters in settings in the postprocessing tab.

- Minimum confidence: minimum score that each detected face must meet during detection phase.
- Max faces: maximum number of faces per image it will try to run on.
- Max face overlap: maximum overlap of when multiple faces are detected before it considers them a single object.
- Min face size: minimum face size it should attempt to fix (e.g. do not try to fix very small faces in the background).
- Max face size: maximum face size it should attempt to fix (e.g. why try to fix something if its already front-and-center and takes 90% of image).
- Face padding: when cropping face, add padding around it.
<br>
<!-- Add image here -->
<br>
<br>
<br>

# Second pass

## Introduction

Second pass means that after an image is finished with it's first pass, it has the function to do another pass, Second pass.

## Usage

You can enable the second pass on the refine tab in ther generation settings. <br>

You can customize the second pass to your needs: <br>
<!-- Add image here -->
- **Upscaler**: Here you can choose a upscaler model.
- **Rescale by**: Here you can choose how much the resolution of your image gets multiplied. (You can also choose a custom resolution at Width resize and Height resize).
- **Force HiRes**: This will force it to execute HiRes fix, that means it will not only multiply the resolution of the image, but also add more detail to it, you can also use HiRes fix without forcing it by selecting a latent upscaling method.
- **Secondary sampler**: Here you can choose a different sampler that will be used during HiRes fix, you can also let it use the same one as the first pass by setting it to "Same as primary".
- **HiRes steps**: How much steps you want your HiRes fix to take.
- **Strength**: Strength is basically the amount that HiRes Fix or Refiner is allowed to change.
- **Refiner start**: Refiner pass will start when base model is this much complete. (Set bigger than 0 and smaller than 1 to run after full base model run).
- **Refiner steps**: How much steps you want your Refiner pass to take.
- **Secondary prompts**: You can also add an secondary positive and negative prompt to your HiRes fix, so you can make a base with the first pass and add the details in the second pass.

### Refiner

Refiner will only start if you have selected a refiner model in "Execution & Models" tab in the settings:
<!-- Add image here -->

<br>
<br>
<br>

# Styles

## Introduction

Styles are prompt presets you can enable, it saves both headaches and time, because it adds a specific style to your prompt without you having to type it yourself.

## Usage

You can select a style under the generation controls:
<!-- Add image here -->

## Adding your own styles

You can add your own styles in `.\*Your SD.NEXT directory*\models\styles`

<br>
<br>
<br>

# Clip skip

## Introduction

Clip Skip plays a significant role in stable diffusion models. It is a technique used to improve the performance of image and video compression in stable diffusion algorithms. By allowing the skipping of certain pixels or blocks, Clip Skip reduces the amount of data that needs to be processed, resulting in faster and more efficient compression. This technique also helps to reduce artifacts and enhance the overall quality of compressed images and videos.

## Usage

You can enable clip skip in the advanced tab in the image generation settings. The default is 1, but also a very popular value is 2 and lots of models are compatible with it. <br>
<!-- Add image here -->

<br>
<br>
<br>

# Embedding

## Introduction

Embedding, also called textual inversion, is an alternative way to control the style of your images in Stable Diffusion. Embedding is the result of textual inversion, a method to define new keywords in a model without modifying it. The method has gained attention because its capable of injecting new styles or objects to a model with as few as 3 -5 sample images.

## Usage

You can either access the embedding in the networks menu or you can type it yourself by writing the embedding name without the file extensions.<br>
(You can also edit the weight of the embedding by writing `(*Embedding name*:*desired weight*)`). <br>
<!-- Add Image Here -->

## Add embeddings

You can add your embeddings in `.\*Your SD.NEXT directory*\models\embeddings`

<br>
<br>
<br>

# Upscaling

## Introduction

Upscaling is when an image is upscaled using stable diffusion, the algorithm analyzes the image's pixel values to determine the diffusion rate. The rate calculated is then used to expand the pixels in higher resolution, resulting in a sharper and clearer image without compromising its quality.

## Usage

In SD.NEXT there are 2 ways to use upscaling, you can either enable it in the second pass(**this will not upscale the end results from the second pass, it will only upscale the image or latent from the first pass**) or you can use it in the process tab under the upscaling menu in the image generation settings.
<!-- Add image here -->
<br>
<!-- Add image here -->
<br>

## Custom upscale models

You can also add your own upscale models to SD.NEXT in the directories below:

- ESRGAN
- LDSR
- SCUNet
- SwinIR
- RealESRGAN
- chaiNNer
<br>

<br>
You can find these directories in `.\*Your SD.NEXT directory*\models`

## Most used

General:

- **Ultrasharp 4x**
- **RealESRGAN 4x+**

Anime:

- **Animesharp 4x**
- **RealESRGAN 4x+ Anime6B**

<br>
<br>
<br>

# Samplers

## Introduction

To produce an image, Stable Diffusion first generates a completely random image in the latent space. The noise predictor then estimates the noise of the image. The predicted noise is subtracted from the image. This process is repeated a dozen times. In the end, you get a clean image.

This denoising process is called sampling because Stable Diffusion generates a new sample image in each step. The method used in sampling is called the sampler or sampling method.

For more information you can click on the links below:

- [Complete Samplers Guide](https://www.felixsanz.dev/articles/complete-guide-to-samplers-in-stable-diffusion)
- [Stable Diffusion Sampler Art](https://stable-diffusion-art.com/samplers/)

Both guides explain in detail several different samplers and their capabilities along with their advantages and disadvantages and the appropiate amount of steps you should use with each sampler.

Below is a screenshot of all the samplers SD.Next provides:
<!-- Add image here -->

<br>
<br>
<br>

# PAG

## Introduction

PAG or Pertubed Attention Guidance is like a modern/better version of CFG scale, although less universal as it cannot be applied in all circumstances. If it's applied, it will be added to image metadata and you'll see it in the log as `StableDiffusionPAGPipeline`. <br>

You can find more information about PAG by clicking on [this](https://github.com/KU-CVLAB/Perturbed-Attention-Guidance).

## Usage

You can enable PAG by setting the attention guidance slider above 0. The attention guidance slider is located in the advanced tab of the image generation settings.  <br>
<!-- Add image here -->

<br>
<br>
<br>

# Interrogate

## Introduction

Interrogation, or captioning helps us refine the prompts we use, enabling us to see how the AI system tags and classifies, and what terms it uses. By looking at these we can further refine our images to attain the concept we have in mind, or remove them via negative prompts.

## Usage

You can find the interrogate option on the process tab under "Interrogate image" or "Interrogate batch" if you want to interrogate a batch of images, then select the CLIP Model you want to use to interrogate the image(s).
<br>
<!-- Add image here -->

<br>
<br>
<br>

# VAE

## Introduction

VAE or Variational Auto Encoder encodes an image into a latent space, which is a lower-dimensional representation of the image. The latent space is then decoded into a new image, which is typically of higher quality than the original image.
<br>
<br>
There are two main types of VAEs that can be used with Stable Diffusion: exponential moving average (EMA) and mean squared error (MSE).  EMA is generally considered to be the better VAE for most applications, as it produces images that are sharper and more realistic. MSE can be used to produce images that are smoother and less noisy, but it may not be as realistic as images generated by EMA.

## Usage

You can change the VAE in the settings in "Execution & Models". <br>
<br>
You can add the VAE's in `.\*Your SD.NEXT directory*\models\VAE`.

<br>
<br>
<br>

# CFG scale

## Introduction

the CFG scale (classifier-free guidance scale) or guidance scale is a parameter that controls how much the image generation process follows the text prompt. The higher the value, the more the image sticks to a given text input.
<br>
<br>
But this does not mean that the value should always be set to maximum, as more guidance means less diversity and quality.

## Usage

Simply use the CFG scale slider in the image generation settings. <br>
<!-- Add image here -->

<br>
<br>
<br>

# Live Preview

## Introduction

The live preview feature allows you to see the image before it is fully generated, so in other words you can see the progress of the image while it is generating.

## Usage

You can modify the live preview settings to your liking in settings > Live Previews. <br>
<!-- Add image here -->
<br>

**Live preview display period**: The amount of steps it has to take before requesting a preview image.
**Live preview method**: The method you want SD.NEXT to use to display preview images.

### Methods

#### Simple

Very cheap approximation. Very fast compared to VAE, but produces pictures with 8 times smaller horizontal/vertical resolution and extremely low quality.

#### Approximate

Cheap neural network approximation. Very fast compared to VAE, but produces pictures with 4 times smaller horizontal/vertical resolution and lower quality.

#### TAESD

TAESD is very tiny autoencoder which uses the same "latent API" as Stable Diffusion's VAE*. TAESD can decode Stable Diffusion's latents into full-size images at (nearly) zero cost.

#### Full VAE

Uses the entire VAE to decode the full resolution image as preview during the generation, this is by far the slowest of the other 3 options.

<br>
<br>
<br>
