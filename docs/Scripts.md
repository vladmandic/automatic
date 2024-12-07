# Scripts

![image](https://github.com/user-attachments/assets/46d1143d-c02a-40f4-b2b3-0310e39d93d9)

## Quick links

- [X/Y/Z Grid](#xyz-grid)
- [Face script](#face-script)
- [Kohya Hires Fix](#kohya-hires-fix)
- [Layer diffuse](#layerdiffuse)
- [Mixture tiling](#mixture-tiling)
- [MuLan](#mulan)
- [Prompt Matrix](#prompt-matrix)
- [Prompt from file](#prompt-from-file)
- [Regional prompting](#regional-prompting)
- [ResAdapter](#resadapter)
- [T-Gate](#t-gate)
- [Text-to-Video](#text-to-video)
- [DemoFusion](#demofusion)

## X/Y/Z Grid

The X/Y/Z Grid script is a way of generating multiple images with automatic changes in the image, and then displaying the result in labelled grids.

To activate the X/Y/Z Grid, scroll down to the Script dropdown and select "X/Y/Z Grid" within.

Several new UI elements will appear.

X, Y, and Z types are where you can specify what to change in your image.

X type will create columns, Y type will create rows, and Z type will create separate grid images, to emulate a "3D grid"
The X, Y, Z values are where to specify what to change. For some types, there will be a dropdown box to select values, otherwise these values are comma-separated.

Most of these are fairly self explanatory, such as Model, Seed, VAE, Clip skip, and so on.

## Prompt S/R

"Prompt S/R" is Prompt Search and Replace. After selecting this type, the first word in your value should be a word already in your prompt, followed by comma-separated words to change from this word to other words.

For example, if you're generating an image with the prompt "a lazy cat" and you set Prompt S/R to `cat,dog,monkey`, the script will create 3 images of;
`a lazy cat`, `a lazy dog`, and `a lazy monkey`.

You're not restricted to a single word, you could have multiple words; `lazy cat,boisterous dog,mischeavous monkey`, or the entire prompt; `a lazy cat,three blind mice,an astronaut on the moon`.

Embeddings and Loras are also valid Search and Replace terms; `<lora:FirstLora:1>,<lora:SecondLora:1>,<lora:ThirdLora:1>`.

You could also change the strength of a lora; `<lora:FirstLora:1>,<lora:FirstLora:0.75>,<lora:FirstLora:0.5>,<lora:FirstLora:0.25>`.  
(Note: You could strip this down to `FirstLora:1,FirstLora:0.75,FirstLora:0.5,FirstLora:0.25`.)

<br>
<br>
<br>

## Face script

SD.NEXT's face script is used for 4 different face scripts:

- [FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)
- [FaceSwap](https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md)
- [InstantID](https://github.com/InstantID/InstantID)
- [PhotoMaker](https://photo-maker.github.io/)

### FaceID

First select your desired FaceID model and then upload a good picture of the desired face.

**Strength**: How much the script should be applied to the image. <br>
**Structure**: How much similarity there is between the uploaded image and the generated image. <br>

### FaceSwap

You only have to upload a good picture of the desired face.

### InstantID

Add an input image with a good picture of the desired face.
**Strength**: How much the script should be applied to the image. <br>
**Control**: How much similarity there is between the uploaded image and the generated image. <br>

### PhotoMaker

Add an input image with a good picture of the desired face.
**Strength**: How much the script should be applied to the image. <br>
**Start**: When the script should be activated during the image generation process. <br>
<!-- Add image here -->

## Kohya HiRes fix

The [Kohya HiRes fix](https://github.com/wcde/sd-webui-kohya-hiresfix) in SD.NEXT is a great way to generate higher resolution images without getting deformities, it's quite easy to use and not that intensive on your system. It's pretty straight forward but it will take some experimenting to see what's the best settings for your image.

### Usage

You select the kohya hires fix in the scripts and then you can change the settings to your needs. (**Note: it takes a lot of experimentation to see what works for you**).
<br>
<!-- Add image here -->
<br>

- **Scale Factor**: The value that determines the scaling factor applied to the input data during the processing. It controls the magnitude of the changes made to the data.
- **Timestep**: Timestep represents the time step used in the the processing. It determines the granularity of the processing and how the input data is transformed over time.
- **Block**: Block represents the number of blocks used in the processing. It determines the partitioning of the input data into smaller segments for processing.

<br>
<br>
<br>

## LayerDiffuse

LayerDiffuse allows you to create transparent images with Diffusers.
<br>

Example:
<!-- Add image here -->
<br>

### Usage

Simply select [LayerDiffuse](https://github.com/rootonchair/diffuser_layerdiffuse) in the scripts and then click on apply to model after setting everything up. If you want to disable it, simply disable the script. (**Note: You have to reload the model and applying it again after making changes like: adding Lora, Controlnet or IP Adapters**).

<br>
<br>
<br>

## Mixture Tiling

[Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers), an algorithm that builds over existing diffusion models to provide a more detailed control over composition. By harmonizing several diffusion processes acting on different regions of a canvas, it allows generating larger images, where the location of each object and style is controlled by a separate diffusion process.

### Usage

To use it you have to select in the scripts, then you have to write your prompts with newlines between them, so for example: <br>
**bird**<br>
**plane**<br>
**dog**<br>
**cat**<br>
X and Y have to be so if you do X times Y the outcome would be the amount of lines/prompts you have. For the example above you would have to do X=2 and Y=2, X times Y = 4 and you have 4 lines of prompts in the example above.
<br>

**The X and Y overlap**: if you set overlap regions to 0, you're basically getting a combined grid of images. adjust overlap so images can blend and the resulting image actually makes sense.
<br>

**Note: each region is a separate generate process and then they are combined.**

<br>
<br>
<br>

## MuLan

MuLan, a versatile framework to equip any diffusion model with multilingual generation abilities natively by up to 110+ languages around the world.

### Usage

Simply enable [MuLan](https://github.com/mulanai/MuLan) in the scripts and start prompting in your desired language, then click generate.

<br>
<br>
<br>

## Prompt Matrix

Prompt Matrix is useful for testing and comparing the changes prompts are making to your generated images.

### Usage

First enable prompt matrix in your scripts.
<br>
<br>
Then create your prompt like this: <br>
`Woman|Red hair|Blue eyes` <br>
You can make the prompt like this as big as you want. What it will do is it will generate a grid of images, one without the red hair and blue eyes, one with Woman + Red hair, one with Woman + Blue eyes and one with Woman + Red hair + Blue eyes.
Example: <br>
<!-- Add image here -->
<br>

- **Set at prompt start**: This will make it so the example of above will be used like this Red hair|Blue eyes|Woman, so in other words it will use the secondary prompts first before adding Woman to it, for example like this: Red hair, Woman.
- **Random seeds**: It will use a different seed for every image in the grid.
- **Prompt type**: To pick for what prompt you wanna use this script.
- **Joining char**: Comma: Woman, Red hair, space: Woman Red hair.
- **Grid margins**: The space between each image in the grid.

<br>
<br>
<br>

## Prompt from file

Prompt from file allows you to use generation settings from a file including the prompt.

### Usage

First you need to create a .txt file and type something like this: <br>
`--prompt "what ever you want" --negative_prompt "whatever you don't want" --steps 30 --cfg_scale 10 --sampler_name "DPM++ SDE Karras" --seed -1 --width 512 --height 768` <br>
Then upload the file to SD.NEXT at upload prompts. <br>
You can also type it in the prompts box for the same result although it won't be saved if you shutdown SD.NEXT.

## Regional prompting

This pipeline is a port of the [Regional Prompter](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline) extension for Stable Diffusion web UI to diffusers. This code implements a pipeline for the Stable Diffusion model, enabling the division of the canvas into multiple regions, with different prompts applicable to each region. Users can specify regions in two ways: using Cols and Rows modes for grid-like divisions, or the Prompt mode for regions calculated based on prompts.

### Usage

#### Cols and Rows

In the Cols, Rows mode, you can split the screen vertically and horizontally and assign prompts to each region. The split ratio can be specified by 'div', and you can set the division ratio like '3;3;2' or '0.1;0.5'. Furthermore, as will be described later, you can also subdivide the split Cols, Rows to specify more complex regions. <br>
<br>
In this image, the image is divided into three parts, and a separate prompt is applied to each. The prompts are divided by 'BREAK', and each is applied to the respective region. <br>
<!-- Add image here --> <br>
Mode used: `rows` <br>
Prompt used: <br> 
`green hair twintail BREAK` <br>
`red blouse BREAK` <br> 
`blue skirt` <br>
Grid sections: `1,1,1` <br>
<br>
Here is a more advanced example: <br>
<!-- Add image here --> <br>
Mode used: `rows` <br>
Prompt used: <br> 
`blue sky BREAK` <br>
`green hair BREAK` <br>
`book shelf BREAK` <br>
`terrarium on the desk BREAK` <br>
`orange dress and sofa` <br>
Grid sections: `1,2,1,1;2,4,6` <br>

#### Prompt and Prompt-EX

The difference is that in Prompt, duplicate regions are added, whereas in Prompt-EX, duplicate regions are overwritten sequentially. Since they are processed in order, setting a TARGET with a large regions first makes it easier for the effect of small regions to remain unmuffled. <br>
<br>
Prompt-EX example: <br>
<!-- Add image here --> <br>
Mode used: `Prompt-EX` <br>
Prompt used: <br> 
`a girl in street with shirt, tie, skirt BREAK`<br>
`red, shirt BREAK`<br>
`green, tie BREAK`<br>
`blue , skirt`<br>
Prompt thresholds: `0.4,0.6,0.6` <br>
<br>

#### Threshold

The threshold used to determine the mask created by the prompt. This can be set as many times as there are masks, as the range varies widely depending on the target prompt. If multiple regions are used, enter them separated by commas. For example, hair tends to be ambiguous and requires a small value, while face tends to be large and requires a small value. These should be ordered by BREAK.

#### Power

Idicates how much regional prompting is applied to the image generation.

## ResAdapter

ResAdapter, a plug-and-play resolution adapter for enabling any diffusion model generate resolution-free images: no additional training, no additional inference and no style transfer.

### Usage

|               Models              	| Parameters 	|  Resolution Range 	|    Ratio Range   	|
|:---------------------------------:	|:----------:	|:-----------------:	|:----------------:	|
| resadapter_v2_sd1.5               	| 0.9M       	| 128 <= x <= 1024  	| 0.28 <= r <= 3.5 	|
| resadapter_v2_sdxl                	| 0.5M       	| 256 <= x <= 1536  	| 0.28 <= r <= 3.5 	|
| resadapter_v1_sd1.5               	| 0.9M       	| 128 <= x <= 1024  	| 0.5 <= r <= 2    	|
| resadapter_v1_sd1.5_extrapolation 	| 0.9M       	| 512 <= x <= 1024  	| 0.5 <= r <= 2    	|
| resadapter_v1_sd1.5_interpolation 	| 0.9M       	| 128 <= x <= 512   	| 0.5 <= r <= 2    	|
| resadapter_v1_sdxl                	| 0.5M       	| 256 <= x <= 1536  	| 0.5 <= r <= 2    	|
| resadapter_v1_sdxl_extrapolation  	| 0.5M       	| 1024 <= x <= 1536 	| 0.5 <= r <= 2    	|
| resadapter_v1_sdxl_interpolation  	| 0.5M       	| 256 <= x <= 1024  	| 0.5 <= r <= 2    	|

### Weight
How much ResAdapter should be applied to the image generation.

## T-Gate

T-Gate efficiently generates images by caching and reusing attention outputs at scheduled time steps. Experiments show T-Gate’s broad applicability to various existing text-conditional diffusion models which it speeds up by 10-50%.

### Usage

Simply enable T-Gate in the scripts, experiment with the steps a bit to see what works best for your needs.

## Text-to-Video

[Text-to-Video](https://github.com/kabachuha/sd-webui-text2video) is a build in script that makes making animated art very easy, it has multiple models available all of them are personal preference and what works best for your configuration.

### Usage

First choose the script under the scripts, then choose the desired amount of frames, then like you would do normally fill in your positive prompt, negative prompts and etc., then choose the desired output format and click generate.

## DemoFusion

DemoFusion framework seamlessly extends open-source GenAI models, employing Progressive Upscaling, Skip Residual, and Dilated Sampling mechanisms to achieve higher-resolution image generation. The progressive nature of DemoFusion requires more passes, but the intermediate results can serve as "previews", facilitating rapid prompt iteration. You can find more information about DemoFusion [here](https://github.com/PRIS-CV/DemoFusion).

### Usage

<!-- Add image here -->

- **Denoising batch size**: The batch size for multiple denoising paths. Typically, a larger batch size can result in higher efficiency but comes with increased GPU memory requirements.
- **Stride**: The stride of moving local patches. A smaller stride is better for alleviating seam issues, but it also introduces additional computational overhead and inference time.
- **Cosine_scale_1**: Control the decreasing rate of skip-residual. A smaller value results in better consistency with low-resolution results, but it may lead to more pronounced upsampling noise. Please refer to Appendix C in the DemoFusion paper.
- **Cosine_scale_2**: Control the decreasing rate of dilated sampling. A smaller value can better address the repetition issue, but it may lead to grainy images. For specific impacts, please refer to Appendix C in the DemoFusion paper.
- **Cosine_scale_3**: Control the decrease rate of the Gaussian filter. A smaller value results in less grainy images, but it may lead to over-smoothing images. Please refer to Appendix C in the DemoFusion paper.
- **Sigma**: The standard value of the Gaussian filter. A larger sigma promotes the global guidance of dilated sampling, but it has the potential of over-smoothing.
- **Multi_decoder**: Determine whether to use a tiled decoder. Generally, a tiled decoder becomes necessary when the resolution exceeds 3072*3072 on an RTX 3090 GPU.
