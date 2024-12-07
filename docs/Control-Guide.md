# Control Guide

![screenshot-control](https://github.com/user-attachments/assets/cdad2722-ae7c-4c9c-94d6-5ea35a4b1356)

## Introduction to Control

SDNext's Control tab is our long awaited effort to bring ControlNet, IP-Adapters, T2I Adapter, ControlNet XS, and ControlNet LLLite to our users.  

After doing that, we decided that we would add everything else under the sun that we could squeeze in there, and place it directly into your hands with greater options and flexibility than ever before, to allow you to **Control** your image and video generation with as little effort, as as much power, as possible.

**Note that this document is a work in progress, it's all quite complex and will take some time to write up a bit more as well as smooth out the rough edges and correct any issues and bugs that pop up, expect frequent updates!**  

This guide will attempt to explain how to use it so that anyone can understand it and put it to work for themselves.  

Be sure to also check out [the Control resource page](https://github.com/vladmandic/automatic/wiki/Control) which has more technical information as well as some general tips and suggestions. Its usage information will be merged into this page soon™️.

We'll start with the... Control Controls!

## Controls

### Input

The Input control is exactly what it sounds like, it controls what input images (or videos) are contributing to your image generation, by default that is just the image in [Control input pane](https://github.com/vladmandic/automatic/wiki/control-guide#control-input), however if you select `Separate init image`, another image pane will appear below, allowing you to use that as well.  

**Note:** When using a Control input image as well as a Init input image, the Init input dominates. Adjusting denoise to >=0.9 is recommended, as that will allow the Control input to balance with the Init input. Higher values will increase the strength of Control input further, giving it dominance.  

![Input control](https://github.com/vladmandic/automatic/assets/108482020/abe4a404-e0ff-42c8-813a-2355b39c4f2e)

`Show Preview` is simple, it controls the visibility of the preview window in the far right of the middle row. You'll want this on if you're doing any kind of masking or manipulations that you would want to preview before generating.

There are 3 different Input types:

- `Control only`: This uses only the Control input below as a source for any ControlNet or IP Adapter type tasks based on any of our various options.  

- `Init image same as control`: This option will additionally treat any image placed into the `Control input` pane as a source for img2img type tasks, an image to modify for example.  

- `Separate init image`: This option creates an additional window next to `Control input` labeled `Init input`, so you can have a separate image for both Control operations and an init source.

`Denoising strength` is the same as if you were doing any img2img operation. The higher the value, the more denoising that will take place, and the greater any source image will be modified.

### Size

This can be a little confusing at first because of the `Before` and `After` subtabs, however it's really quite simple and extremely powerful.  
The Control size menu allows you to manipulate the size of your input images before and after inference takes place.  

![Size control](https://github.com/vladmandic/automatic/assets/108482020/97102d4b-03d4-4aa8-85de-d6a8c2fa928d)

The `Before` subtab does 2 things:

- If you do not select any `Resize method`, it is only controlling the output image size width and height in pixels as it would in any text2img or img2img operation.

- However, if you do select a `Resize method`, Nearest for example, you can upscale *or* downscale the `Control input` image before any other operations take place. This will be the size of any image used in further operations. Second Pass is not entirely functional yet, but will be part of this.

For example, you might have a much larger image, such as 2048x3072, that you want to use with canny or depth map, but you do not want an image that large to manipulate or guide your generation, that would be prohibitive, slower, and possibly cause an OOM.  

This is where `Resize method` comes in, you would simply select a resize method, typically Nearest or Lanczos, and then either set the pixel width or height you want to resize to under Fixed, or switch over to Scale and select a number below 1. A setting of 0.5 would make your input image effectively 1024x1536 pixels, which would be used as input for later operations.  

The `After` subtab controls any upscaling or downscaling that would take place **at the end of your image generation process**, most commonly this would either be latent upscaling, and ESRGAN model such as 4x Ultrasharp, or one of the various chaiNNer models we provide. This is the same as it would be in a standard upscaling via text2img or img2img.

### Mask

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

### Video

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

## Extensions

These are some nice goodies that we have cooked up so that no actual installed extensions are necessary, you may even find that our version works better!

### AnimateDiff

This is the new home of our Vlad-created implementation of the AnimateDiff extension. Now with **FREE** FreeInit!  

***I honestly don't know how to use this, so I'll update this when I do. My apologies! But if you already do, enjoy!***

![AnimateDiff controls](https://github.com/vladmandic/automatic/assets/108482020/233bc583-2449-4b6c-8ee5-9d4a8f1316c3)

### IP-Adapter

This is our IP Adapter implementation, with 10 available models for your image or face cloning needs!

![IP-Adapter controls](https://github.com/vladmandic/automatic/assets/108482020/fa473529-228b-441b-811d-d5c4c8cda6a1)

## Image Panes

You may notice small icons above the image panes that look like pencils, these are Interrogate buttons. The left one is BLIP, and the right one is DeepBooru. Click one of the buttons to interrogate the image in the pane below it. The results will appear in your prompt area.

![Interrogate buttons](https://github.com/vladmandic/automatic/assets/108482020/6c391a1f-0999-44b6-8363-98c53d576c88)

### Control Input

This is the heart of Control, you may put any image or even video here to be processed by our system, that means any and all scripts, extensions, even the various Controlnet variants below, though you can individually add guidance images to each of those. If an image is placed here, the system will assume you are performing an img2img process of some sort. If you upload a video to SDNext via the Control input pane, you will see that you can play the video, both input and resultant output. Batching and folders should work as expected.

Note below there are 2 other buttons, Inpainting and Outpainting, below.

![Control input](https://github.com/vladmandic/automatic/assets/108482020/6f40cbad-226b-492e-9ff3-2f81404a9ea8)

## ControlNet+

At the very bottom of the Control page, we have what you've all been waiting for, full ControlNet! I do mean full too, we have it all!
This includes SD and SD-XL. at last! You won't ever need the ControlNet extension ever again, much less to touch the original LDM backend.  

This will take a bit more work to document example workflows, but there are tooltips, and if you've used ControlNet before, you shouldn't have any problems! However if you do, hop on by our Discord server and we're happy to help.
