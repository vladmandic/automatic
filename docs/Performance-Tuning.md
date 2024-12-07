# Performance Tuning

## Introduction

Hi folks, it's your (moderately) friendly neighborhood Aptronym here!

People are always asking me how to get the most **it/s** out of their GPUs, now... this is a complicated subject due to the current GPU landscape, the wide range of GPUs that can do stable diffusion just within each manufacturer, going back years (nvidia 1080s, 1050s, or RX500 series), as well as varying and often limited amounts of VRAM, possibly inside of a laptop.

That's not even counting the complicated selection of inference platforms SDNext has available, everything from plain everyday CUDA to Onnxruntime/Olive, and now ZLUDA, as well as our two built-in backends, Original and Diffusers.

I can't promise you'll be able to use all of these options, I can't promise they won't crash your instance (or be buggy for that matter), that's going to vary wildly from GPU, OS, RAM, VRAM, current chosen inference platform, and backend, and some of these options will not work together, or at all, on some platforms. You will have to test that yourselves, but we are hoping to build a matrix of sorts showing what is available and what works with what, but that's going to take some user testing and feedback.

If you help us by providing feedback (issues w/logs, screenshots, etc.), which we never have enough of, we will do our best to correct what we can and ensure that the experience of using SDNext is as optimized as it can be. Some limitations we won't be able to overcome purely do to things beyond our control, such as conflicts caused by the nature of the inference platform.

Just as an added ray of sunshine to give you hope, there are plans in motion to create a more self-optimizing system that will not only configure itself for optimal performance based on your setup, but also react to what you're doing, what model you have loaded (SD15 v. SDXL is a huge difference), and specified preferences, so that it is always as performant as possible while sacrificing minimal quality loss. We would welcome any assistance with this, recently-laid-off-due-to-AI developers (Curse you Devin!) or any developers, or people with coding experience, at all.

## Backends

Let's start with the simplest thing, your chosen backend. I know there are some Original mode holdouts, that are clinging on to it for various reasons (I was one myself for months), sometimes simply due to being ignorant of the fact that the Diffusers backend can do SD 1.5 models as well as SDXL and the others.

So let me put that to rest once and for all:

**You need to be using Diffusers**

It is not only now the default on installation, it is faster and better with your precious VRAM usage, as well as being the only backend where future enhancements and features will take place. Wean yourself off of Original, as you are missing out on a lot of functionality, scripts, and features. I know you may have some precious extension that only works in Original, but we have a lot built in now that you may not have noticed.

If the extension is so important to have, then pop by on discord and let me know what it is and explain why. We've added features for less, so if it's a good thing to include, or if we can't already do it to some degree, we'll take a look at it and consider it fairly. If you can't present a good use case, your chances are lower (visual aids help too).

All of that being said, Original mode is going to be retired at some point in the near future, there really just aren't any benefits to keeping it around other than more work for our small development team and contributors.

**One more time, USE DIFFUSERS!**

## Compute Settings

*Note: Changing any of the settings on this page will require you to, at the least, unload the model and then reload it (after hitting apply!), as these settings are applied on model load, not in realtime.*

Generally speaking, for most GPUs our user-base has (mostly Nvidia on Windows judging by discord roles, so using CUDA), you are going to want the settings below (BF16 is possible too if using 30xx+).  

Good settings:  

![Best-compute1](https://github.com/vladmandic/automatic/assets/108482020/913a5a52-14a3-499f-ac67-0f19a3cb5b1e)  

Bad settings:  

![Bad-compute1](https://github.com/vladmandic/automatic/assets/108482020/508d8da6-adf6-484e-a503-f640fa251e29)  


In general, using any of these selected "bad" settings is considered a bad thing, you only want to use any of these if necessary to make your card work with SD. They're slower and use up a lot more memory. *Unless you're on OpenVINO, things show up as fp32 there anyway due to how it works. Leave that alone*

That being said, if you are having strange issues with squares and whatnot and you're on something other than a newer Nvidia GPU (2000s and up?), you may wish to try these settings. Upcast sampling is the better version of --no-half, use it if at all possible.  
Try these one by one, unload the model, reload the model, then test generate, hopefully you will find one, or a combination that works.  

![CopyQ lQSCVs](https://github.com/vladmandic/automatic/assets/108482020/a9ef1145-8f51-4426-93d4-7ba621946417)

### Model Compile

To use any of the model compile options, you must select via the checkboxes, at least one of these: Model, VAE, Text Encoder, Upscaler. It's probably pretty pointless to compile the text encoder, but Model and VAE will net you a large boost in speed. If you use upscalers a lot, by all means, select that too.

#### Stable-fast

Stable-fast is one of the model compiling options, and if you can use it with your setup (Nvidia GPUs, maybe Zluda), you should, as it's a big speed-up. First you will need to open a terminal, activate the venv for sdnext, and from the root sdnext folder (typically *automatic*), you will type `python cli\install-sf.py` and stand back while it hopefully works its magic, acquiring, or potentially compiling then installing the most recent version of stable-fast.  

Note that you can do the install of stable-fast while SDNext is already running in another terminal, it will attempt to load the library when you select it, so there is no need to shut down or restart.

### OneDiff  

Linux (and perhaps Mac) users may also use `OneDiff`, which should be better/faster/stronger than `Stable-Fast`, though you will need to manually execute a `pip install -U onediff` from a venv console to install the necessary libraries.  

NOTE: Do NOT compile the Text Encoder with OneDiff, it makes things slower.  

If you want to thank anyone for OneDiff support, hit up @aifartist on our Discord.  

## Inference options

I'm skipping ahead here a bit since I want to get to the heart of the matter and expand later.

These will be some of the easiest things you can do.

### Token Merging (ToMe)

*Sadly ToMe does not work at the same time as Hypertile. It will be disabled if Hypertile is enabled because Hypertile is faster. If hypertile works for you right now, don't even bother touching this.*

Token merging, aka ToMe, has been around for quite a while and still provides a performance gain if you desire to use it.  
In short it merges tokens, saving memory and speeding up generations.
You can easily use it at 0.3-0.4, performance goes up as the number does, going up higher is up to you but you can always do an xyz and test to see.

**Default settings:**  

![CopyQ blZizQ](https://github.com/vladmandic/automatic/assets/108482020/81c04c4d-9810-45aa-8b24-e3c395a6ff00)  

**Suggested settings:**  

![CopyQ HmmAXj](https://github.com/vladmandic/automatic/assets/108482020/6741f100-50a5-4d75-9465-4bc8df374a65)  


**Honestly, 0.5 and up is the real performance gain, but you test and decide yourself.**  

**Bear in mind that it does have a quality impact on your image, greater the higher the setting, and will make perfect reproduction from the same seed and prompt impossible afaik.**

There is a new implementation along the same lines as ToMe, called ToDo, but works far better. That's in our queue for the near future!

### Hypertile

*Overrides and is incompatible with ToMe, also can cause issues with some platforms, so if you get errors after turning this on, that might be why. Using Hypertile VAE might also cause issues, so try on and off. Hypertile is a much preferable option to `Token Merging` at the moment.*


As long as you enable `Hypertile UNet` you're good to go, the default settings should suffice, as 0 is auto-adjusting the tile size, it adapts to be half the size of your shortest image side.  

Just don't even mess with `Hypertile VAE`, tends to cause more issues than it could possibly be worth, which isn't much.

You may of course screw with the `swap size` and `UNet depth` as you like, I did test them briefly but saw some little benefit.

**Default settings:**  

![CopyQ gwehbW](https://github.com/vladmandic/automatic/assets/108482020/d1015926-3a64-43e5-b1f7-a9a3ed1064fc)


**Suggested settings:**  

![Hypertile-Suggested](https://github.com/vladmandic/automatic/assets/108482020/c2d1ae3e-5cd0-4b4e-8776-e9730f0278a4)


**Also has a quality impact, but I've never measured it personally**


### Other settings

Parallel process images in batch is intended for img2img batch mode.  
If you set batch size=n, typically it generates n images for each input, with this setting, it will generate 1 image for each input, but process n in parallel.  

![CopyQ IARldt](https://github.com/vladmandic/automatic/assets/108482020/dda83a13-a831-4a92-afed-c94a94ae7e0a)
