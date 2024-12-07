# Getting started

This section describes how to use sdnext with assumption of basic knowledge of stable Diffusion. This section will show you how easy it is to generate images with SDNext with a few clicks. As a user only adjust and click the settings highlighted and noted in red ink

On running the `./webui.bat` the main page will load up which looks like the following below:
 <!-- image here -->

The following sections are important and need to be modified by the user for an image to be generated. All red sections selected with red ink are sections the user must adjust and implement.

**Base Model**
This is the base model you are going to use to generate an image with. This is also called a checkpoint. From the civitAi website there are hundreds if not thousands of models you can download and use. Note that models are typically very large typically in the range of ~2Gb to 30Gb of space.
these models are typically stored in the models/stable-diffusion folder of your sdnext directory.

**Positive Prompt**
There are two prompts positive and negative. The positive prompts is where you write and describe the image and picture you wish to generate.

**Negative Prompt**
This is the section where you write descriptions and components you DON'T want in your image. For example if your are drawing an anime girl sitting on a desk, typical negative prompts can and could be making sure you have a detailed face or not having several hand and legs. Note that there are also embeddings you can use such as bad hands which will be written and placed here.

**Width and Height**
are the image size you wish to work with. For SD1.5 image typically you start with 512x512 or 512x768 images. Using images larger then the training size do not help in image generation and result in poor or horrible images generated. Typically the following image resolutions are used and desired based on the following

**Sampling Types:**
The sampler you wish to use that will generate the image itself along with the number of steps the sampler will use. increasing the steps does not improve image quality and only consumes more resources and may potentially degrade image quality as well therefore it is ideal to choose an appropiate number of steps. For example Euler A typically can generate images in 24 steps. Please see the Sampler section for more information on samplers.

**CFG Scale:**
How closely do you wish the AI to follow your prompt. Again an appropiate value should be chosen. If the CFG value is too low the AI is given lots of freedom and conversely if the CFG value is very high you restrict the AI and force it to follow your prompt more closely. Note that having too high a CFG value will result in image generation. Typical values for CFG are between 6-9. Any lower and higher typically does not produce any good quality image at all.

**Seed:**
The seed is the number. An unique number for every image that is generated. Setting it to -1 you will get a random new image everytime. If you wish to upload or use a pre-existing image say from civitAI, use the process tab and import the settings and you will see the seed value will be the number of the existing generated image.
