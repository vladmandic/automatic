# How to use X/Y/Z Grid

## Introduction

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
