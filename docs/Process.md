# Process

This guide covers the process tab in the app. The process tab is where you can upload or otherwise provide your image(s) and apply the functions to them.

**Tabs:**

- [Process Image](#process-image) <br>
- [Process Batch](#process-batch) <br>
- [Process Folder](#process-folder) <br>
- [Interrogate Image](#interrogate-image) <br>
- [Interrogate Batch](#interrogate-batch) <br>
- [Visual Query](#visual-query)

**Functions:**

- [Upscale](#upscale) <br>
- [Video](#video) <br>
- [GFPGAN](#gpfgan) <br>
- [CodeFormer](#codeformer) <br>
- [Remove background](#remove-background)

## Tabs

### Process Image

Upload your image so you can apply the functions explained below.

### Process Batch

Upload your Batch so you can apply the functions explained below.

### Process Folder

Type in the location of your input directory and output directory so you can apply the functions explained below.

### Interrogate Image

Upload your image so you can Interrogate or Analyze your image. <br>
**Interrogate**: Gives description of image <br>
**Analyze**: Gives description of image in percentages

### Interrogate Batch

Upload your Batch so you can Interrogate or Analyze your Batch. <br>
**Interrogate**: Gives description of batch <br>
**Analyze**: Gives description of batch in percentages

### Visual Query

Upload your image so you can process user query on your image.
Question can be left blank in which case its a default for the selected model which generates a description, similar to interrogate (but typically more detailed) or you can type something like this:
```Describe this image``` or ```How many cats are in this image?```

## Functions

### Upscale

Upscale your image with the chosen model.

### Video

Processes a bath or folder of images into a video, gif or png.

### GPFGAN

Uses the GPFGAN model on your image and applies face restore that enhances facial details.

### CodeFormer

Use the CodeFormer model on your image and applies face restore that enhances facial details.

### Remove background

Uses a model to remove the background of your main subject in your image, experiment with the settings for desired result.
