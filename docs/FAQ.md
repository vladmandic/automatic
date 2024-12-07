# FAQ: Frequently asked Questions

### Where is the "PNG Info" tab?

*The functionality is integrated into the "Process Image" tabs, removing the need for a separate "PNG Info" tab.*

### Where are the command-line flags like --xformers?

*Most command-line flags have been moved to UI → Settings. For a complete list, start the web UI with the --help flag.*

### How can I get more information about what’s happening with my app?

*Launch the app with the --debug flag and review the setup.log file for detailed insights.*

### How do I add command-line options to webui.bat in Windows?

*1. Right-click on webui.bat and select "Create Shortcut".
2. Right-click the shortcut and open "Properties".
3. Add the options at the end of the "Target" field after a space. Example:
``"C:\path\to\webui.bat" --medvram --autolaunch``
4. Click OK and use this shortcut to start the app.*

### How do I use an AMD GPU on Windows?

*Add the --use-directml command-line flag when starting the app.*

### How can I create large images (e.g., 2048x2048) with limited VRAM?

*Render a smaller image (e.g., 512x512) and upscale it using the "Process Image" tab with an upscaler like SwinIR_4x. This method tiles the image to minimize VRAM usage.*

### Why are my images dull or have incorrect colors?

*Try a different VAE file. Add ,sd_vae to the Settings → User Interface → Quicksettings list to enable quick VAE selection. Place new VAE files in the models\VAE folder.*

### How do I update to the latest version?

*Run webui.bat --upgrade. This pulls the latest updates and applies them to your installation.*

### Something broke. How do I restore my SD.Next installation?

*Use webui.bat --reinstall to reinstall required components or webui.bat --reset to reset to the latest version.*

### My image folder is too large. How do I manage storage?

*In Settings → Image Options, switch from PNG to JPG for smaller file sizes and disable "Always save all generated images." Save only desired images manually.*

### I keep getting out-of-memory errors. What can I do?

*1. Render smaller images (512x512) and upscale later.
2. In Settings → Compute Settings, enable FP16 precision.
3. Use memory-efficient Cross-Attention Optimization methods like Sub-quadratic attention. Disable "SDP disable memory attention" if enabled.*

### Why are my images distorted (e.g., two heads, merged people)?

*Stable Diffusion models are typically trained on 512x512 images. Rendering at different aspect ratios or sizes may cause artifacts. Stick to standard sizes or upscale afterward.*

### What does the "Hires Fix" option do?

*"Hires Fix" upscales images during generation, avoiding post-generation artifacts from traditional upscalers. It complements, but does not replace, external upscalers.*

### What is CLIP Skip?

*CLIP Skip adjusts how detailed image generation becomes. Higher skip values can result in less specific but potentially higher-quality images. Some models benefit from specific settings. PS: CLIP skip is not needed while using SDXL in most cases.*

### What is the best sampler to use?

*There isn’t a definitive best, but "DPM++ 2M Karras" is commonly recommended as a reliable general-purpose sampler.*

### How do I organize checkpoint (ckpt) files?

*Create subfolders in the models directory and move ckpt files there. Restart the UI for changes to take effect. Use meaningful categories like "Photorealistic" or "Anime".*

### How do I enable auto-updates on startup?

*Add the --upgrade flag to your webui.bat launch parameters to automatically update on every startup.*

### Why do I get an error related to typing-extensions when running SD.Next for the first time?

*This error is caused by a recent upstream Python library conflict.*

**Solution:**
Simply re-run webui.bat or webui.sh.
The issue will resolve automatically, and SD.Next will work as expected.

### Why is my clip-interrogator not working or causing errors?

*The old clip-interrogator is incompatible with the newer transformers package required by SD.Next.*

**Solution:**
If you manually installed clip-interrogator as an extension, remove it.
SD.Next now includes an updated version of clip-interrogator for new installations.
To manually update, run:
``git submodule set-url extensions-builtin/clip-interrogator-ext Dahvikiin/clip-interrogator-ext.git
webui.bat --upgrade``
Then, re-launch SD.Next.

### Why do I get an error: 'StableDiffusionXLPipeline' object has no attribute 'decode_first_stage'?

*This issue is related to the Image Previews feature.*

**Solution:**
Ensure the preview method is not set to Full VAE.
If the error persists, disable previews.

### What does this error mean: `module 'diffusers' has no attribute 'StableDiffusionXLPipeline'?

*This happens when an extension has downgraded the required diffusers package.*

**Solution:**
Disable all user-installed extensions and re-launch SD.Next.

### Why do I see errors with xyz_grid.py or 'Options' object has no attribute 'uni_pc_order'?

*These errors occur because some legacy variables are not initialized when using specific --backend arguments.*

**Solution:**
Start SD.Next without any --backend arguments to initialize the variables.
This issue will be addressed in future updates.

### Which HIP SDK Version Should I Use?

*HIP SDK version 5.7.1 is recommended over version 6.1 due to its superior compatibility and more consistent performance in most scenarios.*
