import gradio as gr
from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionXLPipeline  # pylint: disable=unused-import
from PIL import Image
import numpy as np
from modules import shared, scripts, processing

"""
Automatic Color Inpaint Script for SD.NEXT - SD & SDXL Support

Author: Artheriax
Credits: SD.NEXT team for script template
Version: W.I.P

Contributions: A new script to automatically inpaint colors in images using Stable Diffusion or Stable Diffusion XL.
"""

## Config

# script title
supported_models = ['sd','sdxl']

title = 'Automatic Color Inpaint'

# is script available in txt2img tab
txt2img = False

# is script available in img2img tab
img2img = True

# is pipeline ok to run in pure latent mode without implicit conversions
latent = True

# pipeline args values are defined in ui method below, here we need to define their exact names
params = ['color_to_mask', 'mask_tolerance']


### Script definition

class Script(scripts.Script):
    def title(self):
        return title

    def show(self, is_img2img):
        if shared.native:
            return img2img if is_img2img else txt2img
        return False

    # Define UI for pipeline
    def ui(self, _is_img2img):
        with gr.Row():
            color_picker = gr.ColorPicker(
                label="Color to Mask",
                value="#04F404",  # Default to green screen green
                tooltip="Pick the color you want to mask and inpaint. Click on the color in the image to automatically select it.\n Advised to use images like green screens to get precise results."
            )
            tolerance_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                value=25,
                label="Color Tolerance",
                tooltip="Adjust the tolerance to include similar colors in the mask.\nLower values mask only very similar colors.\nHigher values mask a wider range of similar colors.\n\nTweak both Tolerance and Denoising Strength in the Inpaint tab to achieve desired inpaint results."
            )
        return [color_picker, tolerance_slider]

    # Run pipeline
    def run(self, p: processing.StableDiffusionProcessing, *args):  # pylint: disable=arguments-differ
        if shared.sd_model_type not in supported_models:
            shared.log.warning(f'MoD: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_models}')
            return None
        color_to_mask_hex, mask_tolerance = args

        # Convert hex color to RGB tuple (0-255)
        color_to_mask_rgb = tuple(int(color_to_mask_hex[i:i+2], 16) for i in (1, 3, 5))

        shared.log.debug(f'{title}: Color to Mask={color_to_mask_rgb}, Tolerance={mask_tolerance}')

        # Create Color Mask using vectorized operations
        init_image = p.init_images[0].convert("RGB")
        image_np = np.array(init_image)

        # Calculate Euclidean distance for all pixels at once
        diff = np.linalg.norm(image_np.astype(np.int16) - np.array(color_to_mask_rgb, dtype=np.int16), axis=2)
        mask_np = (diff <= mask_tolerance).astype(np.uint8) * 255

        mask_image = Image.fromarray(mask_np).convert("L")

        # If an inpaint mask is already provided from the UI, combine it with the color mask
        if p.image_mask:
            combined_mask = Image.composite(
                Image.new("L", mask_image.size, "white"),
                p.image_mask.convert("L"),
                mask_image
            )
            p.image_mask = combined_mask
        else:
            p.image_mask = mask_image

        # override inpaint parameters
        p.inpaint_full_res = True
        p.task_args['mask_blur'] = 0

        # Process the image using SD.Next’s inpainting
        processed: processing.Processed = processing.process_images(p)
        return processed