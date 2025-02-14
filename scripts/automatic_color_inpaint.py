import gradio as gr
from PIL import Image
import numpy as np
from modules import shared, scripts, processing

"""
Automatic Color Inpaint Script for SD.NEXT - SD & SDXL Support

Author: Artheriax
Credits: SD.NEXT team for script template
Version: v1

Contributions: A new script to automatically inpaint colors in images using Stable Diffusion, Stable Diffusion XL or Flux.
"""

## Config

# script title
supported_models = ['sd','sdxl', 'flux']

title = 'Automatic Color Inpaint'

# is script available in txt2img tab
txt2img = False

# is script available in img2img tab
img2img = True

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
            gr.HTML("&nbsp ACI: Automatic Color Inpaint<br>")
        with gr.Row():
            color_picker = gr.ColorPicker(
                label="ACI: Color to Mask",
                value="#04F404",  # Default to green screen green
                # info="Pick the color you want to mask and inpaint."
            )
            tolerance_slider = gr.Slider(
                minimum=0,
                maximum=100,
                step=1,
                value=25,
                label="ACI: Color Tolerance",
            )
            denoising_slider = gr.Slider(
                minimum=0.01,
                maximum=1,
                step=0.01,
                value=1,
                label="ACI: Denoising Strength",
            )
        with gr.Row():
            padding_slider = gr.Slider(
                minimum=0,
                maximum=256,
                step=1,
                value=2,
                label="ACI: Mask Padding",
                # info="(Recommended value = 2 to remove leftovers at edges)"
            )
            blur_slider = gr.Slider(
                minimum=0,
                maximum=64,
                step=1,
                value=0,
                label="ACI: Mask Blur",
                # info="(Recommended value = 0 for sharpness)"
            )
        return [color_picker, tolerance_slider, padding_slider, blur_slider, denoising_slider]

    # Run pipeline
    def run(self, p: processing.StableDiffusionProcessing, *args):  # pylint: disable=arguments-differ
        if shared.sd_model_type not in supported_models:
            shared.log.warning(f'MoD: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_models}')
            return None
        color_to_mask_hex, mask_tolerance, mask_padding, mask_blur, inpaint_denoising_strength = args

        # Convert hex color to RGB tuple (0-255)
        color_to_mask_rgb = tuple(int(color_to_mask_hex[i:i+2], 16) for i in (1, 3, 5))

        shared.log.debug(f'ACI: rgb={color_to_mask_rgb} tolerance={mask_tolerance} padding={mask_padding} blur={mask_blur} denoise={inpaint_denoising_strength}')

        # Create Color Mask using vectorized operations
        init_image = p.init_images[0].convert("RGB")
        image_np = np.array(init_image)

        # Calculate Euclidean distance for all pixels at once
        diff = np.linalg.norm(image_np.astype(np.int16) - np.array(color_to_mask_rgb, dtype=np.int16), axis=2)
        calc_tolerance = (diff.max() - diff.min()) * mask_tolerance/100
        mask_np = (diff <= calc_tolerance).astype(np.uint8) * 255

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
        p.inpaint_full_res_padding = mask_padding
        p.mask_blur = mask_blur
        p.denoising_strength = inpaint_denoising_strength

        return None
