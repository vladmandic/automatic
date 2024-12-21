import sys
import time
import numpy as np
from PIL import Image
from modules import shared


def resize_image(resize_mode: int, im: Image.Image, width: int, height: int, upscaler_name: str=None, output_type: str='image', context: str=None):
    upscaler_name = upscaler_name or shared.opts.upscaler_for_img2img

    def latent(im, w, h, upscaler):
        from modules.processing_vae import vae_encode, vae_decode
        import torch
        latents = vae_encode(im, shared.sd_model, full_quality=False) # TODO enable full VAE mode for resize-latent
        latents = torch.nn.functional.interpolate(latents, size=(int(h // 8), int(w // 8)), mode=upscaler["mode"], antialias=upscaler["antialias"])
        im = vae_decode(latents, shared.sd_model, output_type='pil', full_quality=False)[0]
        return im

    def resize(im, w, h):
        w = int(w)
        h = int(h)
        if upscaler_name is None or upscaler_name == "None" or im.mode == 'L':
            return im.resize((w, h), resample=Image.Resampling.LANCZOS) # force for mask
        scale = max(w / im.width, h / im.height)
        if scale > 1.0:
            upscalers = [x for x in shared.sd_upscalers if x.name.lower().replace('-', ' ') == upscaler_name.lower().replace('-', ' ')]
            if len(upscalers) > 0:
                upscaler = upscalers[0]
                im = upscaler.scaler.upscale(im, scale, upscaler.data_path)
            else:
                upscaler = shared.latent_upscale_modes.get(upscaler_name, None)
                if upscaler is not None:
                    im = latent(im, w, h, upscaler)
                else:
                    upscaler = shared.sd_upscalers[0]
                    shared.log.warning(f"Resize upscaler: invalid={upscaler_name} fallback={upscaler.name}")
                    shared.log.debug(f"Resize upscaler: available={[u.name for u in shared.sd_upscalers]}")
        if im.width != w or im.height != h: # probably downsample after upscaler created larger image
            im = im.resize((w, h), resample=Image.Resampling.LANCZOS)
        return im

    def crop(im):
        ratio = width / height
        src_ratio = im.width / im.height
        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width
        resized = resize(im, src_w, src_h)
        res = Image.new(im.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        return res

    def fill(im, color=None):
        color = color or shared.opts.image_background
        """
        ratio = round(width / height, 1)
        src_ratio = round(im.width / im.height, 1)
        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width
        resized = resize(im, src_w, src_h)
        res = Image.new(im.mode, (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if width > 0 and fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)), box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if height > 0 and fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)), box=(fill_width + src_w, 0))
        return res
        """
        ratio = min(width / im.width, height / im.height)
        im = resize(im, int(im.width * ratio), int(im.height * ratio))
        res = Image.new(im.mode, (width, height), color=color)
        res.paste(im, box=((width - im.width)//2, (height - im.height)//2))
        return res

    def context_aware(im, width, height, context):
        import seam_carving # https://github.com/li-plus/seam-carving
        if 'forward' in context.lower():
            energy_mode = "forward"
        elif 'backward' in context.lower():
            energy_mode = "backward"
        else:
            return im
        if 'add' in context.lower():
            src_ratio = min(width / im.width, height / im.height)
            src_w = int(im.width * src_ratio)
            src_h = int(im.height * src_ratio)
            src_image = resize(im, src_w, src_h)
        elif 'remove' in context.lower():
            ratio = width / height
            src_ratio = im.width / im.height
            src_w = width if ratio > src_ratio else im.width * height // im.height
            src_h = height if ratio <= src_ratio else im.height * width // im.width
            src_image = resize(im, src_w, src_h)
        else:
            return im
        res = Image.fromarray(seam_carving.resize(
            src_image, # source image (rgb or gray)
            size=(width, height),  # target size
            energy_mode=energy_mode,  # choose from {backward, forward}
            order="width-first",  # choose from {width-first, height-first}
            keep_mask=None,  # object mask to protect from removal
        ))
        return res

    t0 = time.time()
    if resize_mode is None:
        resize_mode = 0
    if resize_mode == 0 or (im.width == width and im.height == height) or (width == 0 and height == 0): # none
        res = im.copy()
    elif resize_mode == 1: # fixed
        res = resize(im, width, height)
    elif resize_mode == 2: # crop
        res = crop(im)
    elif resize_mode == 3: # fill
        res = fill(im)
    elif resize_mode == 4: # edge
        from modules import masking
        res = fill(im, color=0)
        res, _mask = masking.outpaint(res)
    elif resize_mode == 5: # context-aware
        res = context_aware(im, width, height, context)
    else:
        res = im.copy()
        shared.log.error(f'Invalid resize mode: {resize_mode}')
    t1 = time.time()
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    shared.log.debug(f'Image resize: input={im} width={width} height={height} mode="{shared.resize_modes[resize_mode]}" upscaler="{upscaler_name}" context="{context}" type={output_type} result={res} time={t1-t0:.2f} fn={fn}') # pylint: disable=protected-access
    return np.array(res) if output_type == 'np' else res
