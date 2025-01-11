import os
import time
import math
import random
import warnings
from einops import repeat, rearrange
import torch
import numpy as np
import cv2
from PIL import Image
from skimage import exposure
from blendmodes.blend import blendLayers, BlendType
from modules import shared, devices, images, sd_models, sd_samplers, sd_hijack_hypertile, processing_vae, timer


debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_steps = shared.log.trace if os.environ.get('SD_STEPS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_steps('Trace: STEPS')


def is_txt2img():
    return sd_models.get_diffusers_task(shared.sd_model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE


def is_refiner_enabled(p):
    return p.enable_hr and p.refiner_steps > 0 and p.refiner_start > 0 and p.refiner_start < 1 and shared.sd_refiner is not None


def setup_color_correction(image):
    debug("Calibrating color correction")
    correction_target = cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    shared.log.debug(f"Applying color correction: correction={correction.shape} image={original_image}")
    np_image = np.asarray(original_image)
    np_recolor = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
    np_match = exposure.match_histograms(np_recolor, correction, channel_axis=2)
    np_output = cv2.cvtColor(np_match, cv2.COLOR_LAB2RGB)
    image = Image.fromarray(np_output.astype("uint8"))
    image = blendLayers(image, original_image, BlendType.LUMINOSITY)
    return image


def apply_overlay(image: Image, paste_loc, index, overlays):
    if overlays is None or index >= len(overlays):
        return image
    debug(f'Apply overlay: image={image} loc={paste_loc} index={index} overlays={overlays}')
    overlay = overlays[index]
    if not isinstance(image, Image.Image) or not isinstance(overlay, Image.Image):
        return image
    try:
        if paste_loc is not None and (isinstance(paste_loc, tuple) or isinstance(paste_loc, list)):
            x, y, w, h = paste_loc
            if x is None or y is None or w is None or h is None:
                return image
            if image.width != w or image.height != h or x != 0 or y != 0:
                base_image = Image.new('RGBA', (overlay.width, overlay.height))
                image = images.resize_image(2, image, w, h)
                base_image.paste(image, (x, y))
                image = base_image
        image = image.convert('RGBA')
        image.alpha_composite(overlay)
        image = image.convert('RGB')
    except Exception as e:
        shared.log.error(f'Apply overlay: {e}')
    return image


def create_binary_mask(image):
    if image.mode == 'RGBA' and image.getextrema()[-1] != (255, 255):
        image = image.split()[-1].convert("L").point(lambda x: 255 if x > 128 else 0)
    else:
        image = image.convert('L')
    return image


def images_tensor_to_samples(image, approximation=None, model=None): # pylint: disable=unused-argument
    if model is None:
        model = shared.sd_model
    model.first_stage_model.to(devices.dtype_vae)
    image = image.to(shared.device, dtype=devices.dtype_vae)
    image = image * 2 - 1
    if len(image) > 1:
        x_latent = torch.stack([
            model.get_first_stage_encoding(model.encode_first_stage(torch.unsqueeze(img, 0)))[0]
            for img in image
        ])
    else:
        x_latent = model.get_first_stage_encoding(model.encode_first_stage(image))
    return x_latent


def get_sampler_name(sampler_index: int, img: bool = False) -> str:
    sampler_index = sampler_index or 0
    if len(sd_samplers.samplers) > sampler_index:
        sampler_name = sd_samplers.samplers[sampler_index].name
    else:
        sampler_name = "UniPC"
        shared.log.warning(f'Sampler not found: index={sampler_index} available={[s.name for s in sd_samplers.samplers]} fallback={sampler_name}')
    if img and sampler_name == "PLMS":
        sampler_name = "UniPC"
        shared.log.warning(f'Sampler not compatible: name=PLMS fallback={sampler_name}')
    return sampler_name


def get_sampler_index(sampler_name: str) -> int:
    sampler_index = 0
    for i, sampler in enumerate(sd_samplers.samplers):
        if sampler.name == sampler_name:
            sampler_index = i
            break
    return sampler_index


def slerp(val, lo, hi): # from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    lo_norm = lo / torch.norm(lo, dim=1, keepdim=True)
    hi_norm = hi / torch.norm(hi, dim=1, keepdim=True)
    dot = (lo_norm * hi_norm).sum(1)
    dot_mean = dot.mean()
    if dot_mean > 0.9999: # simplifies slerp to lerp if vectors are nearly parallel
        return lo * val + hi * (1 - val)
    if dot_mean < 0.0001: # also simplifies slerp to lerp to avoid division-by-zero later on
        return lo * (1.0 - val) + hi * val
    omega = torch.acos(dot)
    so = torch.sin(omega)
    lo_res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1)
    hi_res = (torch.sin(val * omega) / so).unsqueeze(1)
    # lo_res[lo_res != lo_res] = 0 # replace nans with zeros, but should not happen with dot_mean filtering
    # hi_res[hi_res != hi_res] = 0
    res = lo * lo_res + hi * hi_res
    return res


def slerp_alt(val, lo, hi): # from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    lo_norm = lo / torch.linalg.norm(lo, dim=1, keepdim=True)
    hi_norm = hi / torch.linalg.norm(hi, dim=1, keepdim=True)
    dot = (lo_norm * hi_norm).sum(1)
    dot_mean = dot.mean().abs()
    if dot_mean > 0.9999: # simplifies slerp to lerp if vectors are nearly parallel
        lerp_val = lo * val + hi * (1 - val)
        return lerp_val / torch.linalg.norm(lerp_val) * torch.sqrt(torch.linalg.norm(hi_norm) * torch.linalg.norm(lo_norm))
    if dot_mean < 0.0001: # also simplifies slerp to lerp to avoid division-by-zero later on
        lerp_val = lo * (1.0 - val) + hi * val
        return lerp_val / torch.linalg.norm(lerp_val) * torch.sqrt(torch.linalg.norm(hi_norm) * torch.linalg.norm(lo_norm))
    omega = torch.acos(dot)
    so = torch.sin(omega)
    lo_res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1)
    hi_res = (torch.sin(val * omega) / so).unsqueeze(1)
    res = lo * lo_res + hi * hi_res
    return res


def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    eta_noise_seed_delta = shared.opts.eta_noise_seed_delta or 0
    xs = []
    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and ((len(seeds) > 1 and shared.opts.enable_batch_seeds) or (eta_noise_seed_delta > 0)):
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None
    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)
        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]
            subnoise = devices.randn(subseed, noise_shape)
        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)
        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)
        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)
            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x
        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)
            if eta_noise_seed_delta > 0:
                torch.manual_seed(seed + eta_noise_seed_delta)
            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))
        xs.append(noise)
    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(shared.device) for n in sampler_noises]
    x = torch.stack(xs).to(shared.device)
    return x


def decode_first_stage(model, x, full_quality=True):
    if not shared.opts.keep_incomplete and (shared.state.skipped or shared.state.interrupted):
        shared.log.debug(f'Decode VAE: skipped={shared.state.skipped} interrupted={shared.state.interrupted}')
        x_sample = torch.zeros((len(x), 3, x.shape[2] * 8, x.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
        return x_sample
    prev_job = shared.state.job
    shared.state.job = 'VAE'
    with devices.autocast(disable = x.dtype==devices.dtype_vae):
        try:
            if full_quality:
                if hasattr(model, 'decode_first_stage'):
                    # x_sample = model.decode_first_stage(x) * 0.5 + 0.5
                    x_sample = model.decode_first_stage(x)
                elif hasattr(model, 'vae'):
                    x_sample = processing_vae.vae_decode(latents=x, model=model, output_type='np', full_quality=full_quality)
                else:
                    x_sample = x
                    shared.log.error('Decode VAE unknown model')
            else:
                from modules import sd_vae_taesd
                x_sample = torch.zeros((len(x), 3, x.shape[2] * 8, x.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
                for i in range(len(x_sample)):
                    x_sample[i] = sd_vae_taesd.decode(x[i]) * 0.5 + 0.5
        except Exception as e:
            x_sample = x
            shared.log.error(f'Decode VAE: {e}')
    shared.state.job = prev_job
    return x_sample


def get_fixed_seed(seed):
    if seed is None or seed == '' or seed == -1:
        random.seed()
        seed = int(random.randrange(4294967294))
    return seed


def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)


def old_hires_fix_first_pass_dimensions(width, height):
    """old algorithm for auto-calculating first pass size"""
    desired_pixel_count = 512 * 512
    actual_pixel_count = width * height
    scale = math.sqrt(desired_pixel_count / actual_pixel_count)
    width = math.ceil(scale * width / 64) * 64
    height = math.ceil(scale * height / 64) * 64
    return width, height


def txt2img_image_conditioning(p, x, width=None, height=None):
    width = width or p.width
    height = height or p.height
    if p.sd_model.model.conditioning_key in {'hybrid', 'concat'}: # Inpainting models
        image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
        image_conditioning = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(image_conditioning))
        image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0) # pylint: disable=not-callable
        image_conditioning = image_conditioning.to(x.dtype)
        return image_conditioning
    elif p.sd_model.model.conditioning_key == "crossattn-adm": # UnCLIP models
        return x.new_zeros(x.shape[0], 2*p.sd_model.noise_augmentor.time_embed.dim, dtype=x.dtype, device=x.device)
    else:
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)


def img2img_image_conditioning(p, source_image, latent_image, image_mask=None):
    from ldm.models.diffusion.ddpm import LatentDepth2ImageDiffusion
    source_image = devices.cond_cast_float(source_image)

    def depth2img_image_conditioning(source_image):
        # Use the AddMiDaS helper to Format our source image to suit the MiDaS model
        from ldm.data.util import AddMiDaS
        transformer = AddMiDaS(model_type="dpt_hybrid")
        transformed = transformer({"jpg": rearrange(source_image[0], "c h w -> h w c")})
        midas_in = torch.from_numpy(transformed["midas_in"][None, ...]).to(device=shared.device)
        midas_in = repeat(midas_in, "1 ... -> n ...", n=p.batch_size)
        conditioning_image = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(source_image))
        conditioning = torch.nn.functional.interpolate(
            p.sd_model.depth_model(midas_in),
            size=conditioning_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )
        (depth_min, depth_max) = torch.aminmax(conditioning)
        conditioning = 2. * (conditioning - depth_min) / (depth_max - depth_min) - 1.
        return conditioning

    def edit_image_conditioning(source_image):
        conditioning_image = p.sd_model.encode_first_stage(source_image).mode()
        return conditioning_image

    def unclip_image_conditioning(source_image):
        c_adm = p.sd_model.embedder(source_image)
        if p.sd_model.noise_augmentor is not None:
            noise_level = 0
            c_adm, noise_level_emb = p.sd_model.noise_augmentor(c_adm, noise_level=repeat(torch.tensor([noise_level]).to(c_adm.device), '1 -> b', b=c_adm.shape[0]))
            c_adm = torch.cat((c_adm, noise_level_emb), 1)
        return c_adm

    def inpainting_image_conditioning(source_image, latent_image, image_mask=None):
        # Handle the different mask inputs
        if image_mask is not None:
            if torch.is_tensor(image_mask):
                conditioning_mask = image_mask
            else:
                conditioning_mask = np.array(image_mask.convert("L"))
                conditioning_mask = conditioning_mask.astype(np.float32) / 255.0
                conditioning_mask = torch.from_numpy(conditioning_mask[None, None])
                # Inpainting model uses a discretized mask as input, so we round to either 1.0 or 0.0
                conditioning_mask = torch.round(conditioning_mask)
        else:
            conditioning_mask = source_image.new_ones(1, 1, *source_image.shape[-2:])
        # Create another latent image, this time with a masked version of the original input.
        # Smoothly interpolate between the masked and unmasked latent conditioning image using a parameter.
        conditioning_mask = conditioning_mask.to(device=source_image.device, dtype=source_image.dtype)
        conditioning_image = torch.lerp(
            source_image,
            source_image * (1.0 - conditioning_mask),
            getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight)
        )
        # Encode the new masked image using first stage of network.
        conditioning_image = p.sd_model.get_first_stage_encoding(p.sd_model.encode_first_stage(conditioning_image))
        # Create the concatenated conditioning tensor to be fed to `c_concat`
        conditioning_mask = torch.nn.functional.interpolate(conditioning_mask, size=latent_image.shape[-2:])
        conditioning_mask = conditioning_mask.expand(conditioning_image.shape[0], -1, -1, -1)
        image_conditioning = torch.cat([conditioning_mask, conditioning_image], dim=1)
        image_conditioning = image_conditioning.to(device=shared.device, dtype=source_image.dtype)
        return image_conditioning

    def diffusers_image_conditioning(_source_image, latent_image, _image_mask=None):
        # shared.log.warning('Diffusers not implemented: img2img_image_conditioning')
        return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)

    # HACK: Using introspection as the Depth2Image model doesn't appear to uniquely
    # identify itself with a field common to all models. The conditioning_key is also hybrid.
    if shared.native:
        return diffusers_image_conditioning(source_image, latent_image, image_mask)
    if isinstance(p.sd_model, LatentDepth2ImageDiffusion):
        return depth2img_image_conditioning(source_image)
    if hasattr(p.sd_model, 'cond_stage_key') and p.sd_model.cond_stage_key == "edit":
        return edit_image_conditioning(source_image)
    if hasattr(p.sampler, 'conditioning_key') and p.sampler.conditioning_key in {'hybrid', 'concat'}:
        return inpainting_image_conditioning(source_image, latent_image, image_mask=image_mask)
    if hasattr(p.sampler, 'conditioning_key') and p.sampler.conditioning_key == "crossattn-adm":
        return unclip_image_conditioning(source_image)
    # Dummy zero conditioning if we're not using inpainting or depth model.
    return latent_image.new_zeros(latent_image.shape[0], 5, 1, 1)


def validate_sample(tensor):
    t0 = time.time()
    if not isinstance(tensor, np.ndarray) and not isinstance(tensor, torch.Tensor):
        return tensor
    dtype = tensor.dtype
    if tensor.dtype == torch.bfloat16: # numpy does not support bf16
        tensor = tensor.to(torch.float16)
    if isinstance(tensor, torch.Tensor) and hasattr(tensor, 'detach'):
        sample = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        sample = tensor
    else:
        shared.log.warning(f'Decode: type={type(tensor)} unknown sample')
    sample = 255.0 * np.moveaxis(sample, 0, 2) if not shared.native else 255.0 * sample
    with warnings.catch_warnings(record=True) as w:
        cast = sample.astype(np.uint8)
    if len(w) > 0:
        nans = np.isnan(sample).sum()
        cast = np.nan_to_num(sample)
        cast = cast.astype(np.uint8)
        vae = shared.sd_model.vae.dtype if hasattr(shared.sd_model, 'vae') else None
        upcast = getattr(shared.sd_model.vae.config, 'force_upcast', None) if hasattr(shared.sd_model, 'vae') and hasattr(shared.sd_model.vae, 'config') else None
        shared.log.error(f'Decode: sample={sample.shape} invalid={nans} dtype={dtype} vae={vae} upcast={upcast} failed to validate')
        if upcast is not None and not upcast:
            setattr(shared.sd_model.vae.config, 'force_upcast', True) # noqa: B010
            shared.log.warning('Decode: upcast=True set, retry operation')
    t1 = time.time()
    timer.process.add('validate', t1 - t0)
    return cast


def resize_init_images(p):
    if getattr(p, 'image', None) is not None and getattr(p, 'init_images', None) is None:
        p.init_images = [p.image]
    if getattr(p, 'init_images', None) is not None and len(p.init_images) > 0:
        tgt_width, tgt_height = 8 * math.ceil(p.init_images[0].width / 8), 8 * math.ceil(p.init_images[0].height / 8)
        if p.init_images[0].size != (tgt_width, tgt_height):
            shared.log.debug(f'Resizing init images: original={p.init_images[0].width}x{p.init_images[0].height} target={tgt_width}x{tgt_height}')
            p.init_images = [images.resize_image(1, image, tgt_width, tgt_height, upscaler_name=None) for image in p.init_images]
            p.height = tgt_height
            p.width = tgt_width
            sd_hijack_hypertile.hypertile_set(p)
        if getattr(p, 'mask', None) is not None and p.mask.size != (tgt_width, tgt_height):
            p.mask = images.resize_image(1, p.mask, tgt_width, tgt_height, upscaler_name=None)
        if getattr(p, 'init_mask', None) is not None and p.init_mask.size != (tgt_width, tgt_height):
            p.init_mask = images.resize_image(1, p.init_mask, tgt_width, tgt_height, upscaler_name=None)
        if getattr(p, 'mask_for_overlay', None) is not None and p.mask_for_overlay.size != (tgt_width, tgt_height):
            p.mask_for_overlay = images.resize_image(1, p.mask_for_overlay, tgt_width, tgt_height, upscaler_name=None)
        return tgt_width, tgt_height
    return p.width, p.height


def resize_hires(p, latents): # input=latents output=pil if not latent_upscaler else latent
    if not torch.is_tensor(latents):
        shared.log.warning('Hires: input is not tensor')
        first_pass_images = processing_vae.vae_decode(latents=latents, model=shared.sd_model, full_quality=p.full_quality, output_type='pil', width=p.width, height=p.height)
        return first_pass_images
    latent_upscaler = shared.latent_upscale_modes.get(p.hr_upscaler, None)
    # shared.log.info(f'Hires: upscaler={p.hr_upscaler} width={p.hr_upscale_to_x} height={p.hr_upscale_to_y} images={latents.shape[0]}')
    if latent_upscaler is not None:
        return torch.nn.functional.interpolate(latents, size=(p.hr_upscale_to_y // 8, p.hr_upscale_to_x // 8), mode=latent_upscaler["mode"], antialias=latent_upscaler["antialias"])
    first_pass_images = processing_vae.vae_decode(latents=latents, model=shared.sd_model, full_quality=p.full_quality, output_type='pil', width=p.width, height=p.height)
    if p.hr_upscale_to_x == 0 or (p.hr_upscale_to_y == 0 and hasattr(p, 'init_hr')):
        shared.log.error('Hires: missing upscaling dimensions')
        return first_pass_images
    resized_images = []
    for img in first_pass_images:
        if latent_upscaler is None:
            resized_image = images.resize_image(p.hr_resize_mode, img, p.hr_upscale_to_x, p.hr_upscale_to_y, upscaler_name=p.hr_upscaler, context=p.hr_resize_context)
        else:
            resized_image = img
        resized_images.append(resized_image)
    devices.torch_gc()
    return resized_images


def fix_prompts(prompts, negative_prompts, prompts_2, negative_prompts_2):
    if type(prompts) is str:
        prompts = [prompts]
    if type(negative_prompts) is str:
        negative_prompts = [negative_prompts]
    while len(negative_prompts) < len(prompts):
        negative_prompts.append(negative_prompts[-1])
    while len(prompts) < len(negative_prompts):
        prompts.append(prompts[-1])
    if type(prompts_2) is str:
        prompts_2 = [prompts_2]
    if type(prompts_2) is list:
        while len(prompts_2) < len(prompts):
            prompts_2.append(prompts_2[-1])
    if type(negative_prompts_2) is str:
        negative_prompts_2 = [negative_prompts_2]
    if type(negative_prompts_2) is list:
        while len(negative_prompts_2) < len(prompts_2):
            negative_prompts_2.append(negative_prompts_2[-1])
    return prompts, negative_prompts, prompts_2, negative_prompts_2


def calculate_base_steps(p, use_denoise_start, use_refiner_start):
    if len(getattr(p, 'timesteps', [])) > 0:
        return None
    if not is_txt2img():
        if use_denoise_start and shared.sd_model_type == 'sdxl':
            steps = p.steps // (1 - p.refiner_start)
        elif shared.sd_model_type == 'omnigen':
            steps = p.steps
        elif p.denoising_strength > 0:
            steps = (p.steps // p.denoising_strength) + 1
        else:
            steps = p.steps
    elif use_refiner_start and shared.sd_model_type == 'sdxl':
        steps = (p.steps // p.refiner_start) + 1
    else:
        steps = p.steps
    debug_steps(f'Steps: type=base input={p.steps} output={steps} task={sd_models.get_diffusers_task(shared.sd_model)} refiner={use_refiner_start} denoise={p.denoising_strength} model={shared.sd_model_type}')
    return max(1, int(steps))


def calculate_hires_steps(p):
    # if len(getattr(p, 'timesteps', [])) > 0:
    #    return None
    if p.hr_second_pass_steps > 0:
        steps = (p.hr_second_pass_steps // p.denoising_strength) + 1
    elif p.denoising_strength > 0:
        steps = (p.steps // p.denoising_strength) + 1
    else:
        steps = 0
    debug_steps(f'Steps: type=hires input={p.hr_second_pass_steps} output={steps} denoise={p.denoising_strength} model={shared.sd_model_type}')
    return max(1, int(steps))


def calculate_refiner_steps(p):
    # if len(getattr(p, 'timesteps', [])) > 0:
    #    return None
    if "StableDiffusionXL" in shared.sd_refiner.__class__.__name__:
        if p.refiner_start > 0 and p.refiner_start < 1:
            #steps = p.refiner_steps // (1 - p.refiner_start) # SDXL with denoise strenght
            steps = (p.refiner_steps // (1 - p.refiner_start) // 2) + 1
        elif p.denoising_strength > 0:
            steps = (p.refiner_steps // p.denoising_strength) + 1
        else:
            steps = 0
    else:
        #steps = p.refiner_steps # SD 1.5 with denoise strenght
        steps = (p.refiner_steps * 1.25) + 1
    debug_steps(f'Steps: type=refiner input={p.refiner_steps} output={steps} start={p.refiner_start} denoise={p.denoising_strength}')
    return max(1, int(steps))


def get_generator(p):
    if shared.opts.diffusers_generator_device == "Unset":
        generator_device = None
        generator = None
    elif getattr(p, "generator", None) is not None:
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
        generator = p.generator
    else:
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
        try:
            p.seeds = [seed if seed != -1 else get_fixed_seed(seed) for seed in p.seeds if seed]
            devices.randn(p.seeds[0])
            generator = [torch.Generator(generator_device).manual_seed(s) for s in p.seeds]
        except Exception as e:
            shared.log.error(f'Torch generator: seeds={p.seeds} device={generator_device} {e}')
            generator = None
    return generator


def set_latents(p):
    def dummy_prepare_latents(*args, **_kwargs):
        return args[0] # just return image to skip re-processing it

    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
    image = shared.sd_model.image_processor.preprocess(p.init_images) # resize to mod8, normalize, transpose, to tensor
    timesteps, steps = retrieve_timesteps(shared.sd_model.scheduler, p.steps, devices.device)
    timesteps, steps = shared.sd_model.get_timesteps(steps, p.denoising_strength, devices.device)
    timestep = timesteps[:1].repeat(p.batch_size) # need to determine level of added noise
    latents = shared.sd_model.prepare_latents(image, timestep, batch_size=p.batch_size, num_images_per_prompt=1, dtype=devices.dtype, device=devices.device, generator=get_generator(p))
    shared.sd_model.prepare_latents = dummy_prepare_latents # stop diffusers processing latents again
    return latents


last_circular = False
def apply_circular(enable, model):
    global last_circular # pylint: disable=global-statement
    if not hasattr(model, 'unet') or not hasattr(model, 'vae'):
        return
    if last_circular == enable:
        return
    try:
        for layer in [layer for layer in model.unet.modules() if type(layer) is torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'
        for layer in [layer for layer in model.vae.modules() if type(layer) is torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'
        last_circular = enable
    except Exception as e:
        debug(f"Diffusers tiling failed: {e}")


def save_intermediate(p, latents, suffix):
    for i in range(len(latents)):
        from modules.processing import create_infotext
        info=create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, [], iteration=p.iteration, position_in_batch=i)
        decoded = processing_vae.vae_decode(latents=latents, model=shared.sd_model, output_type='pil', full_quality=p.full_quality, width=p.width, height=p.height)
        for j in range(len(decoded)):
            images.save_image(decoded[j], path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix=suffix)


def update_sampler(p, sd_model, second_pass=False):
    sampler_selection = p.hr_sampler_name if second_pass else p.sampler_name
    if hasattr(sd_model, 'scheduler'):
        if sampler_selection == 'None':
            return
        if sampler_selection is None:
            sampler = sd_samplers.all_samplers_map.get("UniPC")
        else:
            sampler = sd_samplers.all_samplers_map.get(sampler_selection, None)
        if sampler is None:
            shared.log.warning(f'Sampler: sampler="{sampler_selection}" not found')
            sampler = sd_samplers.all_samplers_map.get("UniPC")
        sampler = sd_samplers.create_sampler(sampler.name, sd_model)
        if sampler is None or sampler_selection == 'Default':
            return
        sampler_options = []
        if sampler.config.get('rescale_betas_zero_snr', False) and shared.opts.schedulers_rescale_betas != shared.opts.data_labels.get('schedulers_rescale_betas').default:
            sampler_options.append('rescale')
        if sampler.config.get('thresholding', False) and shared.opts.schedulers_use_thresholding != shared.opts.data_labels.get('schedulers_use_thresholding').default:
            sampler_options.append('dynamic')
        if 'lower_order_final' in sampler.config and shared.opts.schedulers_use_loworder != shared.opts.data_labels.get('schedulers_use_loworder').default:
            sampler_options.append('low order')
        if len(sampler_options) > 0:
            p.extra_generation_params['Sampler options'] = '/'.join(sampler_options)
