import os
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from modules import shared, devices, sd_models, sd_vae, sd_vae_taesd, errors


debug = os.environ.get('SD_VAE_DEBUG', None) is not None
log_debug = shared.log.trace if debug else lambda *args, **kwargs: None
log_debug('Trace: VAE')
last_latent = None


def create_latents(image, p, dtype=None, device=None):
    from modules.processing import create_random_tensors
    from PIL import Image
    if image is None:
        return image
    elif isinstance(image, Image.Image):
        latents = vae_encode(image, model=shared.sd_model, full_quality=p.full_quality)
    elif isinstance(image, list):
        latents = [vae_encode(i, model=shared.sd_model, full_quality=p.full_quality).squeeze(dim=0) for i in image]
        latents = torch.stack(latents, dim=0).to(shared.device)
    else:
        shared.log.warning(f'Latents: input type: {type(image)} {image}')
        return image
    noise = p.denoising_strength * create_random_tensors(latents.shape[1:], seeds=p.all_seeds, subseeds=p.all_subseeds, subseed_strength=p.subseed_strength, p=p)
    latents = (1 - p.denoising_strength) * latents + noise
    if dtype is not None:
        latents = latents.to(dtype=dtype)
    if device is not None:
        latents = latents.to(device=device)
    return latents


def full_vae_decode(latents, model):
    t0 = time.time()
    if not hasattr(model, 'vae'):
        shared.log.error('VAE not found in model')
        return []
    if debug:
        devices.torch_gc(force=True)
        shared.mem_mon.reset()

    base_device = None
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False):
        base_device = sd_models.move_base(model, devices.cpu)

    if shared.opts.diffusers_offload_mode == "balanced":
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    elif shared.opts.diffusers_offload_mode != "sequential":
        sd_models.move_model(model.vae, devices.device)

    upcast = (model.vae.dtype == torch.float16) and getattr(model.vae.config, 'force_upcast', False)
    if upcast:
        if hasattr(model, 'upcast_vae'): # this is done by diffusers automatically if output_type != 'latent'
            model.upcast_vae()
        model.vae = model.vae.to(dtype=torch.float32)
        latents = latents.to(torch.float32)

    if getattr(model.vae, "post_quant_conv", None) is not None:
        latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)
    elif shared.opts.no_half_vae:
        latents = latents.to(torch.float32)
    else:
        latents = latents.to(model.vae.device)

    # normalize latents
    latents_mean = model.vae.config.get("latents_mean", None)
    latents_std = model.vae.config.get("latents_std", None)
    scaling_factor = model.vae.config.get("scaling_factor", None)
    shift_factor = model.vae.config.get("shift_factor", None)
    if latents_mean and latents_std:
        latents_mean = (torch.tensor(latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype))
        latents_std = (torch.tensor(latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype))
        latents = latents * latents_std / scaling_factor + latents_mean
    else:
        latents = latents / scaling_factor
    if shift_factor:
        latents = latents + shift_factor

    vae_name = os.path.splitext(os.path.basename(sd_vae.loaded_vae_file))[0] if sd_vae.loaded_vae_file is not None else "default"
    vae_stats = f'name="{vae_name}" dtype={model.vae.dtype} device={model.vae.device} upcast={upcast} slicing={getattr(model.vae, "use_slicing", None)} tiling={getattr(model.vae, "use_tiling", None)}'
    latents_stats = f'shape={latents.shape} dtype={latents.dtype} device={latents.device}'
    stats = f'vae {vae_stats} latents {latents_stats}'

    try:
        decoded = model.vae.decode(latents, return_dict=False)[0]
    except Exception as e:
        shared.log.error(f'VAE decode: {stats} {e}')
        errors.display(e, 'VAE decode')
        decoded = []

    # delete vae after OpenVINO compile
    if 'VAE' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx" and shared.compiled_model_state.first_pass_vae:
        shared.compiled_model_state.first_pass_vae = False
        if not shared.opts.openvino_disable_memory_cleanup and hasattr(shared.sd_model, "vae"):
            model.vae.apply(sd_models.convert_to_faketensors)
            devices.torch_gc(force=True)

    if shared.opts.diffusers_offload_mode == "balanced":
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    elif shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and base_device is not None:
        sd_models.move_base(model, base_device)
    t1 = time.time()
    if debug:
        log_debug(f'VAE memory: {shared.mem_mon.read()}')
    shared.log.debug(f'VAE decode: {stats} time={round(t1-t0, 3)}')
    return decoded


def full_vae_encode(image, model):
    log_debug(f'VAE encode: name={sd_vae.loaded_vae_file if sd_vae.loaded_vae_file is not None else "baked"} dtype={model.vae.dtype} upcast={model.vae.config.get("force_upcast", None)}')
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        log_debug('Moving to CPU: model=UNet')
        unet_device = model.unet.device
        sd_models.move_model(model.unet, devices.cpu)
    if not shared.opts.diffusers_offload_mode == "sequential" and hasattr(model, 'vae'):
        sd_models.move_model(model.vae, devices.device)
    encoded = model.vae.encode(image.to(model.vae.device, model.vae.dtype)).latent_dist.sample()
    if shared.opts.diffusers_move_unet and not getattr(model, 'has_accelerate', False) and hasattr(model, 'unet'):
        sd_models.move_model(model.unet, unet_device)
    return encoded


def taesd_vae_decode(latents):
    log_debug(f'VAE decode: name=TAESD images={len(latents)} latents={latents.shape} slicing={shared.opts.diffusers_vae_slicing}')
    if len(latents) == 0:
        return []
    if shared.opts.diffusers_vae_slicing and len(latents) > 1:
        decoded = torch.zeros((len(latents), 3, latents.shape[2] * 8, latents.shape[3] * 8), dtype=devices.dtype_vae, device=devices.device)
        for i in range(latents.shape[0]):
            decoded[i] = sd_vae_taesd.decode(latents[i])
    else:
        decoded = sd_vae_taesd.decode(latents)
    return decoded


def taesd_vae_encode(image):
    log_debug(f'VAE encode: name=TAESD image={image.shape}')
    encoded = sd_vae_taesd.encode(image)
    return encoded


def vae_decode(latents, model, output_type='np', full_quality=True, width=None, height=None):
    global last_latent # pylint: disable=global-statement
    t0 = time.time()
    if latents is None or not torch.is_tensor(latents): # already decoded
        last_latent = None
        return latents
    prev_job = shared.state.job
    shared.state.job = 'VAE'
    if latents.shape[0] == 0:
        shared.log.error(f'VAE nothing to decode: {latents.shape}')
        return []
    if shared.state.interrupted or shared.state.skipped:
        return []
    if not hasattr(model, 'vae'):
        shared.log.error('VAE not found in model')
        return []

    if hasattr(model, "_unpack_latents") and hasattr(model, "vae_scale_factor") and width is not None and height is not None: # FLUX
        latents = model._unpack_latents(latents, height, width, model.vae_scale_factor) # pylint: disable=protected-access
    if len(latents.shape) == 3: # lost a batch dim in hires
        latents = latents.unsqueeze(0)
    if latents.shape[0] == 4 and latents.shape[1] != 4: # likely animatediff latent
        latents = latents.permute(1, 0, 2, 3)
    last_latent = latents.clone().detach()

    if latents.shape[-1] <= 4: # not a latent, likely an image
        decoded = latents.float().cpu().numpy()
    elif full_quality and hasattr(shared.sd_model, "vae"):
        decoded = full_vae_decode(latents=latents, model=shared.sd_model)
    else:
        decoded = taesd_vae_decode(latents=latents)

    if torch.is_tensor(decoded):
        if hasattr(model, 'image_processor'):
            imgs = model.image_processor.postprocess(decoded, output_type=output_type)
        else:
            import diffusers
            model.image_processor = diffusers.image_processor.VaeImageProcessor()
            imgs = model.image_processor.postprocess(decoded, output_type=output_type)
    else:
        imgs = decoded if isinstance(decoded, list) or isinstance(decoded, np.ndarray) else [decoded]

    shared.state.job = prev_job
    if shared.cmd_opts.profile or debug:
        t1 = time.time()
        shared.log.debug(f'Profile: VAE decode: {t1-t0:.2f}')
    devices.torch_gc()
    return imgs


def vae_encode(image, model, full_quality=True): # pylint: disable=unused-variable
    if shared.state.interrupted or shared.state.skipped:
        return []
    if not hasattr(model, 'vae'):
        shared.log.error('VAE not found in model')
        return []
    tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(devices.device, devices.dtype_vae)
    if full_quality:
        tensor = tensor * 2 - 1
        latents = full_vae_encode(image=tensor, model=shared.sd_model)
    else:
        latents = taesd_vae_encode(image=tensor)
    devices.torch_gc()
    return latents


def reprocess(gallery):
    from PIL import Image
    from modules import images
    if last_latent is None or gallery is None:
        return None
    shared.log.info(f'Reprocessing: latent={last_latent.shape}')
    reprocessed = vae_decode(last_latent, shared.sd_model, output_type='pil', full_quality=True)
    outputs = []
    for i0, i1 in zip(gallery, reprocessed):
        if isinstance(i1, np.ndarray):
            i1 = Image.fromarray(i1)
        fn = i0['name']
        i0 = Image.open(fn)
        fn = os.path.splitext(os.path.basename(fn))[0] + '-re'
        i0.load() # wait for info to be populated
        i1.info = i0.info
        info, _params = images.read_info_from_image(i0)
        if shared.opts.samples_save:
            images.save_image(i1, info=info, forced_filename=fn)
            i1.already_saved_as = fn
        outputs.append(i0)
        outputs.append(i1)
    return outputs
