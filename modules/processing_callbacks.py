import typing
import os
import time
import torch
import numpy as np
from modules import shared, processing_correction, extra_networks, timer, prompt_parser_diffusers


p = None
debug = os.environ.get('SD_CALLBACK_DEBUG', None) is not None
debug_callback = shared.log.trace if debug else lambda *args, **kwargs: None


def set_callbacks_p(processing):
    global p # pylint: disable=global-statement
    p = processing


def prompt_callback(step, kwargs):
    if prompt_parser_diffusers.embedder is None or 'prompt_embeds' not in kwargs:
        return kwargs
    try:
        prompt_embeds = prompt_parser_diffusers.embedder('prompt_embeds', step + 1)
        negative_prompt_embeds = prompt_parser_diffusers.embedder('negative_prompt_embeds', step + 1)
        if p.cfg_scale > 1:  # Perform guidance
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)  # Combined embeds
        assert prompt_embeds.shape == kwargs['prompt_embeds'].shape, f"prompt_embed shape mismatch {kwargs['prompt_embeds'].shape} {prompt_embeds.shape}"
        kwargs['prompt_embeds'] = prompt_embeds
    except Exception as e:
        debug_callback(f"Callback: {e}")
    return kwargs


def diffusers_callback_legacy(step: int, timestep: int, latents: typing.Union[torch.FloatTensor, np.ndarray]):
    if p is None:
        return
    if isinstance(latents, np.ndarray): # latents from Onnx pipelines is ndarray.
        latents = torch.from_numpy(latents)
    shared.state.sampling_step = step
    shared.state.current_latent = latents
    latents = processing_correction.correction_callback(p, timestep, {'latents': latents})
    if shared.state.interrupted or shared.state.skipped:
        raise AssertionError('Interrupted...')
    if shared.state.paused:
        shared.log.debug('Sampling paused')
        while shared.state.paused:
            if shared.state.interrupted or shared.state.skipped:
                raise AssertionError('Interrupted...')
            time.sleep(0.1)


def diffusers_callback(pipe, step: int = 0, timestep: int = 0, kwargs: dict = {}):
    t0 = time.time()
    if p is None:
        return kwargs
    latents = kwargs.get('latents', None)
    if debug:
        debug_callback(f'Callback: step={step} timestep={timestep} latents={latents.shape if latents is not None else None} kwargs={list(kwargs)}')
    order = getattr(pipe.scheduler, "order", 1) if hasattr(pipe, 'scheduler') else 1
    shared.state.sampling_step = step // order
    if shared.state.interrupted or shared.state.skipped:
        raise AssertionError('Interrupted...')
    if shared.state.paused:
        shared.log.debug('Sampling paused')
        while shared.state.paused:
            if shared.state.interrupted or shared.state.skipped:
                raise AssertionError('Interrupted...')
            time.sleep(0.1)
    if hasattr(p, "stepwise_lora") and shared.native:
        extra_networks.activate(p, step=step)
    if latents is None:
        return kwargs
    elif shared.opts.nan_skip:
        assert not torch.isnan(latents[..., 0, 0]).all(), f'NaN detected at step {step}: Skipping...'
    if len(getattr(p, 'ip_adapter_names', [])) > 0 and p.ip_adapter_names[0] != 'None':
        ip_adapter_scales = list(p.ip_adapter_scales)
        ip_adapter_starts = list(p.ip_adapter_starts)
        ip_adapter_ends = list(p.ip_adapter_ends)
        if any(end != 1 for end in ip_adapter_ends) or any(start != 0 for start in ip_adapter_starts):
            for i in range(len(ip_adapter_scales)):
                ip_adapter_scales[i] *= float(step >= pipe.num_timesteps * ip_adapter_starts[i])
                ip_adapter_scales[i] *= float(step <= pipe.num_timesteps * ip_adapter_ends[i])
                debug_callback(f"Callback: IP Adapter scales={ip_adapter_scales}")
            pipe.set_ip_adapter_scale(ip_adapter_scales)
    if step != getattr(pipe, 'num_timesteps', 0):
        kwargs = processing_correction.correction_callback(p, timestep, kwargs, initial=step == 0)
    kwargs = prompt_callback(step, kwargs)  # monkey patch for diffusers callback issues
    if step == int(getattr(pipe, 'num_timesteps', 100) * p.cfg_end) and 'prompt_embeds' in kwargs and 'negative_prompt_embeds' in kwargs:
        if "PAG" in shared.sd_model.__class__.__name__:
            pipe._guidance_scale = 1.001 if pipe._guidance_scale > 1 else pipe._guidance_scale  # pylint: disable=protected-access
            pipe._pag_scale = 0.001  # pylint: disable=protected-access
        else:
            pipe._guidance_scale = 0.0  # pylint: disable=protected-access
            for key in {"prompt_embeds", "negative_prompt_embeds", "add_text_embeds", "add_time_ids"} & set(kwargs):
                if kwargs[key] is not None:
                    kwargs[key] = kwargs[key].chunk(2)[-1]
    try:
        current_noise_pred = kwargs.get("noise_pred", None)
        if current_noise_pred is None:
            current_noise_pred = kwargs.get("predicted_image_embedding", None)

        if hasattr(pipe, "_unpack_latents") and hasattr(pipe, "vae_scale_factor"): # FLUX
            if p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5) and p.is_hr_pass:
                width = max(getattr(p, 'width', 0), getattr(p, 'hr_upscale_to_x', 0))
                height = max(getattr(p, 'height', 0), getattr(p, 'hr_upscale_to_y', 0))
            else:
                width = getattr(p, 'width', 0)
                height = getattr(p, 'height', 0)
            shared.state.current_latent = pipe._unpack_latents(kwargs['latents'], height, width, pipe.vae_scale_factor) # pylint: disable=protected-access
            if current_noise_pred is not None:
                shared.state.current_noise_pred = pipe._unpack_latents(current_noise_pred, height, width, pipe.vae_scale_factor) # pylint: disable=protected-access
            else:
                shared.state.current_noise_pred = current_noise_pred
        else:
            shared.state.current_latent = kwargs['latents']
            shared.state.current_noise_pred = current_noise_pred

        if hasattr(pipe, "scheduler") and hasattr(pipe.scheduler, "sigmas") and hasattr(pipe.scheduler, "step_index"):
            shared.state.current_sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index - 1]
            shared.state.current_sigma_next = pipe.scheduler.sigmas[pipe.scheduler.step_index]
    except Exception as e:
        shared.log.error(f'Callback: {e}')
    if shared.cmd_opts.profile and shared.profiler is not None:
        shared.profiler.step()
    t1 = time.time()
    timer.process.add('callback', t1 - t0)
    return kwargs
