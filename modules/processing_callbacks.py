import typing
import os
import time
import torch
import numpy as np
from modules import shared, processing_correction, extra_networks


p = None
debug_callback = shared.log.trace if os.environ.get('SD_CALLBACK_DEBUG', None) is not None else lambda *args, **kwargs: None


def set_callbacks_p(processing):
    global p # pylint: disable=global-statement
    p = processing


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


def diffusers_callback(pipe, step: int, timestep: int, kwargs: dict):
    if p is None:
        return kwargs
    latents = kwargs.get('latents', None)
    debug_callback(f'Callback: step={step} timestep={timestep} latents={latents.shape if latents is not None else None} kwargs={list(kwargs)}')
    shared.state.sampling_step = step
    if shared.state.interrupted or shared.state.skipped:
        raise AssertionError('Interrupted...')
    if shared.state.paused:
        shared.log.debug('Sampling paused')
        while shared.state.paused:
            if shared.state.interrupted or shared.state.skipped:
                raise AssertionError('Interrupted...')
            time.sleep(0.1)
    if hasattr(p, "extra_network_data"):
        extra_networks.activate(p, p.extra_network_data, step=step)
    if latents is None:
        return kwargs
    elif shared.opts.nan_skip:
        assert not torch.isnan(latents[..., 0, 0]).all(), f'NaN detected at step {step}: Skipping...'
    if len(getattr(p, 'ip_adapter_names', [])) > 0:
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
        kwargs = processing_correction.correction_callback(p, timestep, kwargs)
    if p.scheduled_prompt and 'prompt_embeds' in kwargs and 'negative_prompt_embeds' in kwargs:
        try:
            i = (step + 1) % len(p.prompt_embeds)
            kwargs["prompt_embeds"] = p.prompt_embeds[i][0:1].expand(kwargs["prompt_embeds"].shape)
            j = (step + 1) % len(p.negative_embeds)
            kwargs["negative_prompt_embeds"] = p.negative_embeds[j][0:1].expand(kwargs["negative_prompt_embeds"].shape)
        except Exception as e:
            shared.log.debug(f"Callback: {e}")
    if step == int(getattr(pipe, 'num_timesteps', 100) * p.cfg_end) and 'prompt_embeds' in kwargs and 'negative_prompt_embeds' in kwargs:
        if "PAG" in shared.sd_model.__class__.__name__:
            pipe._guidance_scale = 1.001 if pipe._guidance_scale > 1 else pipe._guidance_scale  # pylint: disable=protected-access
            pipe._pag_scale = 0.001  # pylint: disable=protected-access
        else:
            pipe._guidance_scale = 0.0  # pylint: disable=protected-access
            for key in {"prompt_embeds", "negative_prompt_embeds", "add_text_embeds", "add_time_ids"} & set(kwargs):
                kwargs[key] = kwargs[key].chunk(2)[-1]
    shared.state.current_latent = kwargs['latents']
    if shared.cmd_opts.profile and shared.profiler is not None:
        shared.profiler.step()
    return kwargs
