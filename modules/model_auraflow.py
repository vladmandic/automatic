import torch
import diffusers


repo_id = 'fal/AuraFlow'


def load_auraflow(_checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = torch.float16

    pipe = diffusers.AuraFlowPipeline.from_pretrained(
        repo_id,
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    devices.torch_gc()
    return pipe
