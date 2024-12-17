import time
import torch
import diffusers


"""
Efficient-Large-Model/Sana_1600M_1024px_MultiLing_diffusers
Efficient-Large-Model/Sana_1600M_1024px_diffusers
Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers
Efficient-Large-Model/Sana_1600M_512px_MultiLing_diffusers
Efficient-Large-Model/Sana_1600M_512px_diffusers
Efficient-Large-Model/Sana_600M_1024px_diffusers
Efficient-Large-Model/Sana_600M_512px_diffusers
"""


def load_sana(checkpoint_info, kwargs={}):
    from modules import shared, sd_models, devices, modelloader, model_quant
    modelloader.hf_login()

    repo_id = checkpoint_info if isinstance(checkpoint_info, str) else checkpoint_info.path
    repo_id = sd_models.path_to_repo(repo_id)
    kwargs.pop('load_connected_pipeline', None)
    kwargs.pop('safety_checker', None)
    kwargs.pop('requires_safety_checker', None)
    kwargs.pop('torch_dtype', None)

    if 'Sana_1600M' in repo_id:
        if devices.dtype == torch.bfloat16:
            repo_id = 'Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers'
            kwargs['variant'] = 'bf16'
            kwargs['torch_dtype'] = devices.dtype
        else:
            repo_id = 'Efficient-Large-Model/Sana_1600M_1024px_diffusers'
            kwargs['variant'] = 'fp16'
    if 'Sana_600M' in repo_id:
        repo_id = 'Efficient-Large-Model/Sana_600M_1024px_diffusers'
        kwargs['variant'] = 'fp16'

    kwargs = model_quant.create_bnb_config(kwargs)
    shared.log.debug(f'Load model: type=Sana repo="{repo_id}" args={kwargs}')
    t0 = time.time()
    pipe = diffusers.SanaPipeline.from_pretrained(repo_id, cache_dir = shared.opts.diffusers_dir, **kwargs)
    if devices.dtype == torch.bfloat16 or devices.dtype == torch.float32:
        pipe.transformer = pipe.transformer.to(dtype=devices.dtype)
        pipe.text_encoder = pipe.text_encoder.to(dtype=devices.dtype)
        pipe.vae = pipe.vae.to(dtype=devices.dtype)
    if devices.dtype == torch.float16:
        pipe.transformer = pipe.transformer.to(dtype=devices.dtype)
        pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float32) # gemma2 does not support fp16
        pipe.vae = pipe.vae.to(dtype=torch.float32) # dc-ae often overflows in fp16
    if shared.opts.diffusers_eval:
        pipe.text_encoder.eval()
        pipe.transformer.eval()
    t1 = time.time()
    shared.log.debug(f'Load model: type=Sana target={devices.dtype} te={pipe.text_encoder.dtype} transformer={pipe.transformer.dtype} vae={pipe.vae.dtype} time={t1-t0:.2f}')

    devices.torch_gc()
    return pipe
