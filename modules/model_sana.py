import diffusers


def load_sana(checkpoint_info, diffusers_load_config={}):
    from modules import shared, sd_models, devices, modelloader, model_quant
    modelloader.hf_login()

    repo_id = checkpoint_info if isinstance(checkpoint_info, str) else checkpoint_info.path
    repo_id = sd_models.path_to_repo(repo_id)

    diffusers_load_config['variant'] = 'fp16'
    diffusers_load_config['torch_dtype'] = devices.dtype
    diffusers_load_config = model_quant.create_bnb_config(diffusers_load_config)
    pipe = diffusers.SanaPipeline.from_pretrained( # SanaPAGPipeline
        repo_id,
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    if shared.opts.diffusers_eval:
        pipe.text_encoder.eval()
        pipe.transformer.eval()

    devices.torch_gc()
    return pipe
