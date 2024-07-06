import diffusers


def load_lumina(_checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, modelloader
    modelloader.hf_login()
    # {'low_cpu_mem_usage': True, 'torch_dtype': torch.float16, 'load_connected_pipeline': True, 'safety_checker': None, 'requires_safety_checker': False}
    if 'torch_dtype' not in diffusers_load_config:
        diffusers_load_config['torch_dtype'] = 'torch.float16'
    if 'low_cpu_mem_usage' in diffusers_load_config:
        del diffusers_load_config['low_cpu_mem_usage']
    if 'load_connected_pipeline' in diffusers_load_config:
        del diffusers_load_config['load_connected_pipeline']
    if 'safety_checker' in diffusers_load_config:
        del diffusers_load_config['safety_checker']
    if 'requires_safety_checker' in diffusers_load_config:
        del diffusers_load_config['requires_safety_checker']
    pipe = diffusers.LuminaText2ImgPipeline.from_pretrained(
        'Alpha-VLLM/Lumina-Next-SFT-diffusers',
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    print('HERE2', diffusers_load_config)
    devices.torch_gc()
    return pipe
