import diffusers


def load_pixart(checkpoint_info, diffusers_load_config={}):
    from modules import shared, devices, modelloader, model_t5
    modelloader.hf_login()
    # shared.opts.data['cuda_dtype'] = 'FP32' # override
    # shared.opts.data['diffusers_model_cpu_offload'] = True # override
    # devices.set_cuda_params()
    fn = checkpoint_info.path.replace('huggingface/', '')
    t5 = model_t5.load_t5(shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
    transformer = diffusers.PixArtTransformer2DModel.from_pretrained(
        fn,
        subfolder = 'transformer',
        cache_dir = shared.opts.diffusers_dir,
        **diffusers_load_config,
    )
    transformer.to(devices.device)
    kwargs = { 'transformer': transformer }
    if t5 is not None:
        kwargs['text_encoder'] = t5
    pipe = diffusers.PixArtSigmaPipeline.from_pretrained(
        'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS',
        cache_dir = shared.opts.diffusers_dir,
        **kwargs,
        **diffusers_load_config,
    )
    devices.torch_gc()
    return pipe
