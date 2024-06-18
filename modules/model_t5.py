import transformers


def load_t5(t5=None, cache_dir=None):
    from modules import devices, modelloader
    repo_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    if 'fp16' in t5.lower():
        modelloader.hf_login()
        t5 = transformers.T5EncoderModel.from_pretrained(
            repo_id,
            subfolder='text_encoder_3',
            # torch_dtype=dtype,
            cache_dir=cache_dir,
            torch_dtype=devices.dtype,
        )
    elif 'fp4' in t5.lower():
        modelloader.hf_login()
        from installer import install
        install('bitsandbytes', quiet=True)
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
        t5 = transformers.T5EncoderModel.from_pretrained(
            repo_id,
            subfolder='text_encoder_3',
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            torch_dtype=devices.dtype,
        )
    elif 'fp8' in t5.lower():
        modelloader.hf_login()
        from installer import install
        install('bitsandbytes', quiet=True)
        quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        t5 = transformers.T5EncoderModel.from_pretrained(
            repo_id,
            subfolder='text_encoder_3',
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            torch_dtype=devices.dtype,
        )
    else:
        t5 = None
    return t5


def set_t5(pipe, module, t5=None, cache_dir=None):
    from modules import devices
    if pipe is None or not hasattr(pipe, module):
        return pipe
    t5 = load_t5(t5=t5, cache_dir=cache_dir)
    setattr(pipe, module, t5)
    devices.torch_gc()
