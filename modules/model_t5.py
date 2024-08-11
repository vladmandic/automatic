import torch
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
    elif 'qint8' in t5.lower():
        modelloader.hf_login()
        from installer import install
        install('optimum-quanto', quiet=True)
        from modules.sd_models_compile import optimum_quanto_model
        t5 = transformers.T5EncoderModel.from_pretrained(
            repo_id,
            subfolder='text_encoder_3',
            cache_dir=cache_dir,
            torch_dtype=devices.dtype,
        )
        t5 = optimum_quanto_model(t5, weights="qint8")
    elif 'int8' in t5.lower():
        modelloader.hf_login()
        from installer import install
        install('nncf==2.7.0', quiet=True)
        from modules.sd_models_compile import nncf_compress_model
        from modules.sd_hijack import NNCF_T5DenseGatedActDense
        t5 = transformers.T5EncoderModel.from_pretrained(
            repo_id,
            subfolder='text_encoder_3',
            cache_dir=cache_dir,
            torch_dtype=devices.dtype,
        )
        for i in range(len(t5.encoder.block)):
            t5.encoder.block[i].layer[1].DenseReluDense = NNCF_T5DenseGatedActDense(
                t5.encoder.block[i].layer[1].DenseReluDense,
                dtype=torch.float32 if devices.dtype != torch.bfloat16 else torch.bfloat16
            )
        t5 = nncf_compress_model(t5)
    else:
        t5 = None
    return t5


def set_t5(pipe, module, t5=None, cache_dir=None):
    from modules import devices, shared
    if pipe is None or not hasattr(pipe, module):
        return pipe
    t5 = load_t5(t5=t5, cache_dir=cache_dir)
    setattr(pipe, module, t5)
    if shared.opts.diffusers_offload_mode == "sequential":
        from accelerate import cpu_offload
        getattr(pipe, module).to("cpu")
        cpu_offload(getattr(pipe, module), devices.device, offload_buffers=len(getattr(pipe, module)._parameters) > 0) # pylint: disable=protected-access
    elif shared.opts.diffusers_offload_mode == "cpu":
        if not hasattr(pipe, "_all_hooks") or len(pipe._all_hooks) == 0: # pylint: disable=protected-access
            pipe.enable_model_cpu_offload(device=devices.device)
    if hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
    devices.torch_gc()
    return pipe
