import os
import diffusers
import transformers


def load_sd3(checkpoint_info, cache_dir=None, config=None):
    from modules import devices, modelloader, sd_models
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    dtype = devices.dtype
    kwargs = {}
    if checkpoint_info.path is not None and checkpoint_info.path.endswith('.safetensors') and os.path.exists(checkpoint_info.path):
        loader = diffusers.StableDiffusion3Pipeline.from_single_file
        fn_size = os.path.getsize(checkpoint_info.path)
        if fn_size < 5e9:
            kwargs = {
                'text_encoder': transformers.CLIPTextModelWithProjection.from_pretrained(
                    repo_id,
                    subfolder='text_encoder',
                    cache_dir=cache_dir,
                    torch_dtype=dtype,
                ),
                'text_encoder_2': transformers.CLIPTextModelWithProjection.from_pretrained(
                    repo_id,
                    subfolder='text_encoder_2',
                    cache_dir=cache_dir,
                    torch_dtype=dtype,
                ),
                'tokenizer': transformers.CLIPTokenizer.from_pretrained(
                    repo_id,
                    subfolder='tokenizer',
                    cache_dir=cache_dir,
                ),
                'tokenizer_2': transformers.CLIPTokenizer.from_pretrained(
                    repo_id,
                    subfolder='tokenizer_2',
                    cache_dir=cache_dir,
                ),
                'text_encoder_3': None,
            }
        elif fn_size < 1e10: # if model is below 10gb it does not have te3
            kwargs = {
                'text_encoder_3': None,
            }
        else:
            kwargs = {}
    else:
        modelloader.hf_login()
        loader = diffusers.StableDiffusion3Pipeline.from_pretrained
        kwargs['variant'] = 'fp16'
    pipe = loader(
        repo_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        config=config,
        **kwargs,
    )
    devices.torch_gc()
    return pipe
