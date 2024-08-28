import os
import torch
import diffusers
import transformers


def load_sd3(fn=None, cache_dir=None, config=None):
    from modules import devices, modelloader
    repo_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    model_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    dtype = torch.float16
    kwargs = {}
    if fn is not None and fn.endswith('.safetensors') and os.path.exists(fn):
        model_id = fn
        loader = diffusers.StableDiffusion3Pipeline.from_single_file
        _diffusers_major, diffusers_minor, diffusers_micro = int(diffusers.__version__.split('.')[0]), int(diffusers.__version__.split('.')[1]), int(diffusers.__version__.split('.')[2]) # pylint: disable=use-maxsplit-arg
        fn_size = os.path.getsize(fn)
        if (diffusers_minor <= 29 and diffusers_micro < 1) or fn_size < 5e9: # te1/te2 do not get loaded correctly in diffusers 0.29.0 if model is without te1/te2
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
        model_id = repo_id
        loader = diffusers.StableDiffusion3Pipeline.from_pretrained
    pipe = loader(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        config=config,
        **kwargs,
    )
    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING["stable-diffusion-3"] = diffusers.StableDiffusion3Pipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING["stable-diffusion-3"] = diffusers.StableDiffusion3Img2ImgPipeline
    devices.torch_gc()
    return pipe
