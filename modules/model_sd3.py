import os
import warnings
import torch
import diffusers
import transformers
import rich.traceback


rich.traceback.install()
warnings.filterwarnings(action="ignore", category=FutureWarning)
loggedin = False


def load_sd3(fn=None, cache_dir=None, config=None):
    from modules import devices, modelloader
    modelloader.hf_login()
    repo_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    model_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    dtype = torch.float16
    kwargs = {}
    if fn is not None and fn.endswith('.safetensors') and os.path.exists(fn):
        model_id = fn
        loader = diffusers.StableDiffusion3Pipeline.from_single_file
        diffusers_minor = int(diffusers.__version__.split('.')[1])
        fn_size = os.path.getsize(fn)
        if diffusers_minor < 30 or fn_size < 5e9: # te1/te2 do not get loaded correctly in diffusers 0.29.0 or model is without te1/te2
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
        elif fn_size < 1e10: # if model is below 10gb it does not have te4
            kwargs = {
                'text_encoder_3': None,
            }
        else:
            kwargs = {}
    else:
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


def load_te3(pipe, te3=None, cache_dir=None):
    from modules import devices, modelloader
    modelloader.hf_login()
    repo_id = 'stabilityai/stable-diffusion-3-medium-diffusers'
    if pipe is None or not hasattr(pipe, 'text_encoder_3'):
        return pipe
    if 'fp16' in te3.lower():
        pipe.text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
            repo_id,
            subfolder='text_encoder_3',
            # torch_dtype=dtype,
            cache_dir=cache_dir,
            torch_dtype=pipe.text_encoder.dtype,
        )
    elif 'fp8' in te3.lower():
        from installer import install
        install('bitsandbytes', quiet=True)
        quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        pipe.text_encoder_3 = transformers.T5EncoderModel.from_pretrained(
            repo_id,
            subfolder='text_encoder_3',
            quantization_config=quantization_config,
            cache_dir=cache_dir,
            torch_dtype=pipe.text_encoder.dtype,
        )
    else:
        pipe.text_encoder_3 = None
    if getattr(pipe, 'text_encoder_3', None) is not None and getattr(pipe, 'tokenizer_3', None) is None:
        pipe.tokenizer_3 = transformers.T5TokenizerFast.from_pretrained(
            repo_id,
            subfolder='tokenizer_3',
            cache_dir=cache_dir,
        )
    devices.torch_gc()
