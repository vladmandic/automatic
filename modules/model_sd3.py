import os
import diffusers
import transformers
from modules import shared, devices, sd_models, sd_unet


def load_overrides(kwargs, cache_dir):
    if shared.opts.sd_unet != 'None':
        try:
            fn = sd_unet.unet_dict[shared.opts.sd_unet]
            kwargs['transformer'] = diffusers.SD3Transformer2DModel.from_single_file(fn, cache_dir=cache_dir, torch_dtype=devices.dtype)
            shared.log.debug(f'Load model: type=SD3 unet="{shared.opts.sd_unet}"')
        except Exception as e:
            shared.log.error(f"Load model: type=SD3 failed to load UNet: {e}")
            shared.opts.sd_unet = 'None'
            sd_unet.failed_unet.append(shared.opts.sd_unet)
    if shared.opts.sd_text_encoder != 'None':
        try:
            from modules.model_te import load_t5, load_vit_l, load_vit_g
            if 'vit-l' in shared.opts.sd_text_encoder.lower():
                kwargs['text_encoder'] = load_vit_l()
                shared.log.debug(f'Load model: type=SD3 variant="vit-l" te="{shared.opts.sd_text_encoder}"')
            elif 'vit-g' in shared.opts.sd_text_encoder.lower():
                kwargs['text_encoder_2'] = load_vit_g()
                shared.log.debug(f'Load model: type=SD3 variant="vit-g" te="{shared.opts.sd_text_encoder}"')
            else:
                kwargs['text_encoder_3'] = load_t5(name=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
                shared.log.debug(f'Load model: type=SD3 variant="t5" te="{shared.opts.sd_text_encoder}"')
        except Exception as e:
            shared.log.error(f"Load model: type=SD3 failed to load T5: {e}")
            shared.opts.sd_text_encoder = 'None'
    if shared.opts.sd_vae != 'None' and shared.opts.sd_vae != 'Automatic':
        try:
            from modules import sd_vae
            vae_file = sd_vae.vae_dict[shared.opts.sd_vae]
            if os.path.exists(vae_file):
                vae_config = os.path.join('configs', 'flux', 'vae', 'config.json')
                kwargs['vae'] = diffusers.AutoencoderKL.from_single_file(vae_file, config=vae_config, cache_dir=cache_dir, torch_dtype=devices.dtype)
                shared.log.debug(f'Load model: type=SD3 vae="{shared.opts.sd_vae}"')
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX failed to load VAE: {e}")
            shared.opts.sd_vae = 'None'
    return kwargs


def load_quants(kwargs, repo_id, cache_dir):
    if len(shared.opts.bnb_quantization) > 0:
        from modules.model_quant import load_bnb
        load_bnb('Load model: type=SD3')
        bnb_config = diffusers.BitsAndBytesConfig(
            load_in_8bit=shared.opts.bnb_quantization_type in ['fp8'],
            load_in_4bit=shared.opts.bnb_quantization_type in ['nf4', 'fp4'],
            bnb_4bit_quant_storage=shared.opts.bnb_quantization_storage,
            bnb_4bit_quant_type=shared.opts.bnb_quantization_type,
            bnb_4bit_compute_dtype=devices.dtype
        )
        if 'Model' in shared.opts.bnb_quantization and 'transformer' not in kwargs:
            kwargs['transformer'] = diffusers.SD3Transformer2DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=cache_dir, quantization_config=bnb_config, torch_dtype=devices.dtype)
            shared.log.debug(f'Quantization: module=transformer type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
        if 'Text Encoder' in shared.opts.bnb_quantization and 'text_encoder_3' not in kwargs:
            kwargs['text_encoder_3'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_3", variant='fp16', cache_dir=cache_dir, quantization_config=bnb_config, torch_dtype=devices.dtype)
            shared.log.debug(f'Quantization: module=t5 type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
    return kwargs


def load_missing(kwargs, fn, cache_dir):
    keys = sd_models.get_safetensor_keys(fn)
    size = os.stat(fn).st_size // 1024 // 1024
    if size > 15000:
        repo_id = 'stabilityai/stable-diffusion-3.5-large'
    else:
        repo_id = 'stabilityai/stable-diffusion-3-medium'
    if 'text_encoder' not in kwargs and 'text_encoder' not in keys:
        kwargs['text_encoder'] = transformers.CLIPTextModelWithProjection.from_pretrained(repo_id, subfolder='text_encoder', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=te1 repo="{repo_id}"')
    if 'text_encoder_2' not in kwargs and 'text_encoder_2' not in keys:
        kwargs['text_encoder_2'] = transformers.CLIPTextModelWithProjection.from_pretrained(repo_id, subfolder='text_encoder_2', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=te2 repo="{repo_id}"')
    if 'text_encoder_3' not in kwargs and 'text_encoder_3' not in keys:
        kwargs['text_encoder_3'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder_3", variant='fp16', cache_dir=cache_dir, torch_dtype=devices.dtype)
        shared.log.debug(f'Load model: type=SD3 missing=te3 repo="{repo_id}"')
    # if 'transformer' not in kwargs and 'transformer' not in keys:
    #    kwargs['transformer'] = diffusers.SD3Transformer2DModel.from_pretrained(default_repo_id, subfolder="transformer", cache_dir=cache_dir, torch_dtype=devices.dtype)
    return kwargs


def load_sd3(checkpoint_info, cache_dir=None, config=None):
    repo_id = sd_models.path_to_repo(checkpoint_info.name)
    fn = checkpoint_info.path

    kwargs = {}
    kwargs = load_overrides(kwargs, cache_dir)
    kwargs = load_quants(kwargs, repo_id, cache_dir)

    if fn is not None and fn.endswith('.safetensors') and os.path.exists(fn):
        kwargs = load_missing(kwargs, fn, cache_dir)
        loader = diffusers.StableDiffusion3Pipeline.from_single_file
        repo_id = fn
    else:
        loader = diffusers.StableDiffusion3Pipeline.from_pretrained
        kwargs['variant'] = 'fp16'

    shared.log.debug(f'Load model: type=FLUX preloaded={list(kwargs)}')

    pipe = loader(
        repo_id,
        torch_dtype=devices.dtype,
        cache_dir=cache_dir,
        config=config,
        **kwargs,
    )
    devices.torch_gc()
    return pipe
