import os
import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules import shared, devices


debug = shared.log.trace if os.environ.get('SD_LOAD_DEBUG', None) is not None else lambda *args, **kwargs: None


def get_quant(file_path):
    if "qint8" in file_path.lower():
        return 'qint8'
    if "qint4" in file_path.lower():
        return 'qint4'
    if "fp8" in file_path.lower():
        return 'fp8'
    if "fp4" in file_path.lower():
        return 'fp4'
    if "nf4" in file_path.lower():
        return 'nf4'
    return 'none'


def load_flux_quanto(checkpoint_info, diffusers_load_config, transformer, text_encoder_2):
    from installer import install
    install('optimum-quanto', quiet=True)
    try:
        from optimum import quanto # pylint: disable=no-name-in-module
        from optimum.quanto import requantize # pylint: disable=no-name-in-module
    except Exception as e:
        shared.log.error(f"FLUX: Failed to import optimum-quanto: {e}")
        raise
    quanto.tensor.qbits.QBitsTensor.create = lambda *args, **kwargs: quanto.tensor.qbits.QBitsTensor(*args, **kwargs)

    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path

    if transformer is None:
        quantization_map = os.path.join(repo_path, "transformer", "quantization_map.json")
        if not os.path.exists(quantization_map):
            repo_id = checkpoint_info.name.replace('Diffusers/', '')
            quantization_map = hf_hub_download(repo_id, subfolder='transformer', filename='quantization_map.json', **diffusers_load_config)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        state_dict = load_file(os.path.join(repo_path, "transformer", "diffusion_pytorch_model.safetensors"))
        dtype = state_dict['context_embedder.bias'].dtype
        with torch.device("meta"):
            transformer = diffusers.FluxTransformer2DModel.from_config(os.path.join(repo_path, "transformer", "config.json")).to(dtype=dtype)
        requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
        transformer.eval()
        if transformer.dtype != devices.dtype:
            try:
                transformer = transformer.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"FLUX: Failed to cast transformer to {devices.dtype}, set dtype to {transformer.dtype}")
                raise

    if text_encoder_2 is None:
        quantization_map = os.path.join(repo_path, "text_encoder_2", "quantization_map.json")
        if not os.path.exists(quantization_map):
            repo_id = checkpoint_info.name.replace('Diffusers/', '')
            quantization_map = hf_hub_download(repo_id, subfolder='text_encoder_2', filename='quantization_map.json', **diffusers_load_config)
        with open(quantization_map, "r", encoding='utf8') as f:
            quantization_map = json.load(f)
        with open(os.path.join(repo_path, "text_encoder_2", "config.json"), encoding='utf8') as f:
            t5_config = transformers.T5Config(**json.load(f))
        state_dict = load_file(os.path.join(repo_path, "text_encoder_2", "model.safetensors"))
        dtype = state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'].dtype
        with torch.device("meta"):
            text_encoder_2 = transformers.T5EncoderModel(t5_config).to(dtype=dtype)
        requantize(text_encoder_2, state_dict, quantization_map, device=torch.device("cpu"))
        text_encoder_2.eval()
        if text_encoder_2.dtype != devices.dtype:
            try:
                text_encoder_2 = text_encoder_2.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"FLUX: Failed to cast text encoder to {devices.dtype}, set dtype to {text_encoder_2.dtype}")
                raise


def load_flux_bnb(checkpoint_info, diffusers_load_config, transformer, text_encoder_2): # pylint: disable=unused-argument
    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path
    from installer import install
    install('bitsandbytes', quiet=True)
    from diffusers import FluxTransformer2DModel
    quant = get_quant(repo_path)
    if quant == 'fp8':
        quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)
        if transformer is None:
            transformer = FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
    elif quant == 'fp4':
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
        if transformer is None:
            transformer = FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
    else:
        if transformer is None:
            transformer = FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config)


def load_transformer(file_path, transformer): # triggered by opts.sd_unet change
    quant = get_quant(file_path)
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "cache_dir": shared.opts.hfcache_dir,
    }
    shared.log.info(f'Loading UNet: type=FLUX file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant={quant} dtype={devices.dtype}')
    if 'nf4' in quant:
        from modules.model_flux_nf4 import load_flux_nf4
        load_flux_nf4(file_path, diffusers_load_config, transformer, text_encoder_2='skip')
    elif quant == 'qint8' or quant == 'qint4':
        load_flux_quanto(file_path, diffusers_load_config, transformer, text_encoder_2='skip')
    elif quant == 'fp8' or quant == 'fp4':
        load_flux_bnb(file_path, diffusers_load_config, transformer, text_encoder_2='skip')
    else:
        from diffusers import FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_single_file(file_path, **diffusers_load_config)
    if transformer is None:
        shared.log.error('Failed to load UNet model')
    return transformer


def load_flux(checkpoint_info, diffusers_load_config): # triggered by opts.sd_checkpoint change
    quant = get_quant(checkpoint_info.path)
    shared.log.debug(f'Loading FLUX: model="{checkpoint_info.name}" unet="{shared.opts.sd_unet}" t5="{shared.opts.sd_text_encoder}" vae="{shared.opts.sd_vae}" quant={quant} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')

    transformer = None
    text_encoder_2 = None
    vae = None

    # load overrides if any
    if shared.opts.sd_unet != 'None':
        debug(f'Loading FLUX: unet="{shared.opts.sd_unet}"')
        from modules import sd_unet
        load_transformer(sd_unet.unet_dict[shared.opts.sd_unet], transformer)
    if shared.opts.sd_text_encoder != 'None':
        debug(f'Loading FLUX: t5="{shared.opts.sd_text_encoder}"')
        from modules.model_t5 import load_t5
        text_encoder_2 = load_t5(t5=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
    if shared.opts.sd_vae != 'None':
        debug(f'Loading FLUX: vae="{shared.opts.sd_vae}"')
        from modules import sd_vae
        # vae = sd_vae.load_vae_diffusers(None, sd_vae.vae_dict[shared.opts.sd_vae], 'override')
        vae_file = sd_vae.vae_dict[shared.opts.sd_vae]
        if os.path.exists(vae_file):
            vae_config = os.path.join('configs', 'flux', 'vae', 'config.json')
            vae = diffusers.AutoencoderKL.from_single_file(vae_file, config=vae_config, **diffusers_load_config)

    # load quantized components if any
    if quant == 'nf4':
        from modules.model_flux_nf4 import load_flux_nf4
        load_flux_nf4(checkpoint_info, diffusers_load_config, transformer, text_encoder_2)
    if quant == 'qint8' or quant == 'qint4':
        load_flux_quanto(checkpoint_info, diffusers_load_config, transformer, text_encoder_2)

    # initialize pipeline with pre-loaded components
    components = {}
    if transformer is not None:
        components['transformer'] = transformer
    if text_encoder_2 is not None:
        components['text_encoder_2'] = text_encoder_2
    if vae is not None:
        components['vae'] = vae
    debug(f'Loading FLUX: preloaded={list(components)}')
    pipe = diffusers.FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev', cache_dir=shared.opts.diffusers_dir, **components, **diffusers_load_config)
    return pipe
