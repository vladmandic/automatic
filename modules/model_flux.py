import os
import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from accelerate.utils import compute_module_sizes
from modules import shared, devices


debug = os.environ.get('SD_LOAD_DEBUG', None) is not None


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


def load_flux_quanto(checkpoint_info, diffusers_load_config, transformer_only=False):
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
    if transformer_only:
        return transformer, None

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
    return transformer, text_encoder_2


def load_flux_bnb(checkpoint_info, diffusers_load_config, transformer_only=False):
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
        transformer = FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
    elif quant == 'fp4':
        quantization_config = transformers.BitsAndBytesConfig(load_in_4bit=True)
        transformer = FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config, quantization_config=quantization_config)
    else:
        transformer = FluxTransformer2DModel.from_single_file(repo_path, **diffusers_load_config)
    if transformer_only:
        return transformer, None
    # TODO load text_encoder_2


def load_transformer(file_path): # triggered by opts.sd_unet change
    quant = get_quant(file_path)
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "cache_dir": shared.opts.hfcache_dir,
    }
    shared.log.info(f'Loading UNet: type=FLUX file="{file_path}" offload={shared.opts.diffusers_offload_mode} quant={quant} dtype={devices.dtype}')
    if 'nf4' in quant:
        from modules.model_flux_nf4 import load_flux_nf4
        transformer = load_flux_nf4(file_path, diffusers_load_config, transformer_only=True)
    elif quant == 'qint8' or quant == 'qint4':
        transformer, _ = load_flux_quanto(file_path, diffusers_load_config, transformer_only=True)
    elif quant == 'fp8' or quant == 'fp4':
        transformer, _ = load_flux_bnb(file_path, diffusers_load_config, transformer_only=True)
    else:
        from diffusers import FluxTransformer2DModel
        transformer = FluxTransformer2DModel.from_single_file(file_path, **diffusers_load_config)
    if transformer is None:
        shared.log.error('Failed to load UNet model')
    if debug:
        shared.log.debug(f'FLUX transformer: size={round(compute_module_sizes(transformer)[""] / 1024 / 1204)}')
    return transformer


def load_flux(checkpoint_info, diffusers_load_config): # triggered by opts.sd_checkpoint change
    quant = get_quant(checkpoint_info.path)
    shared.log.debug(f'Loading FLUX: model="{checkpoint_info.name}" quant={quant} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype}')
    if quant == 'nf4':
        from modules.model_flux_nf4 import load_flux_nf4
        pipe = load_flux_nf4(checkpoint_info, diffusers_load_config)
    elif quant == 'qint8' or quant == 'qint4':
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, transformer=None, text_encoder_2=None, **diffusers_load_config)
        pipe.transformer, pipe.text_encoder_2 = load_flux_quanto(checkpoint_info, diffusers_load_config)
    else:
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
    if debug:
        shared.log.debug(f'FLUX transformer: size={round(compute_module_sizes(pipe.transformer)[""] / 1024 / 1204)}')
    return pipe
