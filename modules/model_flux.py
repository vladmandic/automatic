import os
import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from accelerate.utils import compute_module_sizes
from modules import shared, devices


def load_quanto_transformer(repo_path):
    from optimum.quanto import requantize # pylint: disable=no-name-in-module
    with open(repo_path + "/" + "transformer/quantization_map.json", "r", encoding='utf8') as f:
        quantization_map = json.load(f)
    state_dict = load_file(repo_path + "/" + "transformer/diffusion_pytorch_model.safetensors")
    dtype = state_dict['context_embedder.bias'].dtype
    with torch.device("meta"):
        transformer = diffusers.FluxTransformer2DModel.from_config(repo_path + "/" + "transformer/config.json").to(dtype=dtype)
    requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
    transformer.eval()
    return transformer


def load_quanto_text_encoder_2(repo_path):
    from optimum.quanto import requantize # pylint: disable=no-name-in-module
    with open(repo_path + "/" + "text_encoder_2/quantization_map.json", "r", encoding='utf8') as f:
        quantization_map = json.load(f)
    with open(repo_path + "/" + "text_encoder_2/config.json", encoding='utf8') as f:
        t5_config = transformers.T5Config(**json.load(f))
    state_dict = load_file(repo_path + "/" + "text_encoder_2/model.safetensors")
    dtype = state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'].dtype
    with torch.device("meta"):
        text_encoder_2 = transformers.T5EncoderModel(t5_config).to(dtype=dtype)
    requantize(text_encoder_2, state_dict, quantization_map, device=torch.device("cpu"))
    text_encoder_2.eval()
    return text_encoder_2


def load_flux(checkpoint_info, diffusers_load_config):
    if "qint8" in checkpoint_info.path.lower():
        quant = 'qint8'
    elif "qint4" in checkpoint_info.path.lower():
        quant = 'qint4'
    elif "nf4" in checkpoint_info.path.lower():
        quant = 'nf4'
    else:
        quant = None
    shared.log.debug(f'Loading FLUX: model="{checkpoint_info.name}" quant={quant}')
    if quant == 'nf4':
        from installer import install
        install('bitsandbytes', quiet=True)
        try:
            import bitsandbytes # pylint: disable=unused-import
        except Exception as e:
            shared.log.error(f"FLUX: Failed to import bitsandbytes: {e}")
            raise
        from modules.model_flux_nf4 import load_flux_nf4
        pipe = load_flux_nf4(checkpoint_info, diffusers_load_config)
    elif quant == 'qint8' or quant == 'qint4':
        from installer import install
        install('optimum-quanto', quiet=True)
        try:
            from optimum import quanto # pylint: disable=no-name-in-module
        except Exception as e:
            shared.log.error(f"FLUX: Failed to import optimum-quanto: {e}")
            raise
        quanto.tensor.qbits.QBitsTensor.create = lambda *args, **kwargs: quanto.tensor.qbits.QBitsTensor(*args, **kwargs)
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, transformer=None, text_encoder_2=None, **diffusers_load_config)
        pipe.transformer = load_quanto_transformer(checkpoint_info.path)
        pipe.text_encoder_2 = load_quanto_text_encoder_2(checkpoint_info.path)
        if pipe.transformer.dtype != devices.dtype:
            try:
                pipe.transformer = pipe.transformer.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"FLUX: Failed to cast transformer to {devices.dtype}, set dtype to {pipe.transformer.dtype}")
                raise
        if pipe.text_encoder_2.dtype != devices.dtype:
            try:
                pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"FLUX: Failed to cast text encoder to {devices.dtype}, set dtype to {pipe.text_encoder_2.dtype}")
                raise
    else:
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
    if devices.dtype == torch.float16 and not shared.opts.no_half_vae:
        shared.log.warning("FLUX: does not support FP16 VAE, enabling no-half-vae")
        shared.opts.no_half_vae = True
    shared.log.debug(f'FLUX computed size: {round(compute_module_sizes(pipe.transformer)[""] / 1024 / 1204)}')
    return pipe
