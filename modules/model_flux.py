import json
import torch
import diffusers
import transformers
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from modules import shared, devices



def load_quanto_transformer(repo_path):
    from optimum.quanto import requantize
    with open(repo_path + "/" + "transformer/quantization_map.json", "r") as f:
        quantization_map = json.load(f)
    state_dict = load_file(repo_path + "/" + "transformer/diffusion_pytorch_model.safetensors")
    dtype = state_dict['context_embedder.bias'].dtype
    with torch.device("meta"):
        transformer = diffusers.FluxTransformer2DModel.from_config(repo_path + "/" + "transformer/config.json").to(dtype=dtype)
    requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
    transformer.eval()
    return transformer


def load_quanto_text_encoder_2(repo_path):
    from optimum.quanto import requantize
    with open(repo_path + "/" + "text_encoder_2/quantization_map.json", "r") as f:
        quantization_map = json.load(f)
    with open(repo_path + "/" + "text_encoder_2/config.json") as f:
        t5_config = transformers.T5Config(**json.load(f))
    state_dict = load_file(repo_path + "/" + "text_encoder_2/model.safetensors")
    dtype = state_dict['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'].dtype
    with torch.device("meta"):
        text_encoder_2 = transformers.T5EncoderModel(t5_config).to(dtype=dtype)
    requantize(text_encoder_2, state_dict, quantization_map, device=torch.device("cpu"))
    text_encoder_2.eval()
    return text_encoder_2

def load_flux(checkpoint_info, diffusers_load_config):
    if "qint8" in checkpoint_info.name.lower() or "qint4" in checkpoint_info.name.lower():
        shared.log.debug(f'Loading FLUX: model="{checkpoint_info.name}" quant=True')
        from installer import install
        install('optimum-quanto', quiet=True)
        from optimum import quanto
        quanto.tensor.qbits.QBitsTensor.create = lambda *args, **kwargs: quanto.tensor.qbits.QBitsTensor(*args, **kwargs)
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, transformer=None, text_encoder_2=None, **diffusers_load_config)
        pipe.transformer = load_quanto_transformer(checkpoint_info.path)
        pipe.text_encoder_2 = load_quanto_text_encoder_2(checkpoint_info.path)
        if pipe.transformer.dtype != devices.dtype:
            try:
                pipe.transformer = pipe.transformer.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"FLUX: Failed to cast the transformer to {devices.dtype}! Set dtype to {pipe.transformer.dtype}")
                raise
        if pipe.text_encoder_2.dtype != devices.dtype:
            try:
                pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=devices.dtype)
            except Exception:
                shared.log.error(f"FLUX: Failed to cast the text encoder to {devices.dtype}! Set dtype to {pipe.text_encoder_2.dtype}")
                raise
    else:
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
        shared.log.debug(f'Loading FLUX: model="{checkpoint_info.name}" quant=False')
    if devices.dtype == torch.float16 and not shared.opts.no_half_vae:
        shared.log.warning("FLUX VAE doesn't support FP16! Enabling no-half-vae")
        shared.opts.no_half_vae = True
    return pipe
