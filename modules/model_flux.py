import json
import torch
import diffusers
import transformers
from optimum.quanto import requantize
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from modules import devices, shared



def load_quanto_transformer(repo_path):
    with open(repo_path + "/" + "transformer/quantization_map.json", "r") as f:
        quantization_map = json.load(f)
    with torch.device("meta"):
        transformer = diffusers.FluxTransformer2DModel.from_config(repo_path + "/" + "transformer/config.json").to(torch.bfloat16)
    state_dict = load_file(repo_path + "/" + "transformer/diffusion_pytorch_model.safetensors")
    requantize(transformer, state_dict, quantization_map, device=torch.device("cpu"))
    return transformer


def load_quanto_text_encoder_2(repo_path):
    with open(repo_path + "/" + "text_encoder_2/quantization_map.json", "r") as f:
        quantization_map = json.load(f)
    with open(repo_path + "/" + "text_encoder_2/config.json") as f:
        t5_config = transformers.T5Config(**json.load(f))
    with torch.device("meta"):
        text_encoder_2 = transformers.T5EncoderModel(t5_config).to(torch.bfloat16)
    state_dict = load_file(repo_path + "/" + "text_encoder_2/model.safetensors")
    requantize(text_encoder_2, state_dict, quantization_map, device=torch.device("cpu"))
    return text_encoder_2

def load_flux(checkpoint_info, diffusers_load_config):
    if "qint8" in checkpoint_info.name.lower() or "qint4" in checkpoint_info.name.lower():
        shared.log.debug(f'Loading FLUX: model="{checkpoint_info.name}" quant=True')
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, transformer=None, text_encoder_2=None, **diffusers_load_config)
        pipe.transformer = load_quanto_transformer(checkpoint_info.path)
        pipe.text_encoder_2 = load_quanto_text_encoder_2(checkpoint_info.path)
    else:
        pipe = diffusers.FluxPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
        shared.log.debug(f'Loading FLUX: model="{checkpoint_info.name}" quant=False')
    return pipe
