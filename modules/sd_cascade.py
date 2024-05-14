import os
from modules import shared, devices

def load_text_encoder(path):
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection
    from accelerate.utils.modeling import set_module_tensor_to_device
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    try:
        config = CLIPTextConfig(
            architectures=["CLIPTextModelWithProjection"],
            attention_dropout=0.0,
            bos_token_id=49406,
            dropout=0.0,
            eos_token_id=49407,
            hidden_act="gelu",
            hidden_size=1280,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=5120,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            model_type="clip_text_model",
            num_attention_heads=20,
            num_hidden_layers=32,
            pad_token_id=1,
            projection_dim=1280,
            vocab_size=49408
        )

        shared.log.info(f'Loading Text Encoder: name="{os.path.basename(os.path.splitext(path)[0])}" file="{path}"')

        with init_empty_weights():
            text_encoder = CLIPTextModelWithProjection(config)

        state_dict = load_file(path)

        for key in list(state_dict.keys()):
            set_module_tensor_to_device(text_encoder, key, devices.device, value=state_dict.pop(key), dtype=devices.dtype)

        return text_encoder

    except Exception as e:
        text_encoder = None
        shared.log.error(f'Failed to load Text Encoder model: {e}')
        return None


def load_prior(path, config_file="default"):
    from diffusers import StableCascadeUNet
    prior_text_encoder = None

    if config_file == "default":
        config_file = os.path.splitext(path)[0] + '.json'
    if not os.path.exists(config_file):
        if round(os.path.getsize(path) / 1024 / 1024 / 1024) < 5: # diffusers fails to find the configs from huggingface
            config_file = "configs/stable-cascade/prior_lite/config.json"
        else:
            config_file = "configs/stable-cascade/prior/config.json"

    shared.log.info(f'Loading UNet: name="{os.path.basename(os.path.splitext(path)[0])}" file="{path}" config="{config_file}"')
    prior_unet = StableCascadeUNet.from_single_file(path, config=config_file, torch_dtype=devices.dtype_unet, cache_dir=shared.opts.diffusers_dir)

    if os.path.isfile(os.path.splitext(path)[0] + "_text_encoder.safetensors"): # OneTrainer
        prior_text_encoder = load_text_encoder(os.path.splitext(path)[0] + "_text_encoder.safetensors")
    elif os.path.isfile(os.path.splitext(path)[0] + "_text_model.safetensors"): # KohyaSS
        prior_text_encoder = load_text_encoder(os.path.splitext(path)[0] + "_text_model.safetensors")

    return prior_unet, prior_text_encoder

