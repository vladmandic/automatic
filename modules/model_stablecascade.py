import os
import copy
import torch
from modules import shared, devices

def get_timestep_ratio_conditioning(t, alphas_cumprod):
    s = torch.tensor([0.008]) # diffusers uses 0.003 while the original is 0.008
    clamp_range = [0, 1]
    min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
    var = alphas_cumprod[t]
    var = var.clamp(*clamp_range)
    s, min_var = s.to(var.device), min_var.to(var.device)
    ratio = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
    return ratio

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
    from diffusers.models.unets import StableCascadeUNet
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


def load_cascade_combined(checkpoint_info, diffusers_load_config):
    from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline, StableCascadeCombinedPipeline
    from diffusers.models.unets import StableCascadeUNet
    from modules.sd_unet import unet_dict

    diffusers_load_config.pop("vae", None)
    if 'cascade' in checkpoint_info.name.lower():
        diffusers_load_config["variant"] = 'bf16'

    if shared.opts.sd_unet != "None" or 'stabilityai' in checkpoint_info.name.lower():
        if 'cascade' in checkpoint_info.name and ('lite' in checkpoint_info.name or (checkpoint_info.hash is not None and 'abc818bb0d' in checkpoint_info.hash)):
            decoder_folder = 'decoder_lite'
            prior_folder = 'prior_lite'
        else:
            decoder_folder = 'decoder'
            prior_folder = 'prior'
        if 'cascade' in checkpoint_info.name.lower():
            decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder=decoder_folder, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
            decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", cache_dir=shared.opts.diffusers_dir, decoder=decoder_unet, text_encoder=None, **diffusers_load_config)
        else:
            decoder = StableCascadeDecoderPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, text_encoder=None, **diffusers_load_config)
        shared.log.debug(f'StableCascade {decoder_folder}: scale={decoder.latent_dim_scale}')
        prior_text_encoder = None
        if shared.opts.sd_unet != "None":
            prior_unet, prior_text_encoder = load_prior(unet_dict[shared.opts.sd_unet])
        else:
            prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder=prior_folder, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
        if prior_text_encoder is not None:
            prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", cache_dir=shared.opts.diffusers_dir, prior=prior_unet, text_encoder=prior_text_encoder, image_encoder=None, feature_extractor=None, **diffusers_load_config)
        else:
            prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", cache_dir=shared.opts.diffusers_dir, prior=prior_unet, image_encoder=None, feature_extractor=None, **diffusers_load_config)
        shared.log.debug(f'StableCascade {prior_folder}: scale={prior.resolution_multiple}')
        sd_model = StableCascadeCombinedPipeline(
            tokenizer=decoder.tokenizer,
            text_encoder=None,
            decoder=decoder.decoder,
            scheduler=decoder.scheduler,
            vqgan=decoder.vqgan,
            prior_prior=prior.prior,
            prior_text_encoder=prior.text_encoder,
            prior_tokenizer=prior.tokenizer,
            prior_scheduler=prior.scheduler,
            prior_feature_extractor=None,
            prior_image_encoder=None)
    else:
        sd_model = StableCascadeCombinedPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)

    shared.log.debug(f'StableCascade combined: {sd_model.__class__.__name__}')

    return sd_model

def cascade_post_load(sd_model):
    sd_model.prior_pipe.scheduler.config.clip_sample = False
    sd_model.default_scheduler = copy.deepcopy(sd_model.prior_pipe.scheduler)
    sd_model.prior_pipe.get_timestep_ratio_conditioning = get_timestep_ratio_conditioning
    sd_model.decoder_pipe.text_encoder = sd_model.text_encoder = None  # Nothing uses the decoder's text encoder
    sd_model.prior_pipe.image_encoder = sd_model.prior_image_encoder = None # No img2img is implemented yet
    sd_model.prior_pipe.feature_extractor = sd_model.prior_feature_extractor = None # No img2img is implemented yet

    #de-dupe
    del sd_model.decoder_pipe.text_encoder
    del sd_model.prior_prior
    del sd_model.prior_text_encoder
    del sd_model.prior_tokenizer
    del sd_model.prior_scheduler
    del sd_model.prior_feature_extractor
    del sd_model.prior_image_encoder
    return sd_model
