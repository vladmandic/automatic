"""
Tiny AutoEncoder for Stable Diffusion
(DNN for encoding / decoding SD's latent space)

https://github.com/madebyollin/taesd
"""
import os
import threading
from PIL import Image
import torch
from modules import devices, paths


TAESD_MODELS = {
    'TAESD 1.3 Mocha Croissant': { 'fn': 'taesd_13_', 'uri': 'https://github.com/madebyollin/taesd/raw/7f572ca629c9b0d3c9f71140e5f501e09f9ea280', 'model': None },
    'TAESD 1.2 Chocolate-Dipped Shortbread': { 'fn': 'taesd_12_', 'uri': 'https://github.com/madebyollin/taesd/raw/8909b44e3befaa0efa79c5791e4fe1c4d4f7884e', 'model': None },
    'TAESD 1.1 Fruit Loops': { 'fn': 'taesd_11_', 'uri': 'https://github.com/madebyollin/taesd/raw/3e8a8a2ab4ad4079db60c1c7dc1379b4cc0c6b31', 'model': None },
    'TAESD 1.0': { 'fn': 'taesd_10_', 'uri': 'https://github.com/madebyollin/taesd/raw/88012e67cf0454e6d90f98911fe9d4aef62add86', 'model': None },
}
CQYAN_MODELS = {
    'Hybrid-Tiny SD': {
        'sd': { 'repo': 'cqyan/hybrid-sd-tinyvae', 'model': None },
        'sdxl': { 'repo': 'cqyan/hybrid-sd-tinyvae-xl', 'model': None },
    },
    'Hybrid-Small SD': {
        'sd': { 'repo': 'cqyan/hybrid-sd-small-vae', 'model': None },
        'sdxl': { 'repo': 'cqyan/hybrid-sd-small-vae-xl', 'model': None },
    },
}

prev_warnings = False
prev_cls = ''
prev_type = ''
prev_model = ''
lock = threading.Lock()


def warn_once(msg):
    from modules import shared
    global prev_warnings # pylint: disable=global-statement
    if not prev_warnings:
        prev_warnings = True
        shared.log.error(f'Decode: type="taesd" variant="{shared.opts.taesd_variant}": {msg}')
    return Image.new('RGB', (8, 8), color = (0, 0, 0))


def get_model(model_type = 'decoder'):
    global prev_cls, prev_type, prev_model # pylint: disable=global-statement
    from modules import shared
    cls = shared.sd_model_type
    if cls == 'ldm':
        cls = 'sd'
    folder = os.path.join(paths.models_path, "TAESD")
    os.makedirs(folder, exist_ok=True)
    if 'sd' not in cls and 'f1' not in cls:
        warn_once(f'cls={shared.sd_model.__class__.__name__} type={cls} unsuppported')
        return None
    if shared.opts.taesd_variant.startswith('TAESD'):
        cfg = TAESD_MODELS[shared.opts.taesd_variant]
        if (cls == prev_cls) and (model_type == prev_type) and (shared.opts.taesd_variant == prev_model) and (cfg['model'] is not None):
            return cfg['model']
        fn = os.path.join(folder, cfg['fn'] + cls + '_' + model_type + '.pth')
        if not os.path.exists(fn):
            uri = cfg['uri'] + '/tae' + cls + '_' + model_type + '.pth'
            try:
                shared.log.info(f'Decode: type="taesd" variant="{shared.opts.taesd_variant}": uri="{uri}" fn="{fn}" download')
                torch.hub.download_url_to_file(uri, fn)
            except Exception as e:
                warn_once(f'download uri={uri} {e}')
        if os.path.exists(fn):
            prev_cls = cls
            prev_type = model_type
            prev_model = shared.opts.taesd_variant
            shared.log.debug(f'Decode: type="taesd" variant="{shared.opts.taesd_variant}" fn="{fn}" load')
            from modules.taesd.taesd import TAESD
            TAESD_MODELS[shared.opts.taesd_variant]['model'] = TAESD(decoder_path=fn if model_type=='decoder' else None, encoder_path=fn if model_type=='encoder' else None)
            return TAESD_MODELS[shared.opts.taesd_variant]['model']
    elif shared.opts.taesd_variant.startswith('Hybrid'):
        cfg = CQYAN_MODELS[shared.opts.taesd_variant].get(cls, None)
        if (cls == prev_cls) and (model_type == prev_type) and (shared.opts.taesd_variant == prev_model) and (cfg['model'] is not None):
            return cfg['model']
        if cfg is None:
            warn_once(f'cls={shared.sd_model.__class__.__name__} type={cls} unsuppported')
            return None
        repo = cfg['repo']
        prev_cls = cls
        prev_type = model_type
        prev_model = shared.opts.taesd_variant
        shared.log.debug(f'Decode: type="taesd" variant="{shared.opts.taesd_variant}" id="{repo}" load')
        dtype = devices.dtype_vae if devices.dtype_vae != torch.bfloat16 else torch.float16 # taesd does not support bf16
        if 'tiny' in repo:
            from diffusers.models import AutoencoderTiny
            vae = AutoencoderTiny.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir, torch_dtype=dtype)
        else:
            from modules.taesd.hybrid_small import AutoencoderSmall
            vae = AutoencoderSmall.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir, torch_dtype=dtype)
        vae = vae.to(devices.device, dtype=dtype)
        CQYAN_MODELS[shared.opts.taesd_variant][cls]['model'] = vae
        return vae
    else:
        warn_once(f'cls={shared.sd_model.__class__.__name__} type={cls} unsuppported')
    return None


def decode(latents):
    with lock:
        from modules import shared
        vae = get_model(model_type='decoder')
        if vae is None or max(latents.shape) > 256: # safetey check of large tensors
            return latents
        try:
            with devices.inference_context():
                tensor = latents.unsqueeze(0) if len(latents.shape) == 3 else latents
                tensor = tensor.half().detach().clone().to(devices.device, dtype=vae.dtype)
                if shared.opts.taesd_variant.startswith('TAESD'):
                    image = vae.decoder(tensor).clamp(0, 1).detach()
                    return image[0]
                else:
                    image = vae.decode(tensor, return_dict=False)[0]
                    image = (image / 2.0 + 0.5).clamp(0, 1).detach()
                    return image
        except Exception as e:
            return warn_once(f'decode {e}')


def encode(image):
    with lock:
        vae = get_model(model_type='encoder')
        if vae is None:
            return image
        try:
            with devices.inference_context():
                latents = vae.encoder(image)
            return latents.detach()
        except Exception as e:
            return warn_once(f'encode {e}')
