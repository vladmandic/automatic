import inspect
import diffusers
import transformers
import safetensors.torch
from modules import shared, devices, model_quant


def get_safetensor_keys(filename):
    keys = []
    try:
        with safetensors.torch.safe_open(filename, framework="pt", device="cpu") as f:
            keys = f.keys()
    except Exception as e:
        shared.log.error(f'Load dict: path="{filename}" {e}')
    return keys


def get_modules(model: callable):
    signature = inspect.signature(model.__init__, follow_wrapped=True)
    params = {param.name: param.annotation for param in signature.parameters.values() if param.annotation != inspect._empty and hasattr(param.annotation, 'from_pretrained')} # pylint: disable=protected-access
    for name, cls in params.items():
        shared.log.debug(f'Analyze: model={model} module={name} class={cls.__name__} loadable={getattr(cls, "from_pretrained", None)}')
    return params


def load_modules(repo_id: str, params: dict):
    cache_dir = shared.opts.hfcache_dir
    modules = {}
    for name, cls in params.items():
        subfolder = None
        kwargs = {}
        if cls == diffusers.AutoencoderKL:
            subfolder = 'vae'
        if cls == transformers.CLIPTextModel: # clip-vit-l
            subfolder = 'text_encoder'
        if cls == transformers.CLIPTextModelWithProjection: # clip-vit-g
            subfolder = 'text_encoder_2'
        if cls == transformers.T5EncoderModel: # t5-xxl
            subfolder = 'text_encoder_3'
            kwargs['quantization_config'] = model_quant.create_bnb_config()
            kwargs['variant'] = 'fp16'
        if cls == diffusers.SD3Transformer2DModel:
            subfolder = 'transformer'
            kwargs['quantization_config'] = model_quant.create_bnb_config()
        if subfolder is None:
            continue
        shared.log.debug(f'Load: module={name} class={cls.__name__} repo={repo_id} location={subfolder}')
        modules[name] = cls.from_pretrained(repo_id, subfolder=subfolder, cache_dir=cache_dir, torch_dtype=devices.dtype, **kwargs)
    return modules
