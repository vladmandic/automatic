import io
import json
import inspect
import os.path
from rich import progress # pylint: disable=redefined-builtin
import torch
import safetensors.torch

from modules import paths, shared, errors
from modules.sd_checkpoint import CheckpointInfo, select_checkpoint, list_models, checkpoints_list, checkpoint_titles, get_closet_checkpoint_match, model_hash, update_model_hashes, setup_model, write_metadata, read_metadata_from_safetensors # pylint: disable=unused-import
from modules.sd_offload import disable_offload, set_diffuser_offload, apply_balanced_offload, set_accelerate # pylint: disable=unused-import
from modules.sd_models_legacy import get_checkpoint_state_dict, load_model_weights, load_model, repair_config # pylint: disable=unused-import


class NoWatermark:
    def apply_watermark(self, img):
        return img


def get_signature(cls):
    signature = inspect.signature(cls.__init__, follow_wrapped=True, eval_str=True)
    return signature.parameters


def get_call(cls):
    if cls is None:
        return []
    signature = inspect.signature(cls.__call__, follow_wrapped=True, eval_str=True)
    return signature.parameters


def path_to_repo(fn: str = ''):
    if isinstance(fn, CheckpointInfo):
        fn = fn.name
    repo_id = fn.replace('\\', '/')
    if 'models--' in repo_id:
        repo_id = repo_id.split('models--')[-1]
        repo_id = repo_id.split('/')[0]
    repo_id = repo_id.split('/')
    repo_id = '/'.join(repo_id[-2:] if len(repo_id) > 1 else repo_id)
    repo_id = repo_id.replace('models--', '').replace('--', '/')
    return repo_id


def convert_to_faketensors(tensor):
    try:
        fake_module = torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True) # pylint: disable=protected-access
        if hasattr(tensor, "weight"):
            tensor.weight = torch.nn.Parameter(fake_module.from_tensor(tensor.weight))
        return tensor
    except Exception:
        pass
    return tensor


def read_state_dict(checkpoint_file, map_location=None, what:str='model'): # pylint: disable=unused-argument
    if not os.path.isfile(checkpoint_file):
        shared.log.error(f'Load dict: path="{checkpoint_file}" not a file')
        return None
    try:
        pl_sd = None
        with progress.open(checkpoint_file, 'rb', description=f'[cyan]Load {what}: [yellow]{checkpoint_file}', auto_refresh=True, console=shared.console) as f:
            _, extension = os.path.splitext(checkpoint_file)
            if extension.lower() == ".ckpt" and shared.opts.sd_disable_ckpt:
                shared.log.warning(f"Checkpoint loading disabled: {checkpoint_file}")
                return None
            if shared.opts.stream_load:
                if extension.lower() == ".safetensors":
                    # shared.log.debug('Model weights loading: type=safetensors mode=buffered')
                    buffer = f.read()
                    pl_sd = safetensors.torch.load(buffer)
                else:
                    # shared.log.debug('Model weights loading: type=checkpoint mode=buffered')
                    buffer = io.BytesIO(f.read())
                    pl_sd = torch.load(buffer, map_location='cpu')
            else:
                if extension.lower() == ".safetensors":
                    # shared.log.debug('Model weights loading: type=safetensors mode=mmap')
                    pl_sd = safetensors.torch.load_file(checkpoint_file, device='cpu')
                else:
                    # shared.log.debug('Model weights loading: type=checkpoint mode=direct')
                    pl_sd = torch.load(f, map_location='cpu')
            sd = get_state_dict_from_checkpoint(pl_sd)
        del pl_sd
    except Exception as e:
        errors.display(e, f'Load model: {checkpoint_file}')
        sd = None
    return sd


def get_state_dict_from_checkpoint(pl_sd):
    checkpoint_dict_replacements = {
        'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
        'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
        'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
    }

    def transform_checkpoint_dict_key(k):
        for text, replacement in checkpoint_dict_replacements.items():
            if k.startswith(text):
                k = replacement + k[len(text):]
        return k

    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)
    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)
        if new_key is not None:
            sd[new_key] = v
    pl_sd.clear()
    pl_sd.update(sd)
    return pl_sd


def patch_diffuser_config(sd_model, model_file):
    def load_config(fn, k):
        model_file = os.path.splitext(fn)[0]
        cfg_file = f'{model_file}_{k}.json'
        try:
            if os.path.exists(cfg_file):
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            cfg_file = f'{os.path.join(paths.sd_configs_path, os.path.basename(model_file))}_{k}.json'
            if os.path.exists(cfg_file):
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    if sd_model is None:
        return sd_model
    if hasattr(sd_model, 'unet') and hasattr(sd_model.unet, 'config') and 'inpaint' in model_file.lower():
        sd_model.unet.config.in_channels = 9
    if not hasattr(sd_model, '_internal_dict'):
        return sd_model
    for c in sd_model._internal_dict.keys(): # pylint: disable=protected-access
        component = getattr(sd_model, c, None)
        if hasattr(component, 'config'):
            override = load_config(model_file, c)
            updated = {}
            for k, v in override.items():
                if k.startswith('_'):
                    continue
                if v != component.config.get(k, None):
                    if hasattr(component.config, '__frozen'):
                        component.config.__frozen = False # pylint: disable=protected-access
                    component.config[k] = v
                    updated[k] = v
    return sd_model
