import re
import io
import sys
import json
import time
import copy
import inspect
import logging
import contextlib
import collections
import os.path
from os import mkdir
from urllib import request
from enum import Enum
import diffusers
import diffusers.loaders.single_file_utils
from rich import progress # pylint: disable=redefined-builtin
import torch
import safetensors.torch
from omegaconf import OmegaConf
from transformers import logging as transformers_logging
from ldm.util import instantiate_from_config
from modules import paths, shared, shared_items, shared_state, modelloader, devices, script_callbacks, sd_vae, sd_unet, errors, hashes, sd_models_config, sd_models_compile, sd_hijack_accelerate
from modules.timer import Timer
from modules.memstats import memory_stats
from modules.modeldata import model_data


transformers_logging.set_verbosity_error()
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
checkpoints_list = {}
checkpoint_aliases = {}
checkpoints_loaded = collections.OrderedDict()
sd_metadata_file = os.path.join(paths.data_path, "metadata.json")
sd_metadata = None
sd_metadata_pending = 0
sd_metadata_timer = 0
debug_move = shared.log.trace if os.environ.get('SD_MOVE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_load = os.environ.get('SD_LOAD_DEBUG', None)
debug_process = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
diffusers_version = int(diffusers.__version__.split('.')[1])


class CheckpointInfo:
    def __init__(self, filename, sha=None):
        self.name = None
        self.hash = sha
        self.filename = filename
        self.type = ''
        relname = filename
        app_path = os.path.abspath(paths.script_path)

        def rel(fn, path):
            try:
                return os.path.relpath(fn, path)
            except Exception:
                return fn

        if relname.startswith('..'):
            relname = os.path.abspath(relname)
        if relname.startswith(shared.opts.ckpt_dir):
            relname = rel(filename, shared.opts.ckpt_dir)
        elif relname.startswith(shared.opts.diffusers_dir):
            relname = rel(filename, shared.opts.diffusers_dir)
        elif relname.startswith(model_path):
            relname = rel(filename, model_path)
        elif relname.startswith(paths.script_path):
            relname = rel(filename, paths.script_path)
        elif relname.startswith(app_path):
            relname = rel(filename, app_path)
        else:
            relname = os.path.abspath(relname)
        relname, ext = os.path.splitext(relname)
        ext = ext.lower()[1:]

        if os.path.isfile(filename): # ckpt or safetensor
            self.name = relname
            self.filename = filename
            self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{relname}")
            self.type = ext
        else: # maybe a diffuser
            if self.hash is None:
                repo = [r for r in modelloader.diffuser_repos if self.filename == r['name']]
            else:
                repo = [r for r in modelloader.diffuser_repos if self.hash == r['hash']]
            if len(repo) == 0:
                self.name = relname
                self.filename = filename
                self.sha256 = None
                self.type = 'unknown'
            else:
                self.name = os.path.join(os.path.basename(shared.opts.diffusers_dir), repo[0]['name'])
                self.filename = repo[0]['path']
                self.sha256 = repo[0]['hash']
                self.type = 'diffusers'

        self.shorthash = self.sha256[0:10] if self.sha256 else None
        self.title = self.name if self.shorthash is None else f'{self.name} [{self.shorthash}]'
        self.path = self.filename
        self.model_name = os.path.basename(self.name)
        self.metadata = read_metadata_from_safetensors(filename)
        # shared.log.debug(f'Checkpoint: type={self.type} name={self.name} filename={self.filename} hash={self.shorthash} title={self.title}')

    def register(self):
        checkpoints_list[self.title] = self
        for i in [self.name, self.filename, self.shorthash, self.title]:
            if i is not None:
                checkpoint_aliases[i] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return None
        self.shorthash = self.sha256[0:10]
        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()
        return self.shorthash


class NoWatermark:
    def apply_watermark(self, img):
        return img


def setup_model():
    list_models()
    sd_hijack_accelerate.hijack_hfhub()
    # sd_hijack_accelerate.hijack_torch_conv()
    if not shared.native:
        enable_midas_autodownload()


def checkpoint_tiles(use_short=False): # pylint: disable=unused-argument
    def convert(name):
        return int(name) if name.isdigit() else name.lower()
    def alphanumeric_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted([x.title for x in checkpoints_list.values()], key=alphanumeric_key)


def list_models():
    t0 = time.time()
    global checkpoints_list # pylint: disable=global-statement
    checkpoints_list.clear()
    checkpoint_aliases.clear()
    ext_filter = [".safetensors"] if shared.opts.sd_disable_ckpt or shared.native else [".ckpt", ".safetensors"]
    model_list = list(modelloader.load_models(model_path=model_path, model_url=None, command_path=shared.opts.ckpt_dir, ext_filter=ext_filter, download_name=None, ext_blacklist=[".vae.ckpt", ".vae.safetensors"]))
    for filename in sorted(model_list, key=str.lower):
        checkpoint_info = CheckpointInfo(filename)
        if checkpoint_info.name is not None:
            checkpoint_info.register()
    if shared.native:
        for repo in modelloader.load_diffusers_models(clear=True):
            checkpoint_info = CheckpointInfo(repo['name'], sha=repo['hash'])
            if checkpoint_info.name is not None:
                checkpoint_info.register()
    if shared.cmd_opts.ckpt is not None:
        if not os.path.exists(shared.cmd_opts.ckpt) and not shared.native:
            if shared.cmd_opts.ckpt.lower() != "none":
                shared.log.warning(f"Requested checkpoint not found: {shared.cmd_opts.ckpt}")
        else:
            checkpoint_info = CheckpointInfo(shared.cmd_opts.ckpt)
            if checkpoint_info.name is not None:
                checkpoint_info.register()
                shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif shared.cmd_opts.ckpt != shared.default_sd_model_file and shared.cmd_opts.ckpt is not None:
        shared.log.warning(f"Checkpoint not found: {shared.cmd_opts.ckpt}")
    shared.log.info(f'Available models: path="{shared.opts.ckpt_dir}" items={len(checkpoints_list)} time={time.time()-t0:.2f}')
    checkpoints_list = dict(sorted(checkpoints_list.items(), key=lambda cp: cp[1].filename))


def update_model_hashes():
    txt = []
    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.hash is None]
    # shared.log.info(f'Models list: short hash missing for {len(lst)} out of {len(checkpoints_list)} models')
    for ckpt in lst:
        ckpt.hash = model_hash(ckpt.filename)
        # txt.append(f'Calculated short hash: <b>{ckpt.title}</b> {ckpt.hash}')
    # txt.append(f'Updated short hashes for <b>{len(lst)}</b> out of <b>{len(checkpoints_list)}</b> models')
    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.sha256 is None or ckpt.shorthash is None]
    shared.log.info(f'Models list: hash missing={len(lst)} total={len(checkpoints_list)}')
    for ckpt in lst:
        ckpt.sha256 = hashes.sha256(ckpt.filename, f"checkpoint/{ckpt.name}")
        ckpt.shorthash = ckpt.sha256[0:10] if ckpt.sha256 is not None else None
        if ckpt.sha256 is not None:
            txt.append(f'Calculated full hash: <b>{ckpt.title}</b> {ckpt.shorthash}')
        else:
            txt.append(f'Skipped hash calculation: <b>{ckpt.title}</b>')
    txt.append(f'Updated hashes for <b>{len(lst)}</b> out of <b>{len(checkpoints_list)}</b> models')
    txt = '<br>'.join(txt)
    return txt


def get_closet_checkpoint_match(search_string):
    if search_string.startswith('huggingface/'):
        model_name = search_string.replace('huggingface/', '')
        checkpoint_info = CheckpointInfo(model_name) # create a virutal model info
        checkpoint_info.type = 'huggingface'
        return checkpoint_info
    checkpoint_info = checkpoint_aliases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info
    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found and len(found) > 0:
        return found[0]
    found = sorted([info for info in checkpoints_list.values() if search_string.split(' ')[0] in info.title], key=lambda x: len(x.title))
    if found and len(found) > 0:
        return found[0]
    for v in shared.reference_models.values():
        if search_string in v['path'] or os.path.basename(search_string) in v['path']:
            model_name = search_string.replace('huggingface/', '')
            checkpoint_info = CheckpointInfo(v['path']) # create a virutal model info
            checkpoint_info.type = 'huggingface'
            return checkpoint_info
    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""
    try:
        with open(filename, "rb") as file:
            import hashlib
            # t0 = time.time()
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            shorthash = m.hexdigest()[0:8]
            # t1 = time.time()
            # shared.log.debug(f'Calculating short hash: {filename} hash={shorthash} time={(t1-t0):.2f}')
            return shorthash
    except FileNotFoundError:
        return 'NOFILE'
    except Exception:
        return 'NOHASH'


def select_checkpoint(op='model'):
    if op == 'dict':
        model_checkpoint = shared.opts.sd_model_dict
    elif op == 'refiner':
        model_checkpoint = shared.opts.data.get('sd_model_refiner', None)
    else:
        model_checkpoint = shared.opts.sd_model_checkpoint
    if model_checkpoint is None or model_checkpoint == 'None':
        return None
    checkpoint_info = get_closet_checkpoint_match(model_checkpoint)
    if checkpoint_info is not None:
        shared.log.info(f'Select: {op}="{checkpoint_info.title if checkpoint_info is not None else None}"')
        return checkpoint_info
    if len(checkpoints_list) == 0:
        shared.log.warning("Cannot generate without a checkpoint")
        shared.log.info("Set system paths to use existing folders")
        shared.log.info("  or use --models-dir <path-to-folder> to specify base folder with all models")
        shared.log.info("  or use --ckpt-dir <path-to-folder> to specify folder with sd models")
        shared.log.info("  or use --ckpt <path-to-checkpoint> to force using specific model")
        return None
    # checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        if model_checkpoint != 'model.ckpt' and model_checkpoint != 'runwayml/stable-diffusion-v1-5':
            shared.log.warning(f'Selected: {op}="{model_checkpoint}" not found')
        else:
            shared.log.info("Selecting first available checkpoint")
        # shared.log.warning(f"Loading fallback checkpoint: {checkpoint_info.title}")
        # shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    else:
        shared.log.info(f'Select: {op}="{checkpoint_info.title if checkpoint_info is not None else None}"')
    return checkpoint_info


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


def get_state_dict_from_checkpoint(pl_sd):
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


def write_metadata():
    global sd_metadata_pending # pylint: disable=global-statement
    if sd_metadata_pending == 0:
        shared.log.debug(f'Model metadata: file="{sd_metadata_file}" no changes')
        return
    shared.writefile(sd_metadata, sd_metadata_file)
    shared.log.info(f'Model metadata saved: file="{sd_metadata_file}" items={sd_metadata_pending} time={sd_metadata_timer:.2f}')
    sd_metadata_pending = 0


def scrub_dict(dict_obj, keys):
    for key in list(dict_obj.keys()):
        if not isinstance(dict_obj, dict):
            continue
        if key in keys:
            dict_obj.pop(key, None)
        elif isinstance(dict_obj[key], dict):
            scrub_dict(dict_obj[key], keys)
        elif isinstance(dict_obj[key], list):
            for item in dict_obj[key]:
                scrub_dict(item, keys)


def read_metadata_from_safetensors(filename):
    global sd_metadata # pylint: disable=global-statement
    if sd_metadata is None:
        sd_metadata = shared.readfile(sd_metadata_file, lock=True) if os.path.isfile(sd_metadata_file) else {}
    res = sd_metadata.get(filename, None)
    if res is not None:
        return res
    if not filename.endswith(".safetensors"):
        return {}
    if shared.cmd_opts.no_metadata:
        return {}
    res = {}
    # try:
    t0 = time.time()
    with open(filename, mode="rb") as file:
        try:
            metadata_len = file.read(8)
            metadata_len = int.from_bytes(metadata_len, "little")
            json_start = file.read(2)
            if metadata_len <= 2 or json_start not in (b'{"', b"{'"):
                shared.log.error(f"Model metadata invalid: fn={filename}")
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                if v.startswith("data:"):
                    v = 'data'
                if k == 'format' and v == 'pt':
                    continue
                large = True if len(v) > 2048 else False
                if large and k == 'ss_datasets':
                    continue
                if large and k == 'workflow':
                    continue
                if large and k == 'prompt':
                    continue
                if large and k == 'ss_bucket_info':
                    continue
                if v[0:1] == '{':
                    try:
                        v = json.loads(v)
                        if large and k == 'ss_tag_frequency':
                            v = { i: len(j) for i, j in v.items() }
                        if large and k == 'sd_merge_models':
                            scrub_dict(v, ['sd_merge_recipe'])
                    except Exception:
                        pass
                res[k] = v
        except Exception as e:
            shared.log.error(f"Model metadata: fn={filename} {e}")
    sd_metadata[filename] = res
    global sd_metadata_pending # pylint: disable=global-statement
    sd_metadata_pending += 1
    t1 = time.time()
    global sd_metadata_timer # pylint: disable=global-statement
    sd_metadata_timer += (t1 - t0)
    # except Exception as e:
    #    shared.log.error(f"Error reading metadata from: {filename} {e}")
    return res


def read_state_dict(checkpoint_file, map_location=None): # pylint: disable=unused-argument
    if not os.path.isfile(checkpoint_file):
        shared.log.error(f"Model is not a file: {checkpoint_file}")
        return None
    try:
        pl_sd = None
        with progress.open(checkpoint_file, 'rb', description=f'[cyan]Loading model: [yellow]{checkpoint_file}', auto_refresh=True, console=shared.console) as f:
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


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    if not os.path.isfile(checkpoint_info.filename):
        return None
    if checkpoint_info in checkpoints_loaded:
        shared.log.info("Model weights loading: from cache")
        checkpoints_loaded.move_to_end(checkpoint_info, last=True)  # FIFO -> LRU cache
        return checkpoints_loaded[checkpoint_info]
    res = read_state_dict(checkpoint_info.filename)
    if shared.opts.sd_checkpoint_cache > 0 and not shared.native:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = res
        # clean up cache if limit is reached
        while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
            checkpoints_loaded.popitem(last=False)
    timer.record("load")
    return res


def load_model_weights(model: torch.nn.Module, checkpoint_info: CheckpointInfo, state_dict, timer):
    _pipeline, _model_type = detect_pipeline(checkpoint_info.path, 'model')
    shared.log.debug(f'Model weights loading: {memory_stats()}')
    timer.record("hash")
    if model_data.sd_dict == 'None':
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        shared.log.error(f'Error loading model weights: {checkpoint_info.filename}')
        shared.log.error(' '.join(str(e).splitlines()[:2]))
        return False
    del state_dict
    timer.record("apply")
    if shared.opts.opt_channelslast:
        model.to(memory_format=torch.channels_last)
        timer.record("channels")
    if not shared.opts.no_half:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)
        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        if shared.opts.no_half_vae:
            model.first_stage_model = None
        # with --upcast-sampling, don't convert the depth model weights to float16
        if shared.opts.upcast_sampling and depth_model:
            model.depth_model = None
        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model
    if shared.opts.cuda_cast_unet:
        devices.dtype_unet = model.model.diffusion_model.dtype
    else:
        model.model.diffusion_model.to(devices.dtype_unet)
    model.first_stage_model.to(devices.dtype_vae)
    model.sd_model_hash = checkpoint_info.calculate_shorthash()
    model.sd_model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    model.is_sdxl = False # a1111 compatibility item
    model.is_sd2 = hasattr(model.cond_stage_model, 'model') # a1111 compatibility item
    model.is_sd1 = not hasattr(model.cond_stage_model, 'model') # a1111 compatibility item
    model.logvar = model.logvar.to(devices.device) if hasattr(model, 'logvar') else None # fix for training
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
    sd_vae.delete_base_vae()
    sd_vae.clear_loaded_vae()
    vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
    sd_vae.load_vae(model, vae_file, vae_source)
    timer.record("vae")
    return True


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """
    import ldm.modules.midas.api
    midas_path = os.path.join(paths.models_path, 'midas')
    for k, v in ldm.modules.midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        ldm.modules.midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)
    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }
    ldm.modules.midas.api.load_model_inner = ldm.modules.midas.api.load_model

    def load_model_wrapper(model_type):
        path = ldm.modules.midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                mkdir(midas_path)
            shared.log.info(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            shared.log.info(f"{model_type} downloaded")
        return ldm.modules.midas.api.load_model_inner(model_type)

    ldm.modules.midas.api.load_model = load_model_wrapper


def repair_config(sd_config):
    if "use_ema" not in sd_config.model.params:
        sd_config.model.params.use_ema = False
    if shared.opts.no_half:
        sd_config.model.params.unet_config.params.use_fp16 = False
    elif shared.opts.upcast_sampling:
        sd_config.model.params.unet_config.params.use_fp16 = True if sys.platform != 'darwin' else False
    if getattr(sd_config.model.params.first_stage_config.params.ddconfig, "attn_type", None) == "vanilla-xformers" and not shared.xformers_available:
        sd_config.model.params.first_stage_config.params.ddconfig.attn_type = "vanilla"
    # For UnCLIP-L, override the hardcoded karlo directory
    if "noise_aug_config" in sd_config.model.params and "clip_stats_path" in sd_config.model.params.noise_aug_config.params:
        karlo_path = os.path.join(paths.models_path, 'karlo')
        sd_config.model.params.noise_aug_config.params.clip_stats_path = sd_config.model.params.noise_aug_config.params.clip_stats_path.replace("checkpoints/karlo_models", karlo_path)


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'


def change_backend():
    shared.log.info(f'Backend changed: from={shared.backend} to={shared.opts.sd_backend}')
    shared.log.warning('Full server restart required to apply all changes')
    unload_model_weights()
    shared.backend = shared.Backend.ORIGINAL if shared.opts.sd_backend == 'original' else shared.Backend.DIFFUSERS
    shared.native = shared.backend == shared.Backend.DIFFUSERS
    checkpoints_loaded.clear()
    from modules.sd_samplers import list_samplers
    list_samplers()
    list_models()
    from modules.sd_vae import refresh_vae_list
    refresh_vae_list()


def detect_pipeline(f: str, op: str = 'model', warning=True, quiet=False):
    guess = shared.opts.diffusers_pipeline
    warn = shared.log.warning if warning else lambda *args, **kwargs: None
    size = 0
    if guess == 'Autodetect':
        try:
            guess = 'Stable Diffusion XL' if 'XL' in f.upper() else 'Stable Diffusion'
            # guess by size
            if os.path.isfile(f) and f.endswith('.safetensors'):
                size = round(os.path.getsize(f) / 1024 / 1024)
                if size < 128:
                    warn(f'Model size smaller than expected: {f} size={size} MB')
                elif (size >= 316 and size <= 324) or (size >= 156 and size <= 164): # 320 or 160
                    warn(f'Model detected as VAE model, but attempting to load as model: {op}={f} size={size} MB')
                    guess = 'VAE'
                elif (size >= 4970 and size <= 4976): # 4973
                    guess = 'Stable Diffusion 2' # SD v2 but could be eps or v-prediction
                # elif size < 0: # unknown
                #    guess = 'Stable Diffusion 2B'
                elif (size >= 5791 and size <= 5799): # 5795
                    if op == 'model':
                        warn(f'Model detected as SD-XL refiner model, but attempting to load a base model: {op}={f} size={size} MB')
                    guess = 'Stable Diffusion XL Refiner'
                elif (size >= 6611 and size <= 7220): # 6617, HassakuXL is 6776, monkrenRealisticINT_v10 is 7217
                    guess = 'Stable Diffusion XL'
                elif (size >= 3361 and size <= 3369): # 3368
                    guess = 'Stable Diffusion Upscale'
                elif (size >= 4891 and size <= 4899): # 4897
                    guess = 'Stable Diffusion XL Inpaint'
                elif (size >= 9791 and size <= 9799): # 9794
                    guess = 'Stable Diffusion XL Instruct'
                elif (size > 3138 and size < 3142): #3140
                    guess = 'Stable Diffusion XL'
                elif (size > 5692 and size < 5698) or (size > 4134 and size < 4138) or (size > 10362 and size < 10366) or (size > 15028 and size < 15228):
                    guess = 'Stable Diffusion 3'
            # guess by name
            """
            if 'LCM_' in f.upper() or 'LCM-' in f.upper() or '_LCM' in f.upper() or '-LCM' in f.upper():
                if shared.backend == shared.Backend.ORIGINAL:
                    warn(f'Model detected as LCM model, but attempting to load using backend=original: {op}={f} size={size} MB')
                guess = 'Latent Consistency Model'
            """
            if 'instaflow' in f.lower():
                guess = 'InstaFlow'
            if 'segmoe' in f.lower():
                guess = 'SegMoE'
            if 'hunyuandit' in f.lower():
                guess = 'HunyuanDiT'
            if 'pixart-xl' in f.lower():
                guess = 'PixArt-Alpha'
            if 'stable-diffusion-3' in f.lower():
                guess = 'Stable Diffusion 3'
            if 'stable-cascade' in f.lower() or 'stablecascade' in f.lower() or 'wuerstchen3' in f.lower():
                if devices.dtype == torch.float16:
                    warn('Stable Cascade does not support Float16')
                guess = 'Stable Cascade'
            if 'pixart-sigma' in f.lower():
                guess = 'PixArt-Sigma'
            if 'lumina-next' in f.lower():
                guess = 'Lumina-Next'
            if 'kolors' in f.lower():
                guess = 'Kolors'
            if 'auraflow' in f.lower():
                guess = 'AuraFlow'
            # switch for specific variant
            if guess == 'Stable Diffusion' and 'inpaint' in f.lower():
                guess = 'Stable Diffusion Inpaint'
            elif guess == 'Stable Diffusion' and 'instruct' in f.lower():
                guess = 'Stable Diffusion Instruct'
            if guess == 'Stable Diffusion XL' and 'inpaint' in f.lower():
                guess = 'Stable Diffusion XL Inpaint'
            elif guess == 'Stable Diffusion XL' and 'instruct' in f.lower():
                guess = 'Stable Diffusion XL Instruct'
            # get actual pipeline
            pipeline = shared_items.get_pipelines().get(guess, None)
            if not quiet:
                shared.log.info(f'Autodetect: {op}="{guess}" class={pipeline.__name__} file="{f}" size={size}MB')
        except Exception as e:
            shared.log.error(f'Error detecting diffusers pipeline: model={f} {e}')
            return None, None
    else:
        try:
            size = round(os.path.getsize(f) / 1024 / 1024)
            pipeline = shared_items.get_pipelines().get(guess, None)
            if not quiet:
                shared.log.info(f'Diffusers: {op}="{guess}" class={pipeline.__name__} file="{f}" size={size}MB')
        except Exception as e:
            shared.log.error(f'Error loading diffusers pipeline: model={f} {e}')

    if pipeline is None:
        shared.log.warning(f'Autodetect: pipeline not recognized: {guess}: {op}={f} size={size}')
        pipeline = diffusers.StableDiffusionPipeline
    return pipeline, guess


def copy_diffuser_options(new_pipe, orig_pipe):
    new_pipe.sd_checkpoint_info = orig_pipe.sd_checkpoint_info
    new_pipe.sd_model_checkpoint = orig_pipe.sd_model_checkpoint
    new_pipe.embedding_db = getattr(orig_pipe, 'embedding_db', None)
    new_pipe.sd_model_hash = getattr(orig_pipe, 'sd_model_hash', None)
    new_pipe.has_accelerate = getattr(orig_pipe, 'has_accelerate', False)
    new_pipe.current_attn_name = getattr(orig_pipe, 'current_attn_name', None)
    new_pipe.default_scheduler = getattr(orig_pipe, 'default_scheduler', None)
    new_pipe.is_sdxl = getattr(orig_pipe, 'is_sdxl', False) # a1111 compatibility item
    new_pipe.is_sd2 = getattr(orig_pipe, 'is_sd2', False)
    new_pipe.is_sd1 = getattr(orig_pipe, 'is_sd1', True)


def set_diffuser_options(sd_model, vae = None, op: str = 'model', offload=True):
    if sd_model is None:
        shared.log.warning(f'{op} is not loaded')
        return

    if hasattr(sd_model, "watermark"):
        sd_model.watermark = NoWatermark()
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False
    if hasattr(sd_model, "vae"):
        if vae is not None:
            sd_model.vae = vae
            shared.log.debug(f'Setting {op} VAE: name={sd_vae.loaded_vae_file}')
        if shared.opts.diffusers_vae_upcast != 'default':
            sd_model.vae.config.force_upcast = True if shared.opts.diffusers_vae_upcast == 'true' else False
            if shared.opts.no_half_vae:
                devices.dtype_vae = torch.float32
                sd_model.vae.to(devices.dtype_vae)
            shared.log.debug(f'Setting {op} VAE: upcast={sd_model.vae.config.force_upcast}')
    if hasattr(sd_model, "enable_vae_slicing"):
        if shared.opts.diffusers_vae_slicing:
            shared.log.debug(f'Setting {op}: enable VAE slicing')
            sd_model.enable_vae_slicing()
        else:
            sd_model.disable_vae_slicing()
    if hasattr(sd_model, "enable_vae_tiling"):
        if shared.opts.diffusers_vae_tiling:
            shared.log.debug(f'Setting {op}: enable VAE tiling')
            sd_model.enable_vae_tiling()
        else:
            sd_model.disable_vae_tiling()
    if hasattr(sd_model, "vqvae"):
        sd_model.vqvae.to(torch.float32) # vqvae is producing nans in fp16

    set_diffusers_attention(sd_model)

    if shared.opts.diffusers_fuse_projections and hasattr(sd_model, 'fuse_qkv_projections'):
        try:
            sd_model.fuse_qkv_projections()
            shared.log.debug(f'Setting {op}: enable fused projections')
        except Exception as e:
            shared.log.error(f'Error enabling fused projections: {e}')
    if shared.opts.diffusers_eval:
        if hasattr(sd_model, "unet") and hasattr(sd_model.unet, "requires_grad_"):
            sd_model.unet.requires_grad_(False)
            sd_model.unet.eval()
        if hasattr(sd_model, "vae") and hasattr(sd_model.vae, "requires_grad_"):
            sd_model.vae.requires_grad_(False)
            sd_model.vae.eval()
        if hasattr(sd_model, "text_encoder") and hasattr(sd_model.text_encoder, "requires_grad_"):
            sd_model.text_encoder.requires_grad_(False)
            sd_model.text_encoder.eval()
    if shared.opts.diffusers_quantization:
        sd_model = sd_models_compile.dynamic_quantization(sd_model)

    if shared.opts.opt_channelslast and hasattr(sd_model, 'unet'):
        shared.log.debug(f'Setting {op}: enable channels last')
        sd_model.unet.to(memory_format=torch.channels_last)

    if offload:
        set_diffuser_offload(sd_model, op)

def set_diffuser_offload(sd_model, op: str = 'model'):
    if sd_model is None:
        shared.log.warning(f'{op} is not loaded')
        return
    if (shared.opts.diffusers_model_cpu_offload or shared.cmd_opts.medvram) and (shared.opts.diffusers_seq_cpu_offload or shared.cmd_opts.lowvram):
        shared.log.warning(f'Setting {op}: Model CPU offload and Sequential CPU offload are not compatible')
        shared.log.debug(f'Setting {op}: disabling model CPU offload')
        shared.opts.diffusers_model_cpu_offload=False
        shared.cmd_opts.medvram=False
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False
    if hasattr(sd_model, "enable_model_cpu_offload"):
        if shared.cmd_opts.medvram or shared.opts.diffusers_model_cpu_offload:
            shared.log.debug(f'Setting {op}: enable model CPU offload')
            if shared.opts.diffusers_move_base or shared.opts.diffusers_move_unet or shared.opts.diffusers_move_refiner:
                shared.opts.diffusers_move_base = False
                shared.opts.diffusers_move_unet = False
                shared.opts.diffusers_move_refiner = False
                shared.log.warning(f'Disabling {op} "Move model to CPU" since "Model CPU offload" is enabled')
            if not hasattr(sd_model, "_all_hooks") or len(sd_model._all_hooks) == 0: # pylint: disable=protected-access
                sd_model.enable_model_cpu_offload(device=devices.device)
            else:
                sd_model.maybe_free_model_hooks()
            sd_model.has_accelerate = True
    if hasattr(sd_model, "enable_sequential_cpu_offload"):
        if shared.cmd_opts.lowvram or shared.opts.diffusers_seq_cpu_offload:
            shared.log.debug(f'Setting {op}: enable sequential CPU offload')
            if shared.opts.diffusers_move_base or shared.opts.diffusers_move_unet or shared.opts.diffusers_move_refiner:
                shared.opts.diffusers_move_base = False
                shared.opts.diffusers_move_unet = False
                shared.opts.diffusers_move_refiner = False
                shared.log.warning(f'Disabling {op} "Move model to CPU" since "Sequential CPU offload" is enabled')
            if sd_model.has_accelerate:
                if op == "vae": # reapply sequential offload to vae
                    from accelerate import cpu_offload
                    sd_model.vae.to("cpu")
                    cpu_offload(sd_model.vae, devices.device, offload_buffers=len(sd_model.vae._parameters) > 0) # pylint: disable=protected-access
                else:
                    pass # do nothing if offload is already applied
            else:
                sd_model.enable_sequential_cpu_offload(device=devices.device)
            sd_model.has_accelerate = True


def move_model(model, device=None, force=False):
    if model is None or device is None:
        return
    if getattr(model, 'vae', None) is not None and get_diffusers_task(model) != DiffusersTaskType.TEXT_2_IMAGE:
        if device == devices.device and model.vae.device.type != "meta": # force vae back to gpu if not in txt2img mode
            model.vae.to(device)
            if hasattr(model.vae, '_hf_hook'):
                debug_move(f'Model move: to={device} class={model.vae.__class__} fn={sys._getframe(1).f_code.co_name}') # pylint: disable=protected-access
                model.vae._hf_hook.execution_device = device # pylint: disable=protected-access
    debug_move(f'Model move: device={device} class={model.__class__} accelerate={getattr(model, "has_accelerate", False)} fn={sys._getframe(1).f_code.co_name}') # pylint: disable=protected-access
    if hasattr(model, "components"): # accelerate patch
        for name, m in model.components.items():
            if not hasattr(m, "_hf_hook"): # not accelerate hook
                break
            if not isinstance(m, torch.nn.Module) or name in model._exclude_from_cpu_offload: # pylint: disable=protected-access
                continue
            for module in m.modules():
                if (hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "execution_device") and module._hf_hook.execution_device is not None): # pylint: disable=protected-access
                    try:
                        module._hf_hook.execution_device = device # pylint: disable=protected-access
                    except Exception as e:
                        if os.environ.get('SD_MOVE_DEBUG', None):
                            shared.log.error(f'Model move execution device: device={device} {e}')
    if getattr(model, 'has_accelerate', False) and not force:
        return
    try:
        try:
            model.to(device)
        except Exception as e0:
            if 'Cannot copy out of meta tensor' in str(e0):
                if hasattr(model, "components"):
                    for _name, component in model.components.items():
                        if hasattr(component, 'modules'):
                            for module in component.modules():
                                try:
                                    module.to(device)
                                except Exception as e2:
                                    if 'Cannot copy out of meta tensor' in str(e2):
                                        if os.environ.get('SD_MOVE_DEBUG', None):
                                            shared.log.warning(f'Model move meta: module={module.__class__}')
                                        module.to_empty(device=device)
            elif 'enable_sequential_cpu_offload' in str(e0):
                pass # ignore model move if sequential offload is enabled
            else:
                raise e0
        if hasattr(model, "prior_pipe"):
            model.prior_pipe.to(device)
    except Exception as e1:
        shared.log.error(f'Model move: device={device} {e1}')
    devices.torch_gc()


def get_load_config(model_file, model_type, config_type='yaml'):
    if config_type == 'yaml':
        yaml = os.path.splitext(model_file)[0] + '.yaml'
        if os.path.exists(yaml):
            return yaml
        if model_type == 'Stable Diffusion':
            return 'configs/v1-inference.yaml'
        if model_type == 'Stable Diffusion XL':
            return 'configs/sd_xl_base.yaml'
        if model_type == 'Stable Diffusion XL Refiner':
            return 'configs/sd_xl_refiner.yaml'
        if model_type == 'Stable Diffusion 2':
            return None # dont know if its eps or v so let diffusers sort it out
            # return 'configs/v2-inference-512-base.yaml'
            # return 'configs/v2-inference-768-v.yaml'
    elif config_type == 'json':
        if not shared.opts.diffuser_cache_config:
            return None
        if model_type == 'Stable Diffusion':
            return 'configs/sd15'
        if model_type == 'Stable Diffusion XL':
            return 'configs/sdxl'
        if model_type == 'Stable Diffusion 3':
            return 'configs/sd3'
    return None


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
        if debug_load:
            shared.log.debug('Model config patch: type=inpaint')
        sd_model.unet.config.in_channels = 9
    if not hasattr(sd_model, '_internal_dict'):
        return sd_model
    for c in sd_model._internal_dict.keys(): # pylint: disable=protected-access
        component = getattr(sd_model, c, None)
        if hasattr(component, 'config'):
            if debug_load:
                shared.log.debug(f'Model config: component={c} config={component.config}')
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
            if updated and debug_load:
                shared.log.debug(f'Model config: component={c} override={updated}')
    return sd_model


def load_diffuser(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model'): # pylint: disable=unused-argument
    if shared.cmd_opts.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
    if timer is None:
        timer = Timer()
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    timer.record("diffusers")
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "load_connected_pipeline": True,
        # sd15 specific but we cant know ahead of time
        "safety_checker": None,
        "requires_safety_checker": False,
        # "use_safetensors": True,
    }
    if shared.opts.diffusers_model_load_variant != 'default':
        diffusers_load_config['variant'] = shared.opts.diffusers_model_load_variant
    if shared.opts.diffusers_pipeline == 'Custom Diffusers Pipeline' and len(shared.opts.custom_diffusers_pipeline) > 0:
        shared.log.debug(f'Diffusers custom pipeline: {shared.opts.custom_diffusers_pipeline}')
        diffusers_load_config['custom_pipeline'] = shared.opts.custom_diffusers_pipeline
    # if 'LCM' in checkpoint_info.path:
        #    diffusers_load_config['custom_pipeline'] = 'latent_consistency_txt2img'
    if shared.opts.data.get('sd_model_checkpoint', '') == 'model.ckpt' or shared.opts.data.get('sd_model_checkpoint', '') == '':
        shared.opts.data['sd_model_checkpoint'] = "runwayml/stable-diffusion-v1-5"

    if op == 'model' or op == 'dict':
        if (model_data.sd_model is not None) and (checkpoint_info is not None) and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if (model_data.sd_refiner is not None) and (checkpoint_info is not None) and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return

    sd_model = None
    try:
        if shared.cmd_opts.ckpt is not None and os.path.isdir(shared.cmd_opts.ckpt) and model_data.initial: # initial load
            ckpt_basename = os.path.basename(shared.cmd_opts.ckpt)
            model_name = modelloader.find_diffuser(ckpt_basename)
            if model_name is not None:
                shared.log.info(f'Load model {op}: {model_name}')
                model_file = modelloader.download_diffusers_model(hub_id=model_name, variant=diffusers_load_config.get('variant', None))
                try:
                    shared.log.debug(f'Model load {op} config: {diffusers_load_config}')
                    sd_model = diffusers.DiffusionPipeline.from_pretrained(model_file, **diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Failed loading model: {model_file} {e}')
                    errors.display(e, f'Load model: {model_file}')
                list_models() # rescan for downloaded model
                checkpoint_info = CheckpointInfo(model_name)

        checkpoint_info = checkpoint_info or select_checkpoint(op=op)
        if checkpoint_info is None:
            unload_model_weights(op=op)
            return

        vae = None
        sd_vae.loaded_vae_file = None
        if op == 'model' or op == 'refiner':
            vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
            vae = sd_vae.load_vae_diffusers(checkpoint_info.path, vae_file, vae_source)
            if vae is not None:
                diffusers_load_config["vae"] = vae

        shared.log.debug(f'Diffusers loading: path="{checkpoint_info.path}"')
        pipeline, model_type = detect_pipeline(checkpoint_info.path, op)
        if os.path.isdir(checkpoint_info.path) or checkpoint_info.type == 'huggingface':
            files = shared.walk_files(checkpoint_info.path, ['.safetensors', '.bin', '.ckpt'])
            if 'variant' not in diffusers_load_config and any('diffusion_pytorch_model.fp16' in f for f in files): # deal with diffusers lack of variant fallback when loading
                diffusers_load_config['variant'] = 'fp16'
            if model_type in ['Stable Cascade']: # forced pipeline
                try:
                    from modules.model_stablecascade import load_cascade_combined
                    sd_model = load_cascade_combined(checkpoint_info, diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type in ['InstaFlow']: # forced pipeline
                try:
                    pipeline = diffusers.utils.get_class_from_dynamic_module('instaflow_one_step', module_file='pipeline.py')
                    sd_model = pipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type in ['SegMoE']: # forced pipeline
                try:
                    from modules.segmoe.segmoe_model import SegMoEPipeline
                    sd_model = SegMoEPipeline(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                    sd_model = sd_model.pipe # segmoe pipe does its stuff in __init__ and __call__ is the original pipeline
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type in ['PixArt-Sigma']: # forced pipeline
                try:
                    from modules.model_pixart import load_pixart
                    sd_model = load_pixart(checkpoint_info, diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type in ['Lumina-Next']: # forced pipeline
                try:
                    from modules.model_lumina import load_lumina
                    sd_model = load_lumina(checkpoint_info, diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type in ['Kolors']: # forced pipeline
                try:
                    from modules.model_kolors import load_kolors
                    sd_model = load_kolors(checkpoint_info, diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type in ['AuraFlow']: # forced pipeline
                try:
                    from modules.model_auraflow import load_auraflow
                    sd_model = load_auraflow(checkpoint_info, diffusers_load_config)
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type in ['Stable Diffusion 3']:
                try:
                    from modules.model_sd3 import load_sd3
                    shared.log.debug('Loading: model="Stable Diffusion 3" variant=medium type=diffusers')
                    shared.opts.scheduler = 'Default'
                    sd_model = load_sd3(cache_dir=shared.opts.diffusers_dir, config=diffusers_load_config.get('config', None))
                except Exception as e:
                    shared.log.error(f'Diffusers Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            elif model_type is not None and pipeline is not None and 'ONNX' in model_type: # forced pipeline
                try:
                    sd_model = pipeline.from_pretrained(checkpoint_info.path)
                except Exception as e:
                    shared.log.error(f'ONNX Failed loading {op}: {checkpoint_info.path} {e}')
                    if debug_load:
                        errors.display(e, 'Load')
                    return
            else:
                err1, err2, err3 = None, None, None
                # diffusers_load_config['use_safetensors'] = True
                if debug_load:
                    shared.log.debug(f'Diffusers load args: {diffusers_load_config}')
                try: # 1 - autopipeline, best choice but not all pipelines are available
                    try:
                        sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                        sd_model.model_type = sd_model.__class__.__name__
                    except ValueError as e:
                        if 'no variant default' in str(e):
                            shared.log.warning(f'Load: variant={diffusers_load_config["variant"]} model={checkpoint_info.path} using default variant')
                            diffusers_load_config.pop('variant', None)
                            sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                            sd_model.model_type = sd_model.__class__.__name__
                        else:
                            raise ValueError from e # reraise
                except Exception as e:
                    err1 = e
                    if debug_load:
                        errors.display(e, 'Load AutoPipeline')
                    # shared.log.error(f'AutoPipeline: {e}')
                try: # 2 - diffusion pipeline, works for most non-linked pipelines
                    if err1 is not None:
                        sd_model = diffusers.DiffusionPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                        sd_model.model_type = sd_model.__class__.__name__
                except Exception as e:
                    err2 = e
                    if debug_load:
                        errors.display(e, "Load DiffusionPipeline")
                    # shared.log.error(f'DiffusionPipeline: {e}')
                try: # 3 - try basic pipeline just in case
                    if err2 is not None:
                        sd_model = diffusers.StableDiffusionPipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                        sd_model.model_type = sd_model.__class__.__name__
                except Exception as e:
                    err3 = e  # ignore last error
                    shared.log.error(f"StableDiffusionPipeline: {e}")
                    if debug_load:
                        errors.display(e, "Load StableDiffusionPipeline")
                if err3 is not None:
                    shared.log.error(f'Failed loading {op}: {checkpoint_info.path} auto={err1} diffusion={err2}')
                    return
        elif os.path.isfile(checkpoint_info.path) and checkpoint_info.path.lower().endswith('.safetensors'):
            diffusers_load_config["local_files_only"] = diffusers_version < 28 # must be true for old diffusers, otherwise false but we override config for sd15/sdxl
            diffusers_load_config["extract_ema"] = shared.opts.diffusers_extract_ema
            if pipeline is None:
                shared.log.error(f'Diffusers {op} pipeline not initialized: {shared.opts.diffusers_pipeline}')
                return
            try:
                if model_type.startswith('Stable Diffusion'):
                    if shared.opts.diffusers_force_zeros:
                        diffusers_load_config['force_zeros_for_empty_prompt '] = shared.opts.diffusers_force_zeros
                    if diffusers_version < 28:
                        diffusers_load_config['original_config_file'] = get_load_config(checkpoint_info.path, model_type, config_type='yaml')
                    else:
                        diffusers_load_config['config'] = get_load_config(checkpoint_info.path, model_type, config_type='json')
                if model_type.startswith('Stable Diffusion 3'):
                    from modules.model_sd3 import load_sd3
                    sd_model = load_sd3(fn=checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, config=diffusers_load_config.get('config', None))
                elif hasattr(pipeline, 'from_single_file'):
                    diffusers.loaders.single_file_utils.CHECKPOINT_KEY_NAMES["clip"] = "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight" # patch for diffusers==0.28.0
                    diffusers_load_config['use_safetensors'] = True
                    diffusers_load_config['cache_dir'] = shared.opts.hfcache_dir # use hfcache instead of diffusers dir as this is for config only in case of single-file
                    if shared.opts.disable_accelerate:
                        from diffusers.utils import import_utils
                        import_utils._accelerate_available = False # pylint: disable=protected-access
                    if shared.opts.diffusers_to_gpu:
                        sd_hijack_accelerate.hijack_accelerate()
                    else:
                        sd_hijack_accelerate.restore_accelerate()
                    sd_model = pipeline.from_single_file(checkpoint_info.path, **diffusers_load_config)
                    # sd_model = patch_diffuser_config(sd_model, checkpoint_info.path)
                elif hasattr(pipeline, 'from_ckpt'):
                    diffusers_load_config['cache_dir'] = shared.opts.hfcache_dir
                    sd_model = pipeline.from_ckpt(checkpoint_info.path, **diffusers_load_config)
                else:
                    shared.log.error(f'Diffusers {op} cannot load safetensor model: {checkpoint_info.path} {shared.opts.diffusers_pipeline}')
                    return
                if debug_load:
                    shared.log.debug(f'Model args: {diffusers_load_config}')
                if sd_model is not None:
                    diffusers_load_config.pop('vae', None)
                    diffusers_load_config.pop('safety_checker', None)
                    diffusers_load_config.pop('requires_safety_checker', None)
                    diffusers_load_config.pop('config_files', None)
                    diffusers_load_config.pop('local_files_only', None)
                    shared.log.debug(f'Setting {op}: pipeline={sd_model.__class__.__name__} config={diffusers_load_config}') # pylint: disable=protected-access
            except Exception as e:
                shared.log.error(f'Diffusers failed loading: {op}={checkpoint_info.path} pipeline={shared.opts.diffusers_pipeline}/{sd_model.__class__.__name__} config={diffusers_load_config} {e}')
                errors.display(e, f'loading {op}={checkpoint_info.path} pipeline={shared.opts.diffusers_pipeline}/{sd_model.__class__.__name__}')
                return
        else:
            shared.log.error(f'Diffusers cannot load: {op}={checkpoint_info.path}')
            return

        if "StableDiffusion" in sd_model.__class__.__name__:
            pass # scheduler is created on first use
        elif "Kandinsky" in sd_model.__class__.__name__:
            sd_model.scheduler.name = 'DDIM'

        if sd_model is None:
            shared.log.error('Diffuser model not loaded')
            return
        sd_model.sd_model_hash = checkpoint_info.calculate_shorthash() # pylint: disable=attribute-defined-outside-init
        sd_model.sd_checkpoint_info = checkpoint_info # pylint: disable=attribute-defined-outside-init
        sd_model.sd_model_checkpoint = checkpoint_info.filename # pylint: disable=attribute-defined-outside-init
        sd_model.default_scheduler = copy.deepcopy(sd_model.scheduler) if hasattr(sd_model, "scheduler") else None
        sd_model.is_sdxl = False # a1111 compatibility item
        sd_model.is_sd2 = hasattr(sd_model, 'cond_stage_model') and hasattr(sd_model.cond_stage_model, 'model') # a1111 compatibility item
        sd_model.is_sd1 = not sd_model.is_sd2 # a1111 compatibility item
        sd_model.logvar = sd_model.logvar.to(devices.device) if hasattr(sd_model, 'logvar') else None # fix for training
        shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
        if hasattr(sd_model, "set_progress_bar_config"):
            sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining}', ncols=80, colour='#327fba')

        if "StableCascade" in sd_model.__class__.__name__: # detection can fail so we are applying post load here
            from modules.model_stablecascade import cascade_post_load
            cascade_post_load(sd_model)
        if model_type not in ['Stable Cascade']: # it will be handled in load_cascade if the detection works
            sd_unet.load_unet(sd_model)
        timer.record("load")

        if op == 'refiner':
            model_data.sd_refiner = sd_model
        else:
            model_data.sd_model = sd_model

        from modules.textual_inversion import textual_inversion
        sd_model.embedding_db = textual_inversion.EmbeddingDatabase()
        sd_model.embedding_db.add_embedding_dir(shared.opts.embeddings_dir)
        sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        timer.record("embeddings")

        from modules.prompt_parser_diffusers import insert_parser_highjack
        insert_parser_highjack(sd_model.__class__.__name__)

        set_diffuser_options(sd_model, vae, op, offload=False)
        if shared.opts.nncf_compress_weights and not (shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx"):
            sd_model = sd_models_compile.nncf_compress_weights(sd_model) # run this before move model so it can be compressed in CPU
        if shared.opts.optimum_quanto_weights:
            sd_model = sd_models_compile.optimum_quanto_weights(sd_model) # run this before move model so it can be compressed in CPU
        timer.record("options")

        set_diffuser_offload(sd_model, op)
        if op == 'model' and not (os.path.isdir(checkpoint_info.path) or checkpoint_info.type == 'huggingface'):
            sd_vae.apply_vae_config(shared.sd_model.sd_checkpoint_info.filename, vae_file, sd_model)
        if op == 'refiner' and shared.opts.diffusers_move_refiner:
            shared.log.debug('Moving refiner model to CPU')
            move_model(sd_model, devices.cpu)
        else:
            move_model(sd_model, devices.device)
        timer.record("move")

        reload_text_encoder(initial=True)

        if shared.opts.ipex_optimize:
            sd_model = sd_models_compile.ipex_optimize(sd_model)

        if (shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none'):
            sd_model = sd_models_compile.compile_diffusers(sd_model)
        timer.record("compile")

    except Exception as e:
        shared.log.error("Failed to load diffusers model")
        errors.display(e, "loading Diffusers model")

    devices.torch_gc(force=True)
    if shared.cmd_opts.profile:
        errors.profile(pr, 'Load')
    script_callbacks.model_loaded_callback(sd_model)
    shared.log.info(f"Load {op}: time={timer.summary()} native={get_native(sd_model)} {memory_stats()}")


class DiffusersTaskType(Enum):
    TEXT_2_IMAGE = 1
    IMAGE_2_IMAGE = 2
    INPAINTING = 3
    INSTRUCT = 4


def get_diffusers_task(pipe: diffusers.DiffusionPipeline) -> DiffusersTaskType:
    if pipe.__class__.__name__ in ["StableVideoDiffusionPipeline", "LEditsPPPipelineStableDiffusion", "LEditsPPPipelineStableDiffusionXL"]:
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif pipe.__class__.__name__ == "StableDiffusionXLInstructPix2PixPipeline":
        return DiffusersTaskType.INSTRUCT
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.values():
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING.values():
        return DiffusersTaskType.INPAINTING
    else:
        return DiffusersTaskType.TEXT_2_IMAGE


def switch_pipe(cls: diffusers.DiffusionPipeline, pipeline: diffusers.DiffusionPipeline = None, args = {}):
    """
    args:
    - cls: can be pipeline class or a string from custom pipelines
      for example: diffusers.StableDiffusionPipeline or 'mixture_tiling'
    - pipeline: source model to be used, if not provided currently loaded model is used
    - args: any additional components to load into the pipeline
      for example: { 'vae': None }
    """
    try:
        if isinstance(cls, str):
            shared.log.debug(f'Pipeline switch: custom={cls}')
            cls = diffusers.utils.get_class_from_dynamic_module(cls, module_file='pipeline.py')
        if pipeline is None:
            pipeline = shared.sd_model
        new_pipe = None
        signature = inspect.signature(cls.__init__, follow_wrapped=True, eval_str=True)
        possible = signature.parameters.keys()
        if isinstance(pipeline, cls) and args == {}:
            return pipeline
        pipe_dict = {}
        components_used = []
        components_skipped = []
        components_missing = []
        switch_mode = 'none'
        if hasattr(pipeline, '_internal_dict'):
            for item in pipeline._internal_dict.keys(): # pylint: disable=protected-access
                if item in possible:
                    pipe_dict[item] = getattr(pipeline, item, None)
                    components_used.append(item)
                else:
                    components_skipped.append(item)
            for item in possible:
                if item in ['self', 'args', 'kwargs']: # skip
                    continue
                if signature.parameters[item].default != inspect._empty: # has default value so we dont have to worry about it # pylint: disable=protected-access
                    continue
                if item not in components_used:
                    shared.log.warning(f'Pipeling switch: missing component={item} type={signature.parameters[item].annotation}')
                    pipe_dict[item] = None # try but not likely to work
                    components_missing.append(item)
            new_pipe = cls(**pipe_dict)
            switch_mode = 'auto'
        elif 'tokenizer_2' in possible and hasattr(pipeline, 'tokenizer_2'):
            new_pipe = cls(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
            )
            move_model(new_pipe, pipeline.device)
            switch_mode = 'sdxl'
        elif 'tokenizer' in possible and hasattr(pipeline, 'tokenizer'):
            new_pipe = cls(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                requires_safety_checker=False,
                safety_checker=None,
            )
            move_model(new_pipe, pipeline.device)
            switch_mode = 'sd'
        else:
            shared.log.error(f'Pipeline switch error: {pipeline.__class__.__name__} unrecognized')
            return pipeline
        if new_pipe is not None:
            for k, v in args.items():
                if k in possible:
                    setattr(new_pipe, k, v)
                    components_used.append(k)
                else:
                    shared.log.warning(f'Pipeline switch skipping unknown: component={k}')
                    components_skipped.append(k)
        if new_pipe is not None:
            copy_diffuser_options(new_pipe, pipeline)
            if hasattr(new_pipe, "watermark"):
                new_pipe.watermark = NoWatermark()
            if switch_mode == 'auto':
                shared.log.debug(f'Pipeline switch: from={pipeline.__class__.__name__} to={new_pipe.__class__.__name__} components={components_used} skipped={components_skipped} missing={components_missing}')
            else:
                shared.log.debug(f'Pipeline switch: from={pipeline.__class__.__name__} to={new_pipe.__class__.__name__} mode={switch_mode}')
            return new_pipe
        else:
            shared.log.error(f'Pipeline switch error: from={pipeline.__class__.__name__} to={cls.__name__} empty pipeline')
    except Exception as e:
        shared.log.error(f'Pipeline switch error: from={pipeline.__class__.__name__} to={cls.__name__} {e}')
        errors.display(e, 'Pipeline switch')
    return pipeline


def clean_diffuser_pipe(pipe):
    if pipe is not None and shared.sd_model_type == 'sdxl' and 'requires_aesthetics_score' in pipe.config and hasattr(pipe, '_internal_dict'):
        debug_process(f'Pipeline clean: {pipe.__class__.__name__}')
        # diffusers adds requires_aesthetics_score with img2img and complains if requires_aesthetics_score exist in txt2img
        internal_dict = dict(pipe._internal_dict) # pylint: disable=protected-access
        internal_dict.pop('requires_aesthetics_score', None)
        del pipe._internal_dict
        pipe.register_to_config(**internal_dict)


def set_diffuser_pipe(pipe, new_pipe_type):
    n = getattr(pipe.__class__, '__name__', '')
    if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
        clean_diffuser_pipe(pipe)

    if get_diffusers_task(pipe) == new_pipe_type:
        return pipe

    # skip specific pipelines
    if n in ['StableDiffusionReferencePipeline', 'StableDiffusionAdapterPipeline', 'AnimateDiffPipeline', 'AnimateDiffSDXLPipeline']:
        return pipe
    if 'Onnx' in pipe.__class__.__name__:
        return pipe

    if new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE or new_pipe_type == DiffusersTaskType.INPAINTING: # in some cases we want to reset the pipeline as they dont have their own variants
        if n == 'StableDiffusionPAGPipeline':
            pipe = switch_pipe(diffusers.StableDiffusionPipeline, pipe)
        if n == 'StableDiffusionXLPAGPipeline':
            pipe = switch_pipe(diffusers.StableDiffusionXLPipeline, pipe)

    sd_checkpoint_info = getattr(pipe, "sd_checkpoint_info", None)
    sd_model_checkpoint = getattr(pipe, "sd_model_checkpoint", None)
    embedding_db = getattr(pipe, "embedding_db", None)
    sd_model_hash = getattr(pipe, "sd_model_hash", None)
    has_accelerate = getattr(pipe, "has_accelerate", None)
    current_attn_name = getattr(pipe, "current_attn_name", None)
    default_scheduler = getattr(pipe, "default_scheduler", None)
    image_encoder = getattr(pipe, "image_encoder", None)
    feature_extractor = getattr(pipe, "feature_extractor", None)

    try:
        if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
            new_pipe = diffusers.AutoPipelineForText2Image.from_pipe(pipe)
        elif new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
            new_pipe = diffusers.AutoPipelineForImage2Image.from_pipe(pipe)
        elif new_pipe_type == DiffusersTaskType.INPAINTING:
            new_pipe = diffusers.AutoPipelineForInpainting.from_pipe(pipe)
        else:
            shared.log.error(f'Pipeline class change failed: type={new_pipe_type} pipeline={pipe.__class__.__name__}')
            return pipe
    except Exception as e: # pylint: disable=unused-variable
        shared.log.warning(f'Pipeline class change failed: type={new_pipe_type} pipeline={pipe.__class__.__name__} {e}')
        return pipe

    # if pipe.__class__ == new_pipe.__class__:
    #    return pipe
    new_pipe.sd_checkpoint_info = sd_checkpoint_info
    new_pipe.sd_model_checkpoint = sd_model_checkpoint
    new_pipe.embedding_db = embedding_db
    new_pipe.sd_model_hash = sd_model_hash
    new_pipe.has_accelerate = has_accelerate
    new_pipe.current_attn_name = current_attn_name
    new_pipe.default_scheduler = default_scheduler
    new_pipe.image_encoder = image_encoder
    new_pipe.feature_extractor = feature_extractor
    new_pipe.is_sdxl = getattr(pipe, 'is_sdxl', False) # a1111 compatibility item
    new_pipe.is_sd2 = getattr(pipe, 'is_sd2', False)
    new_pipe.is_sd1 = getattr(pipe, 'is_sd1', True)
    if hasattr(new_pipe, "watermark"):
        new_pipe.watermark = NoWatermark()
    shared.log.debug(f"Pipeline class change: original={pipe.__class__.__name__} target={new_pipe.__class__.__name__} device={pipe.device} fn={sys._getframe().f_back.f_code.co_name}") # pylint: disable=protected-access
    pipe = new_pipe
    return pipe


def set_diffusers_attention(pipe):
    import diffusers.models.attention_processor as p

    def set_attn(pipe, attention):
        if attention is None:
            return
        if not hasattr(pipe, "_get_signature_keys"):
            return
        module_names, _ = pipe._get_signature_keys(pipe) # pylint: disable=protected-access
        modules = [getattr(pipe, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, torch.nn.Module) and hasattr(m, "set_attn_processor")]
        for module in modules:
            if module.__class__.__name__ in ['SD3Transformer2DModel']:
                module.set_attn_processor(p.JointAttnProcessor2_0())
            elif module.__class__.__name__ in ['HunyuanDiT2DModel']:
                pass
            else:
                module.set_attn_processor(attention)

    if shared.opts.cross_attention_optimization == "Disabled":
        pass # do nothing
    elif shared.opts.cross_attention_optimization == "Scaled-Dot-Product": # The default set by Diffusers
        set_attn(pipe, p.AttnProcessor2_0())
    elif shared.opts.cross_attention_optimization == "xFormers" and hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        pipe.enable_xformers_memory_efficient_attention()
    elif shared.opts.cross_attention_optimization == "Split attention" and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    elif shared.opts.cross_attention_optimization == "Batch matrix-matrix":
        set_attn(pipe, p.AttnProcessor())
    elif shared.opts.cross_attention_optimization == "Dynamic Attention BMM":
        from modules.sd_hijack_dynamic_atten import DynamicAttnProcessorBMM
        set_attn(pipe, DynamicAttnProcessorBMM())
    elif shared.opts.cross_attention_optimization == "Dynamic Attention SDP":
        from modules.sd_hijack_dynamic_atten import DynamicAttnProcessorSDP
        set_attn(pipe, DynamicAttnProcessorSDP())

    pipe.current_attn_name = shared.opts.cross_attention_optimization


def get_native(pipe: diffusers.DiffusionPipeline):
    if hasattr(pipe, "vae") and hasattr(pipe.vae.config, "sample_size"):
        # Stable Diffusion
        size = pipe.vae.config.sample_size
    elif hasattr(pipe, "movq") and hasattr(pipe.movq.config, "sample_size"):
        # Kandinsky
        size = pipe.movq.config.sample_size
    elif hasattr(pipe, "unet") and hasattr(pipe.unet.config, "sample_size"):
        size = pipe.unet.config.sample_size
    else:
        size = 0
    return size


def load_model(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model'):
    from modules import lowvram, sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint(op=op)
    if checkpoint_info is None:
        return
    if op == 'model' or op == 'dict':
        if model_data.sd_model is not None and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if model_data.sd_refiner is not None and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return
    shared.log.debug(f'Load {op}: name={checkpoint_info.filename} dict={already_loaded_state_dict is not None}')
    if timer is None:
        timer = Timer()
    current_checkpoint_info = None
    if op == 'model' or op == 'dict':
        if model_data.sd_model is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            current_checkpoint_info = model_data.sd_model.sd_checkpoint_info
            unload_model_weights(op=op)
    else:
        if model_data.sd_refiner is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_refiner)
            current_checkpoint_info = model_data.sd_refiner.sd_checkpoint_info
            unload_model_weights(op=op)

    if not shared.native:
        from modules import sd_hijack_inpainting
        sd_hijack_inpainting.do_inpainting_hijack()

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    if state_dict is None or checkpoint_config is None:
        shared.log.error(f"Failed to load checkpooint: {checkpoint_info.filename}")
        if current_checkpoint_info is not None:
            shared.log.info(f"Restoring previous checkpoint: {current_checkpoint_info.filename}")
            load_model(current_checkpoint_info, None)
        return
    shared.log.debug(f'Model dict loaded: {memory_stats()}')
    sd_config = OmegaConf.load(checkpoint_config)
    repair_config(sd_config)
    timer.record("config")
    shared.log.debug(f'Model config loaded: {memory_stats()}')
    sd_model = None
    stdout = io.StringIO()
    if os.environ.get('SD_LDM_DEBUG', None) is not None:
        sd_model = instantiate_from_config(sd_config.model)
    else:
        with contextlib.redirect_stdout(stdout):
            """
            try:
                clip_is_included_into_sd = sd1_clip_weight in state_dict or sd2_clip_weight in state_dict
                with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd):
                    sd_model = instantiate_from_config(sd_config.model)
            except Exception as e:
                shared.log.error(f'LDM: instantiate from config: {e}')
                sd_model = instantiate_from_config(sd_config.model)
            """
            sd_model = instantiate_from_config(sd_config.model)
        for line in stdout.getvalue().splitlines():
            if len(line) > 0:
                shared.log.info(f'LDM: {line.strip()}')
    shared.log.debug(f"Model created from config: {checkpoint_config}")
    sd_model.used_config = checkpoint_config
    sd_model.has_accelerate = False
    timer.record("create")
    ok = load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    if not ok:
        model_data.sd_model = sd_model
        current_checkpoint_info = None
        unload_model_weights(op=op)
        shared.log.debug(f'Model weights unloaded: {memory_stats()} op={op}')
        if op == 'refiner':
            # shared.opts.data['sd_model_refiner'] = 'None'
            shared.opts.sd_model_refiner = 'None'
        return
    else:
        shared.log.debug(f'Model weights loaded: {memory_stats()}')
    timer.record("load")
    if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
        lowvram.setup_for_low_vram(sd_model, shared.cmd_opts.medvram)
    else:
        move_model(sd_model, devices.device)
    timer.record("move")
    shared.log.debug(f'Model weights moved: {memory_stats()}')
    sd_hijack.model_hijack.hijack(sd_model)
    timer.record("hijack")
    sd_model.eval()
    if op == 'refiner':
        model_data.sd_refiner = sd_model
    else:
        model_data.sd_model = sd_model
    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model
    timer.record("embeddings")
    script_callbacks.model_loaded_callback(sd_model)
    timer.record("callbacks")
    shared.log.info(f"Model loaded in {timer.summary()}")
    current_checkpoint_info = None
    devices.torch_gc(force=True)
    shared.log.info(f'Model load finished: {memory_stats()} cached={len(checkpoints_loaded.keys())}')


def reload_text_encoder(initial=False):
    if initial and (shared.opts.sd_text_encoder is None or shared.opts.sd_text_encoder == 'None'):
        return # dont unload
    signature = inspect.signature(shared.sd_model.__class__.__init__, follow_wrapped=True, eval_str=True).parameters
    t5 = [k for k, v in signature.items() if 'T5EncoderModel' in str(v)]
    if len(t5) > 0:
        from modules.model_t5 import set_t5
        shared.log.debug(f'Load: t5={shared.opts.sd_text_encoder} module="{t5[0]}"')
        set_t5(pipe=shared.sd_model, module=t5[0], t5=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
    elif hasattr(shared.sd_model, 'text_encoder_3'):
        from modules.model_t5 import set_t5
        shared.log.debug(f'Load: t5={shared.opts.sd_text_encoder} module="text_encoder_3"')
        set_t5(pipe=shared.sd_model, module='text_encoder_3', t5=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)


def reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model', force=False):
    load_dict = shared.opts.sd_model_dict != model_data.sd_dict
    from modules import lowvram, sd_hijack
    checkpoint_info = info or select_checkpoint(op=op) # are we selecting model or dictionary
    next_checkpoint_info = info or select_checkpoint(op='dict' if load_dict else 'model') if load_dict else None
    if checkpoint_info is None:
        unload_model_weights(op=op)
        return None
    orig_state = copy.deepcopy(shared.state)
    shared.state = shared_state.State()
    shared.state.begin('Load')
    if load_dict:
        shared.log.debug(f'Model dict: existing={sd_model is not None} target={checkpoint_info.filename} info={info}')
    else:
        model_data.sd_dict = 'None'
        shared.log.debug(f'Load model: existing={sd_model is not None} target={checkpoint_info.filename} info={info}')
    if sd_model is None:
        sd_model = model_data.sd_model if op == 'model' or op == 'dict' else model_data.sd_refiner
    if sd_model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = getattr(sd_model, 'sd_checkpoint_info', None)
        if current_checkpoint_info is not None and checkpoint_info is not None and current_checkpoint_info.filename == checkpoint_info.filename and not force:
            return None
        if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
            lowvram.send_everything_to_cpu()
        else:
            move_model(sd_model, devices.cpu)
        if (reuse_dict or shared.opts.model_reuse_dict) and not getattr(sd_model, 'has_accelerate', False):
            shared.log.info('Reusing previous model dictionary')
            sd_hijack.model_hijack.undo_hijack(sd_model)
        else:
            unload_model_weights(op=op)
            sd_model = None
    timer = Timer()
    # TODO implement caching after diffusers implement state_dict loading
    state_dict = get_checkpoint_state_dict(checkpoint_info, timer) if not shared.native else None
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    timer.record("config")
    if sd_model is None or checkpoint_config != getattr(sd_model, 'used_config', None):
        sd_model = None
        if not shared.native:
            load_model(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op)
            model_data.sd_dict = shared.opts.sd_model_dict
        else:
            load_diffuser(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op)
        if load_dict and next_checkpoint_info is not None:
            model_data.sd_dict = shared.opts.sd_model_dict
            shared.opts.data["sd_model_checkpoint"] = next_checkpoint_info.title
            reload_model_weights(reuse_dict=True) # ok we loaded dict now lets redo and load model on top of it
        shared.state.end()
        shared.state = orig_state
        # data['sd_model_checkpoint']
        if op == 'model' or op == 'dict':
            shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
            return model_data.sd_model
        else:
            shared.opts.data["sd_model_refiner"] = checkpoint_info.title
            return model_data.sd_refiner

    # fallback
    shared.log.info(f"Loading using fallback: {op} model={checkpoint_info.title}")
    try:
        load_model_weights(sd_model, checkpoint_info, state_dict, timer)
    except Exception:
        shared.log.error("Load model failed: restoring previous")
        load_model_weights(sd_model, current_checkpoint_info, None, timer)
    finally:
        sd_hijack.model_hijack.hijack(sd_model)
        timer.record("hijack")
        script_callbacks.model_loaded_callback(sd_model)
        timer.record("callbacks")
        if sd_model is not None and not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
            move_model(sd_model, devices.device)
            timer.record("device")
    shared.state.end()
    shared.state = orig_state
    shared.log.info(f"Load: {op} time={timer.summary()}")
    return sd_model


def convert_to_faketensors(tensor):
    try:
        fake_module = torch._subclasses.fake_tensor.FakeTensorMode(allow_non_fake_inputs=True) # pylint: disable=protected-access
        if hasattr(tensor, "weight"):
            tensor.weight = torch.nn.Parameter(fake_module.from_tensor(tensor.weight))
        return tensor
    except Exception:
        pass
    return tensor


def disable_offload(sd_model):
    from accelerate.hooks import remove_hook_from_module
    if not getattr(sd_model, 'has_accelerate', False):
        return
    for _name, model in sd_model.components.items():
        if not isinstance(model, torch.nn.Module):
            continue
        remove_hook_from_module(model, recurse=True)


def unload_model_weights(op='model'):
    if shared.compiled_model_state is not None:
        shared.compiled_model_state.compiled_cache.clear()
        shared.compiled_model_state.partitioned_modules.clear()
    if op == 'model' or op == 'dict':
        if model_data.sd_model:
            if not shared.native:
                from modules import sd_hijack
                move_model(model_data.sd_model, devices.cpu)
                sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            elif not (shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx"):
                disable_offload(model_data.sd_model)
                move_model(model_data.sd_model, 'meta')
            model_data.sd_model = None
            devices.torch_gc(force=True)
            shared.log.debug(f'Unload weights {op}: {memory_stats()}')
    elif op == 'refiner':
        if model_data.sd_refiner:
            if not shared.native:
                from modules import sd_hijack
                move_model(model_data.sd_refiner, devices.cpu)
                sd_hijack.model_hijack.undo_hijack(model_data.sd_refiner)
            else:
                disable_offload(model_data.sd_refiner)
                move_model(model_data.sd_refiner, 'meta')
            model_data.sd_refiner = None
            devices.torch_gc(force=True)
            shared.log.debug(f'Unload weights {op}: {memory_stats()}')


def apply_token_merging(sd_model):
    current_tome = getattr(sd_model, 'applied_tome', 0)
    current_todo = getattr(sd_model, 'applied_todo', 0)

    if shared.opts.token_merging_method == 'ToMe' and shared.opts.tome_ratio > 0:
        if current_tome == shared.opts.tome_ratio:
            return
        if shared.opts.hypertile_unet_enabled and not shared.cmd_opts.experimental:
            shared.log.warning('Token merging not supported with HyperTile for UNet')
            return
        try:
            import installer
            installer.install('tomesd', 'tomesd', ignore=False)
            import tomesd
            tomesd.apply_patch(
                sd_model,
                ratio=shared.opts.tome_ratio,
                use_rand=False, # can cause issues with some samplers
                merge_attn=True,
                merge_crossattn=False,
                merge_mlp=False
            )
            shared.log.info(f'Applying ToMe: ratio={shared.opts.tome_ratio}')
            sd_model.applied_tome = shared.opts.tome_ratio
        except Exception:
            shared.log.warning(f'Token merging not supported: pipeline={sd_model.__class__.__name__}')
    else:
        sd_model.applied_tome = 0

    if shared.opts.token_merging_method == 'ToDo' and shared.opts.todo_ratio > 0:
        if current_todo == shared.opts.todo_ratio:
            return
        if shared.opts.hypertile_unet_enabled and not shared.cmd_opts.experimental:
            shared.log.warning('Token merging not supported with HyperTile for UNet')
            return
        try:
            from modules.todo.todo_utils import patch_attention_proc
            token_merge_args = {
                        "ratio": shared.opts.todo_ratio,
                        "merge_tokens": "keys/values",
                        "merge_method": "downsample",
                        "downsample_method": "nearest",
                        "downsample_factor": 2,
                        "timestep_threshold_switch": 0.0,
                        "timestep_threshold_stop": 0.0,
                        "downsample_factor_level_2": 1,
                        "ratio_level_2": 0.0,
                        }
            patch_attention_proc(sd_model.unet, token_merge_args=token_merge_args)
            shared.log.info(f'Applying ToDo: ratio={shared.opts.todo_ratio}')
            sd_model.applied_todo = shared.opts.todo_ratio
        except Exception:
            shared.log.warning(f'Token merging not supported: pipeline={sd_model.__class__.__name__}')
    else:
        sd_model.applied_todo = 0


def remove_token_merging(sd_model):
    current_tome = getattr(sd_model, 'applied_tome', 0)
    current_todo = getattr(sd_model, 'applied_todo', 0)
    try:
        if current_tome > 0:
            import tomesd
            tomesd.remove_patch(sd_model)
            sd_model.applied_tome = 0
    except Exception:
        pass
    try:
        if current_todo > 0:
            from modules.todo.todo_utils import remove_patch
            remove_patch(sd_model)
            sd_model.applied_todo = 0
    except Exception:
        pass
