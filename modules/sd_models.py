import io
import sys
import time
import json
import copy
import inspect
import logging
import contextlib
import os.path
from enum import Enum
import diffusers
import diffusers.loaders.single_file_utils
from rich import progress # pylint: disable=redefined-builtin
import torch
import safetensors.torch
import accelerate
from omegaconf import OmegaConf
from modules import paths, shared, shared_state, modelloader, devices, script_callbacks, sd_vae, sd_unet, errors, sd_models_config, sd_models_compile, sd_hijack_accelerate, sd_detect
from modules.timer import Timer, process as process_timer
from modules.memstats import memory_stats
from modules.modeldata import model_data
from modules.sd_checkpoint import CheckpointInfo, select_checkpoint, list_models, checkpoints_list, checkpoint_titles, get_closet_checkpoint_match, model_hash, update_model_hashes, setup_model, write_metadata, read_metadata_from_safetensors # pylint: disable=unused-import


model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
sd_metadata_file = os.path.join(paths.data_path, "metadata.json")
sd_metadata = None
sd_metadata_pending = 0
sd_metadata_timer = 0
debug_move = shared.log.trace if os.environ.get('SD_MOVE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug_load = os.environ.get('SD_LOAD_DEBUG', None)
debug_process = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
diffusers_version = int(diffusers.__version__.split('.')[1])
checkpoint_tiles = checkpoint_titles # legacy compatibility
should_offload = ['sc', 'sd3', 'f1', 'hunyuandit', 'auraflow', 'omnigen']
offload_hook_instance = None


class NoWatermark:
    def apply_watermark(self, img):
        return img


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


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    if not os.path.isfile(checkpoint_info.filename):
        return None
    """
    if checkpoint_info in checkpoints_loaded:
        shared.log.info("Load model: cache")
        checkpoints_loaded.move_to_end(checkpoint_info, last=True)  # FIFO -> LRU cache
        return checkpoints_loaded[checkpoint_info]
    """
    res = read_state_dict(checkpoint_info.filename, what='model')
    """
    if shared.opts.sd_checkpoint_cache > 0 and not shared.native:
        # cache newly loaded model
        checkpoints_loaded[checkpoint_info] = res
        # clean up cache if limit is reached
        while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
            checkpoints_loaded.popitem(last=False)
    """
    timer.record("load")
    return res


def load_model_weights(model: torch.nn.Module, checkpoint_info: CheckpointInfo, state_dict, timer):
    _pipeline, _model_type = sd_detect.detect_pipeline(checkpoint_info.path, 'model')
    shared.log.debug(f'Load model: memory={memory_stats()}')
    timer.record("hash")
    if model_data.sd_dict == 'None':
        shared.opts.data["sd_model_checkpoint"] = checkpoint_info.title
    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        shared.log.error(f'Load model: path="{checkpoint_info.filename}"')
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
    from modules.sd_samplers import list_samplers
    list_samplers()
    list_models()
    from modules.sd_vae import refresh_vae_list
    refresh_vae_list()


def copy_diffuser_options(new_pipe, orig_pipe):
    new_pipe.sd_checkpoint_info = getattr(orig_pipe, 'sd_checkpoint_info', None)
    new_pipe.sd_model_checkpoint = getattr(orig_pipe, 'sd_model_checkpoint', None)
    new_pipe.embedding_db = getattr(orig_pipe, 'embedding_db', None)
    new_pipe.sd_model_hash = getattr(orig_pipe, 'sd_model_hash', None)
    new_pipe.has_accelerate = getattr(orig_pipe, 'has_accelerate', False)
    new_pipe.current_attn_name = getattr(orig_pipe, 'current_attn_name', None)
    new_pipe.default_scheduler = getattr(orig_pipe, 'default_scheduler', None)
    new_pipe.is_sdxl = getattr(orig_pipe, 'is_sdxl', False) # a1111 compatibility item
    new_pipe.is_sd2 = getattr(orig_pipe, 'is_sd2', False)
    new_pipe.is_sd1 = getattr(orig_pipe, 'is_sd1', True)
    add_noise_pred_to_diffusers_callback(new_pipe)
    if new_pipe.has_accelerate:
        set_accelerate(new_pipe)


def set_vae_options(sd_model, vae = None, op: str = 'model'):
    if hasattr(sd_model, "vae"):
        if vae is not None:
            sd_model.vae = vae
            shared.log.debug(f'Setting {op}: component=VAE name="{sd_vae.loaded_vae_file}"')
        if shared.opts.diffusers_vae_upcast != 'default':
            sd_model.vae.config.force_upcast = True if shared.opts.diffusers_vae_upcast == 'true' else False
            shared.log.debug(f'Setting {op}: component=VAE upcast={sd_model.vae.config.force_upcast}')
        if shared.opts.no_half_vae:
            devices.dtype_vae = torch.float32
            sd_model.vae.to(devices.dtype_vae)
            shared.log.debug(f'Setting {op}: component=VAE no-half=True')
    if hasattr(sd_model, "enable_vae_slicing"):
        if shared.opts.diffusers_vae_slicing:
            shared.log.debug(f'Setting {op}: component=VAE slicing=True')
            sd_model.enable_vae_slicing()
        else:
            sd_model.disable_vae_slicing()
    if hasattr(sd_model, "enable_vae_tiling"):
        if shared.opts.diffusers_vae_tiling:
            if hasattr(sd_model, 'vae') and hasattr(sd_model.vae, 'config') and hasattr(sd_model.vae.config, 'sample_size') and isinstance(sd_model.vae.config.sample_size, int):
                if shared.opts.diffusers_vae_tile_size > 0:
                    sd_model.vae.tile_sample_min_size = int(shared.opts.diffusers_vae_tile_size)
                    sd_model.vae.tile_latent_min_size = int(sd_model.vae.config.sample_size / (2 ** (len(sd_model.vae.config.block_out_channels) - 1)))
                if shared.opts.diffusers_vae_tile_overlap != 0.25:
                    sd_model.vae.tile_overlap_factor = float(shared.opts.diffusers_vae_tile_overlap)
                shared.log.debug(f'Setting {op}: component=VAE tiling=True tile={sd_model.vae.tile_sample_min_size} overlap={sd_model.vae.tile_overlap_factor}')
            else:
                shared.log.debug(f'Setting {op}: component=VAE tiling=True')
            sd_model.enable_vae_tiling()
        else:
            sd_model.disable_vae_tiling()
    if hasattr(sd_model, "vqvae"):
        shared.log.debug(f'Setting {op}: component=VQVAE upcast=True')
        sd_model.vqvae.to(torch.float32) # vqvae is producing nans in fp16


def set_diffuser_options(sd_model, vae = None, op: str = 'model', offload=True):
    if sd_model is None:
        shared.log.warning(f'{op} is not loaded')
        return

    if hasattr(sd_model, "watermark"):
        sd_model.watermark = NoWatermark()
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False

    clear_caches()
    set_vae_options(sd_model, vae, op)
    set_diffusers_attention(sd_model)

    if shared.opts.diffusers_fuse_projections and hasattr(sd_model, 'fuse_qkv_projections'):
        try:
            sd_model.fuse_qkv_projections()
            shared.log.debug(f'Setting {op}: fused-qkv=True')
        except Exception as e:
            shared.log.error(f'Setting {op}: fused-qkv=True {e}')
    if shared.opts.diffusers_fuse_projections and hasattr(sd_model, 'transformer') and hasattr(sd_model.transformer, 'fuse_qkv_projections'):
        try:
            sd_model.transformer.fuse_qkv_projections()
            shared.log.debug(f'Setting {op}: fused-qkv=True')
        except Exception as e:
            shared.log.error(f'Setting {op}: fused-qkv=True {e}')
    if shared.opts.diffusers_eval:
        def eval_model(model, op=None, sd_model=None): # pylint: disable=unused-argument
            if hasattr(model, "requires_grad_"):
                model.requires_grad_(False)
                model.eval()
            return model
        sd_model = sd_models_compile.apply_compile_to_model(sd_model, eval_model, ["Model", "VAE", "Text Encoder"], op="eval")
    if len(shared.opts.torchao_quantization) > 0 and shared.opts.torchao_quantization_mode == 'post':
        sd_model = sd_models_compile.torchao_quantization(sd_model)

    if shared.opts.opt_channelslast and hasattr(sd_model, 'unet'):
        shared.log.debug(f'Setting {op}: channels-last=True')
        sd_model.unet.to(memory_format=torch.channels_last)

    if offload:
        set_diffuser_offload(sd_model, op)


def set_accelerate_to_module(model):
    if hasattr(model, "pipe"):
        set_accelerate_to_module(model.pipe)
    if hasattr(model, "_internal_dict"):
        for k in model._internal_dict.keys(): # pylint: disable=protected-access
            component = getattr(model, k, None)
            if isinstance(component, torch.nn.Module):
                component.has_accelerate = True


def set_accelerate(sd_model):
    sd_model.has_accelerate = True
    set_accelerate_to_module(sd_model)
    if hasattr(sd_model, "prior_pipe"):
        set_accelerate_to_module(sd_model.prior_pipe)
    if hasattr(sd_model, "decoder_pipe"):
        set_accelerate_to_module(sd_model.decoder_pipe)


def set_diffuser_offload(sd_model, op: str = 'model'):
    t0 = time.time()
    if not shared.native:
        shared.log.warning('Attempting to use offload with backend=original')
        return
    if sd_model is None:
        shared.log.warning(f'{op} is not loaded')
        return
    if not (hasattr(sd_model, "has_accelerate") and sd_model.has_accelerate):
        sd_model.has_accelerate = False
    if shared.opts.diffusers_offload_mode == "none":
        if shared.sd_model_type in should_offload:
            shared.log.warning(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} type={shared.sd_model.__class__.__name__} large model')
        else:
            shared.log.debug(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
        if hasattr(sd_model, 'maybe_free_model_hooks'):
            sd_model.maybe_free_model_hooks()
            sd_model.has_accelerate = False
    if shared.opts.diffusers_offload_mode == "model" and hasattr(sd_model, "enable_model_cpu_offload"):
        try:
            shared.log.debug(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
            if shared.opts.diffusers_move_base or shared.opts.diffusers_move_unet or shared.opts.diffusers_move_refiner:
                shared.opts.diffusers_move_base = False
                shared.opts.diffusers_move_unet = False
                shared.opts.diffusers_move_refiner = False
                shared.log.warning(f'Disabling {op} "Move model to CPU" since "Model CPU offload" is enabled')
            if not hasattr(sd_model, "_all_hooks") or len(sd_model._all_hooks) == 0: # pylint: disable=protected-access
                sd_model.enable_model_cpu_offload(device=devices.device)
            else:
                sd_model.maybe_free_model_hooks()
            set_accelerate(sd_model)
        except Exception as e:
            shared.log.error(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} {e}')
    if shared.opts.diffusers_offload_mode == "sequential" and hasattr(sd_model, "enable_sequential_cpu_offload"):
        try:
            shared.log.debug(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} limit={shared.opts.cuda_mem_fraction}')
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
            set_accelerate(sd_model)
        except Exception as e:
            shared.log.error(f'Setting {op}: offload={shared.opts.diffusers_offload_mode} {e}')
    if shared.opts.diffusers_offload_mode == "balanced":
        sd_model = apply_balanced_offload(sd_model)
    process_timer.add('offload', time.time() - t0)


class OffloadHook(accelerate.hooks.ModelHook):
    def __init__(self, checkpoint_name):
        if shared.opts.diffusers_offload_max_gpu_memory > 1:
            shared.opts.diffusers_offload_max_gpu_memory = 0.75
        if shared.opts.diffusers_offload_max_cpu_memory > 1:
            shared.opts.diffusers_offload_max_cpu_memory = 0.75
        self.checkpoint_name = checkpoint_name
        self.min_watermark = shared.opts.diffusers_offload_min_gpu_memory
        self.max_watermark = shared.opts.diffusers_offload_max_gpu_memory
        self.cpu_watermark = shared.opts.diffusers_offload_max_cpu_memory
        self.gpu = int(shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory * 1024*1024*1024)
        self.cpu = int(shared.cpu_memory * shared.opts.diffusers_offload_max_cpu_memory * 1024*1024*1024)
        self.offload_map = {}
        self.param_map = {}
        gpu = f'{shared.gpu_memory * shared.opts.diffusers_offload_min_gpu_memory:.3f}-{shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory}:{shared.gpu_memory}'
        shared.log.info(f'Offload: type=balanced op=init watermark={self.min_watermark}-{self.max_watermark} gpu={gpu} cpu={shared.cpu_memory:.3f} limit={shared.opts.cuda_mem_fraction:.2f}')
        self.validate()
        super().__init__()

    def validate(self):
        if shared.opts.diffusers_offload_mode != 'balanced':
            return
        if shared.opts.diffusers_offload_min_gpu_memory < 0 or shared.opts.diffusers_offload_min_gpu_memory > 1:
            shared.opts.diffusers_offload_min_gpu_memory = 0.25
            shared.log.warning(f'Offload: type=balanced op=validate: watermark low={shared.opts.diffusers_offload_min_gpu_memory} invalid value')
        if shared.opts.diffusers_offload_max_gpu_memory < 0.1 or shared.opts.diffusers_offload_max_gpu_memory > 1:
            shared.opts.diffusers_offload_max_gpu_memory = 0.75
            shared.log.warning(f'Offload: type=balanced op=validate: watermark high={shared.opts.diffusers_offload_max_gpu_memory} invalid value')
        if shared.opts.diffusers_offload_min_gpu_memory > shared.opts.diffusers_offload_max_gpu_memory:
            shared.opts.diffusers_offload_min_gpu_memory = shared.opts.diffusers_offload_max_gpu_memory
            shared.log.warning(f'Offload: type=balanced op=validate: watermark low={shared.opts.diffusers_offload_min_gpu_memory} reset')
        if shared.opts.diffusers_offload_max_gpu_memory * shared.gpu_memory < 4:
            shared.log.warning(f'Offload: type=balanced op=validate: watermark high={shared.opts.diffusers_offload_max_gpu_memory} low memory')

    def model_size(self):
        return sum(self.offload_map.values())

    def init_hook(self, module):
        return module

    def pre_forward(self, module, *args, **kwargs):
        if devices.normalize_device(module.device) != devices.normalize_device(devices.device):
            device_index = torch.device(devices.device).index
            if device_index is None:
                device_index = 0
            max_memory = { device_index: self.gpu, "cpu": self.cpu }
            device_map = getattr(module, "balanced_offload_device_map", None)
            if device_map is None or max_memory != getattr(module, "balanced_offload_max_memory", None):
                device_map = accelerate.infer_auto_device_map(module, max_memory=max_memory)
            offload_dir = getattr(module, "offload_dir", os.path.join(shared.opts.accelerate_offload_path, module.__class__.__name__))
            module = accelerate.dispatch_model(module, device_map=device_map, offload_dir=offload_dir)
            module._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access
            module.balanced_offload_device_map = device_map
            module.balanced_offload_max_memory = max_memory
        return args, kwargs

    def post_forward(self, module, output):
        return output

    def detach_hook(self, module):
        return module


def apply_balanced_offload(sd_model, exclude=[]):
    global offload_hook_instance # pylint: disable=global-statement
    if shared.opts.diffusers_offload_mode != "balanced":
        return sd_model
    t0 = time.time()
    excluded = ['OmniGenPipeline']
    if sd_model.__class__.__name__ in excluded:
        return sd_model
    cached = True
    checkpoint_name = sd_model.sd_checkpoint_info.name if getattr(sd_model, "sd_checkpoint_info", None) is not None else None
    if checkpoint_name is None:
        checkpoint_name = sd_model.__class__.__name__
    if offload_hook_instance is None or offload_hook_instance.min_watermark != shared.opts.diffusers_offload_min_gpu_memory or offload_hook_instance.max_watermark != shared.opts.diffusers_offload_max_gpu_memory or checkpoint_name != offload_hook_instance.checkpoint_name:
        cached = False
        offload_hook_instance = OffloadHook(checkpoint_name)

    def get_pipe_modules(pipe):
        if hasattr(pipe, "_internal_dict"):
            modules_names = pipe._internal_dict.keys() # pylint: disable=protected-access
        else:
            modules_names = get_signature(pipe).keys()
        modules_names = [m for m in modules_names if m not in exclude and not m.startswith('_')]
        modules = {}
        for module_name in modules_names:
            module_size = offload_hook_instance.offload_map.get(module_name, None)
            if module_size is None:
                module = getattr(pipe, module_name, None)
                if not isinstance(module, torch.nn.Module):
                    continue
                try:
                    module_size = sum(p.numel() * p.element_size() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
                    param_num = sum(p.numel() for p in module.parameters(recurse=True)) / 1024 / 1024 / 1024
                except Exception as e:
                    shared.log.error(f'Offload: type=balanced op=calc module={module_name} {e}')
                    module_size = 0
                offload_hook_instance.offload_map[module_name] = module_size
                offload_hook_instance.param_map[module_name] = param_num
            modules[module_name] = module_size
        modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)
        return modules

    def apply_balanced_offload_to_module(pipe):
        used_gpu, used_ram = devices.torch_gc(fast=True)
        if hasattr(pipe, "pipe"):
            apply_balanced_offload_to_module(pipe.pipe)
        if hasattr(pipe, "_internal_dict"):
            keys = pipe._internal_dict.keys() # pylint: disable=protected-access
        else:
            keys = get_signature(pipe).keys()
        keys = [k for k in keys if k not in exclude and not k.startswith('_')]
        for module_name, module_size in get_pipe_modules(pipe): # pylint: disable=protected-access
            module = getattr(pipe, module_name, None)
            if module is None:
                continue
            network_layer_name = getattr(module, "network_layer_name", None)
            device_map = getattr(module, "balanced_offload_device_map", None)
            max_memory = getattr(module, "balanced_offload_max_memory", None)
            module = accelerate.hooks.remove_hook_from_module(module, recurse=True)
            perc_gpu = used_gpu / shared.gpu_memory
            try:
                prev_gpu = used_gpu
                do_offload = (perc_gpu > shared.opts.diffusers_offload_min_gpu_memory) and (module.device != devices.cpu)
                if do_offload:
                    module = module.to(devices.cpu, non_blocking=True)
                    used_gpu -= module_size
                if not cached:
                    shared.log.debug(f'Model module={module_name} type={module.__class__.__name__} dtype={module.dtype} quant={getattr(module, "quantization_method", None)} params={offload_hook_instance.param_map[module_name]:.3f} size={offload_hook_instance.offload_map[module_name]:.3f}')
                debug_move(f'Offload: type=balanced op={"move" if do_offload else "skip"} gpu={prev_gpu:.3f}:{used_gpu:.3f} perc={perc_gpu:.2f} ram={used_ram:.3f} current={module.device} dtype={module.dtype} quant={getattr(module, "quantization_method", None)} module={module.__class__.__name__} size={module_size:.3f}')
            except Exception as e:
                if 'out of memory' in str(e):
                    devices.torch_gc(fast=True, force=True, reason='oom')
                elif 'bitsandbytes' in str(e):
                    pass
                else:
                    shared.log.error(f'Offload: type=balanced op=apply module={module_name} {e}')
                if os.environ.get('SD_MOVE_DEBUG', None):
                    errors.display(e, f'Offload: type=balanced op=apply module={module_name}')
            module.offload_dir = os.path.join(shared.opts.accelerate_offload_path, checkpoint_name, module_name)
            module = accelerate.hooks.add_hook_to_module(module, offload_hook_instance, append=True)
            module._hf_hook.execution_device = torch.device(devices.device) # pylint: disable=protected-access
            if network_layer_name:
                module.network_layer_name = network_layer_name
            if device_map and max_memory:
                module.balanced_offload_device_map = device_map
                module.balanced_offload_max_memory = max_memory
        devices.torch_gc(fast=True, force=True, reason='offload')

    apply_balanced_offload_to_module(sd_model)
    if hasattr(sd_model, "pipe"):
        apply_balanced_offload_to_module(sd_model.pipe)
    if hasattr(sd_model, "prior_pipe"):
        apply_balanced_offload_to_module(sd_model.prior_pipe)
    if hasattr(sd_model, "decoder_pipe"):
        apply_balanced_offload_to_module(sd_model.decoder_pipe)
    set_accelerate(sd_model)
    t = time.time() - t0
    process_timer.add('offload', t)
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    debug_move(f'Apply offload: time={t:.2f} type=balanced fn={fn}')
    if not cached:
        shared.log.info(f'Model class={sd_model.__class__.__name__} modules={len(offload_hook_instance.offload_map)} size={offload_hook_instance.model_size():.3f}')
    return sd_model


def move_model(model, device=None, force=False):
    if model is None or device is None:
        return

    if not shared.native:
        if type(model).__name__ == 'LatentDiffusion':
            model = model.to(device)
            if hasattr(model, 'model'):
                model.model = model.model.to(device)
            if hasattr(model, 'first_stage_model'):
                model.first_stage_model = model.first_stage_model.to(device)
            if hasattr(model, 'cond_stage_model'):
                model.cond_stage_model = model.cond_stage_model.to(device)
        devices.torch_gc()
        return

    if hasattr(model, 'pipe'):
        move_model(model.pipe, device, force)

    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    if getattr(model, 'vae', None) is not None and get_diffusers_task(model) != DiffusersTaskType.TEXT_2_IMAGE:
        if device == devices.device and model.vae.device.type != "meta": # force vae back to gpu if not in txt2img mode
            model.vae.to(device)
            if hasattr(model.vae, '_hf_hook'):
                debug_move(f'Model move: to={device} class={model.vae.__class__} fn={fn}') # pylint: disable=protected-access
                model.vae._hf_hook.execution_device = device # pylint: disable=protected-access
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
    if hasattr(model, "device") and devices.normalize_device(model.device) == devices.normalize_device(device) and not force:
        return
    try:
        t0 = time.time()
        try:
            if hasattr(model, 'to'):
                model.to(device)
            if hasattr(model, "prior_pipe"):
                model.prior_pipe.to(device)
        except Exception as e0:
            if 'Cannot copy out of meta tensor' in str(e0) or 'must be Tensor, not NoneType' in str(e0):
                if hasattr(model, "components"):
                    for _name, component in model.components.items():
                        if hasattr(component, 'modules'):
                            for module in component.modules():
                                try:
                                    if hasattr(module, 'to'):
                                        module.to(device)
                                except Exception as e2:
                                    if 'Cannot copy out of meta tensor' in str(e2):
                                        if os.environ.get('SD_MOVE_DEBUG', None):
                                            shared.log.warning(f'Model move meta: module={module.__class__}')
                                        module.to_empty(device=device)
            elif 'enable_sequential_cpu_offload' in str(e0):
                pass # ignore model move if sequential offload is enabled
            elif 'Params4bit' in str(e0) or 'Params8bit' in str(e0):
                pass # ignore model move if quantization is enabled
            else:
                raise e0
        t1 = time.time()
    except Exception as e1:
        t1 = time.time()
        shared.log.error(f'Model move: device={device} {e1}')
    if 'move' not in process_timer.records:
        process_timer.records['move'] = 0
    process_timer.records['move'] += t1 - t0
    if os.environ.get('SD_MOVE_DEBUG', None) or (t1-t0) > 2:
        shared.log.debug(f'Model move: device={device} class={model.__class__.__name__} accelerate={getattr(model, "has_accelerate", False)} fn={fn} time={t1-t0:.2f}') # pylint: disable=protected-access
    devices.torch_gc()


def move_base(model, device):
    if hasattr(model, 'transformer'):
        key = 'transformer'
    elif hasattr(model, 'unet'):
        key = 'unet'
    else:
        shared.log.warning(f'Model move: model={model.__class__} device={device} key=unknown')
        return None
    shared.log.debug(f'Model move: module={key} device={device}')
    model = getattr(model, key)
    R = model.device
    move_model(model, device)
    return R


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


def load_diffuser_initial(diffusers_load_config, op='model'):
    sd_model = None
    checkpoint_info = None
    ckpt_basename = os.path.basename(shared.cmd_opts.ckpt)
    model_name = modelloader.find_diffuser(ckpt_basename)
    if model_name is not None:
        shared.log.info(f'Load model {op}: path="{model_name}"')
        model_file = modelloader.download_diffusers_model(hub_id=model_name, variant=diffusers_load_config.get('variant', None))
        try:
            shared.log.debug(f'Load {op}: config={diffusers_load_config}')
            sd_model = diffusers.DiffusionPipeline.from_pretrained(model_file, **diffusers_load_config)
        except Exception as e:
            shared.log.error(f'Failed loading model: {model_file} {e}')
            errors.display(e, f'Load {op}: path="{model_file}"')
            return None, None
        list_models() # rescan for downloaded model
        checkpoint_info = CheckpointInfo(model_name)
    return sd_model, checkpoint_info


def load_diffuser_force(model_type, checkpoint_info, diffusers_load_config, op='model'):
    sd_model = None
    try:
        if model_type in ['Stable Cascade']: # forced pipeline
            from modules.model_stablecascade import load_cascade_combined
            sd_model = load_cascade_combined(checkpoint_info, diffusers_load_config)
        elif model_type in ['InstaFlow']: # forced pipeline
            pipeline = diffusers.utils.get_class_from_dynamic_module('instaflow_one_step', module_file='pipeline.py')
            sd_model = pipeline.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
        elif model_type in ['SegMoE']: # forced pipeline
            from modules.segmoe.segmoe_model import SegMoEPipeline
            sd_model = SegMoEPipeline(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
            sd_model = sd_model.pipe # segmoe pipe does its stuff in __init__ and __call__ is the original pipeline
        elif model_type in ['PixArt-Sigma']: # forced pipeline
            from modules.model_pixart import load_pixart
            sd_model = load_pixart(checkpoint_info, diffusers_load_config)
        elif model_type in ['Sana']: # forced pipeline
            from modules.model_sana import load_sana
            sd_model = load_sana(checkpoint_info, diffusers_load_config)
        elif model_type in ['Lumina-Next']: # forced pipeline
            from modules.model_lumina import load_lumina
            sd_model = load_lumina(checkpoint_info, diffusers_load_config)
        elif model_type in ['Kolors']: # forced pipeline
            from modules.model_kolors import load_kolors
            sd_model = load_kolors(checkpoint_info, diffusers_load_config)
        elif model_type in ['AuraFlow']: # forced pipeline
            from modules.model_auraflow import load_auraflow
            sd_model = load_auraflow(checkpoint_info, diffusers_load_config)
        elif model_type in ['FLUX']:
            from modules.model_flux import load_flux
            sd_model = load_flux(checkpoint_info, diffusers_load_config)
        elif model_type in ['Stable Diffusion 3']:
            from modules.model_sd3 import load_sd3
            shared.log.debug(f'Load {op}: model="Stable Diffusion 3"')
            shared.opts.scheduler = 'Default'
            sd_model = load_sd3(checkpoint_info, cache_dir=shared.opts.diffusers_dir, config=diffusers_load_config.get('config', None))
        elif model_type in ['Meissonic']: # forced pipeline
            from modules.model_meissonic import load_meissonic
            sd_model = load_meissonic(checkpoint_info, diffusers_load_config)
        elif model_type in ['OmniGen']: # forced pipeline
            from modules.model_omnigen import load_omnigen
            sd_model = load_omnigen(checkpoint_info, diffusers_load_config)
    except Exception as e:
        shared.log.error(f'Load {op}: path="{checkpoint_info.path}" {e}')
        if debug_load:
            errors.display(e, 'Load')
        return None
    return sd_model


def load_diffuser_folder(model_type, pipeline, checkpoint_info, diffusers_load_config, op='model'):
    sd_model = None
    files = shared.walk_files(checkpoint_info.path, ['.safetensors', '.bin', '.ckpt'])
    if 'variant' not in diffusers_load_config and any('diffusion_pytorch_model.fp16' in f for f in files): # deal with diffusers lack of variant fallback when loading
        diffusers_load_config['variant'] = 'fp16'
    if model_type is not None and pipeline is not None and 'ONNX' in model_type: # forced pipeline
        try:
            sd_model = pipeline.from_pretrained(checkpoint_info.path)
        except Exception as e:
            shared.log.error(f'Load {op}: type=ONNX path="{checkpoint_info.path}" {e}')
            if debug_load:
                errors.display(e, 'Load')
            return None
    else:
        err1, err2, err3 = None, None, None
        if os.path.exists(checkpoint_info.path) and os.path.isdir(checkpoint_info.path):
            if os.path.exists(os.path.join(checkpoint_info.path, 'unet', 'diffusion_pytorch_model.bin')):
                shared.log.debug(f'Load {op}: type=pickle')
                diffusers_load_config['use_safetensors'] = False
        if debug_load:
            shared.log.debug(f'Load {op}: args={diffusers_load_config}')
        try: # 1 - autopipeline, best choice but not all pipelines are available
            try:
                sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                sd_model.model_type = sd_model.__class__.__name__
            except ValueError as e:
                if 'no variant default' in str(e):
                    shared.log.warning(f'Load {op}: variant={diffusers_load_config["variant"]} model="{checkpoint_info.path}" using default variant')
                    diffusers_load_config.pop('variant', None)
                    sd_model = diffusers.AutoPipelineForText2Image.from_pretrained(checkpoint_info.path, cache_dir=shared.opts.diffusers_dir, **diffusers_load_config)
                    sd_model.model_type = sd_model.__class__.__name__
                elif 'safetensors found in directory' in str(err1):
                    shared.log.warning(f'Load {op}: type=pickle')
                    diffusers_load_config['use_safetensors'] = False
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
            shared.log.error(f'Load {op}: {checkpoint_info.path} auto={err1} diffusion={err2}')
            return None
    return sd_model


def load_diffuser_file(model_type, pipeline, checkpoint_info, diffusers_load_config, op='model'):
    sd_model = None
    diffusers_load_config["local_files_only"] = diffusers_version < 28 # must be true for old diffusers, otherwise false but we override config for sd15/sdxl
    diffusers_load_config["extract_ema"] = shared.opts.diffusers_extract_ema
    if pipeline is None:
        shared.log.error(f'Load {op}: pipeline={shared.opts.diffusers_pipeline} not initialized')
        return None
    try:
        if model_type.startswith('Stable Diffusion'):
            if shared.opts.diffusers_force_zeros:
                diffusers_load_config['force_zeros_for_empty_prompt '] = shared.opts.diffusers_force_zeros
            else:
                model_config = sd_detect.get_load_config(checkpoint_info.path, model_type, config_type='json')
                if model_config is not None:
                    if debug_load:
                        shared.log.debug(f'Load {op}: config="{model_config}"')
                    diffusers_load_config['config'] = model_config
        if model_type.startswith('Stable Diffusion 3'):
            from modules.model_sd3 import load_sd3
            sd_model = load_sd3(checkpoint_info=checkpoint_info, cache_dir=shared.opts.diffusers_dir, config=diffusers_load_config.get('config', None))
        elif hasattr(pipeline, 'from_single_file'):
            diffusers.loaders.single_file_utils.CHECKPOINT_KEY_NAMES["clip"] = "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight" # patch for diffusers==0.28.0
            diffusers_load_config['use_safetensors'] = True
            diffusers_load_config['cache_dir'] = shared.opts.hfcache_dir # use hfcache instead of diffusers dir as this is for config only in case of single-file
            if shared.opts.disable_accelerate:
                from diffusers.utils import import_utils
                import_utils._accelerate_available = False # pylint: disable=protected-access
            if shared.opts.diffusers_to_gpu and model_type.startswith('Stable Diffusion'):
                shared.log.debug(f'Diffusers accelerate: direct={shared.opts.diffusers_to_gpu}')
                sd_hijack_accelerate.hijack_accelerate()
            else:
                sd_hijack_accelerate.restore_accelerate()
            sd_model = pipeline.from_single_file(checkpoint_info.path, **diffusers_load_config)
            # sd_model = patch_diffuser_config(sd_model, checkpoint_info.path)
        elif hasattr(pipeline, 'from_ckpt'):
            diffusers_load_config['cache_dir'] = shared.opts.hfcache_dir
            sd_model = pipeline.from_ckpt(checkpoint_info.path, **diffusers_load_config)
        else:
            shared.log.error(f'Load {op}: file="{checkpoint_info.path}" {shared.opts.diffusers_pipeline} cannot load safetensor model')
            return None
        if shared.opts.diffusers_vae_upcast != 'default' and model_type in ['Stable Diffusion', 'Stable Diffusion XL']:
            diffusers_load_config['force_upcast'] = True if shared.opts.diffusers_vae_upcast == 'true' else False
        # if debug_load:
        #    shared.log.debug(f'Model args: {diffusers_load_config}')
        if sd_model is not None:
            diffusers_load_config.pop('vae', None)
            diffusers_load_config.pop('safety_checker', None)
            diffusers_load_config.pop('requires_safety_checker', None)
            diffusers_load_config.pop('config_files', None)
            diffusers_load_config.pop('local_files_only', None)
            shared.log.debug(f'Setting {op}: pipeline={sd_model.__class__.__name__} config={diffusers_load_config}') # pylint: disable=protected-access
    except Exception as e:
        shared.log.error(f'Load {op}: file="{checkpoint_info.path}" pipeline={shared.opts.diffusers_pipeline}/{sd_model.__class__.__name__} config={diffusers_load_config} {e}')
        if 'Weights for this component appear to be missing in the checkpoint' in str(e):
            shared.log.error(f'Load {op}: file="{checkpoint_info.path}" is not a complete model')
        else:
            errors.display(e, 'Load')
        return None
    return sd_model


def load_diffuser(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model', revision=None): # pylint: disable=unused-argument
    if timer is None:
        timer = Timer()
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    timer.record("diffusers")
    diffusers_load_config = {
        "low_cpu_mem_usage": True,
        "torch_dtype": devices.dtype,
        "load_connected_pipeline": True,
        "safety_checker": None, # sd15 specific but we cant know ahead of time
        "requires_safety_checker": False, # sd15 specific but we cant know ahead of time
        # "use_safetensors": True,
    }
    if revision is not None:
        diffusers_load_config['revision'] = revision
    if shared.opts.diffusers_model_load_variant != 'default':
        diffusers_load_config['variant'] = shared.opts.diffusers_model_load_variant
    if shared.opts.diffusers_pipeline == 'Custom Diffusers Pipeline' and len(shared.opts.custom_diffusers_pipeline) > 0:
        shared.log.debug(f'Model pipeline: pipeline="{shared.opts.custom_diffusers_pipeline}"')
        diffusers_load_config['custom_pipeline'] = shared.opts.custom_diffusers_pipeline
    if shared.opts.data.get('sd_model_checkpoint', '') == 'model.safetensors' or shared.opts.data.get('sd_model_checkpoint', '') == '':
        shared.opts.data['sd_model_checkpoint'] = "stabilityai/stable-diffusion-xl-base-1.0"

    if (op == 'model' or op == 'dict'):
        if (model_data.sd_model is not None) and (checkpoint_info is not None) and (getattr(model_data.sd_model, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if (model_data.sd_refiner is not None) and (checkpoint_info is not None) and (getattr(model_data.sd_refiner, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return

    sd_model = None
    try:
        # initial load only
        if sd_model is None:
            if shared.cmd_opts.ckpt is not None and os.path.isdir(shared.cmd_opts.ckpt) and model_data.initial:
                sd_model, checkpoint_info = load_diffuser_initial(diffusers_load_config, op)

        # unload current model
        checkpoint_info = checkpoint_info or select_checkpoint(op=op)
        if checkpoint_info is None:
            unload_model_weights(op=op)
            return

        # detect pipeline
        pipeline, model_type = sd_detect.detect_pipeline(checkpoint_info.path, op)

        # preload vae so it can be used as param
        vae = None
        sd_vae.loaded_vae_file = None
        if model_type is None:
            shared.log.error(f'Load {op}: pipeline={shared.opts.diffusers_pipeline} not detected')
            return
        vae_file = None
        if model_type.startswith('Stable Diffusion') and (op == 'model' or op == 'refiner'): # preload vae for sd models
            vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
            vae = sd_vae.load_vae_diffusers(checkpoint_info.path, vae_file, vae_source)
            if vae is not None:
                diffusers_load_config["vae"] = vae
                timer.record("vae")

        # load with custom loader
        if sd_model is None:
            sd_model = load_diffuser_force(model_type, checkpoint_info, diffusers_load_config, op)

        # load from hf folder-style
        if sd_model is None:
            if os.path.isdir(checkpoint_info.path) or checkpoint_info.type == 'huggingface' or checkpoint_info.type == 'transformer':
                sd_model = load_diffuser_folder(model_type, pipeline, checkpoint_info, diffusers_load_config, op)

        # load from single-file
        if sd_model is None:
            if os.path.isfile(checkpoint_info.path) and checkpoint_info.path.lower().endswith('.safetensors'):
                sd_model = load_diffuser_file(model_type, pipeline, checkpoint_info, diffusers_load_config, op)

        if sd_model is None:
            shared.log.error(f'Load {op}: name="{checkpoint_info.name if checkpoint_info is not None else None}" not loaded')
            return

        sd_model.sd_model_hash = checkpoint_info.calculate_shorthash() # pylint: disable=attribute-defined-outside-init
        sd_model.sd_checkpoint_info = checkpoint_info # pylint: disable=attribute-defined-outside-init
        sd_model.sd_model_checkpoint = checkpoint_info.filename # pylint: disable=attribute-defined-outside-init
        if hasattr(sd_model, "prior_pipe"):
            sd_model.default_scheduler = copy.deepcopy(sd_model.prior_pipe.scheduler) if hasattr(sd_model.prior_pipe, "scheduler") else None
        else:
            sd_model.default_scheduler = copy.deepcopy(sd_model.scheduler) if hasattr(sd_model, "scheduler") else None
        sd_model.is_sdxl = False # a1111 compatibility item
        sd_model.is_sd2 = hasattr(sd_model, 'cond_stage_model') and hasattr(sd_model.cond_stage_model, 'model') # a1111 compatibility item
        sd_model.is_sd1 = not sd_model.is_sd2 # a1111 compatibility item
        sd_model.logvar = sd_model.logvar.to(devices.device) if hasattr(sd_model, 'logvar') else None # fix for training
        shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256
        if hasattr(sd_model, "set_progress_bar_config"):
            sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining}', ncols=80, colour='#327fba')

        if "Kandinsky" in sd_model.__class__.__name__: # need a special case
            sd_model.scheduler.name = 'DDIM'

        if model_type not in ['Stable Cascade']: # need a special-case
            sd_unet.load_unet(sd_model)

        add_noise_pred_to_diffusers_callback(sd_model)

        timer.record("load")

        if op == 'refiner':
            model_data.sd_refiner = sd_model
        else:
            model_data.sd_model = sd_model

        reload_text_encoder(initial=True) # must be before embeddings
        timer.record("te")

        if debug_load:
            shared.log.trace(f'Model components: {list(get_signature(sd_model).values())}')

        from modules.textual_inversion import textual_inversion
        sd_model.embedding_db = textual_inversion.EmbeddingDatabase()
        sd_model.embedding_db.add_embedding_dir(shared.opts.embeddings_dir)
        sd_model.embedding_db.load_textual_inversion_embeddings(force_reload=True)
        timer.record("embeddings")

        from modules import prompt_parser_diffusers
        prompt_parser_diffusers.insert_parser_highjack(sd_model.__class__.__name__)
        prompt_parser_diffusers.cache.clear()

        set_diffuser_options(sd_model, vae, op, offload=False)
        if shared.opts.nncf_compress_weights and not ('Model' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx"):
            sd_model = sd_models_compile.nncf_compress_weights(sd_model) # run this before move model so it can be compressed in CPU
        if shared.opts.optimum_quanto_weights:
            sd_model = sd_models_compile.optimum_quanto_weights(sd_model) # run this before move model so it can be compressed in CPU
        timer.record("options")

        set_diffuser_offload(sd_model, op)
        if op == 'model' and not (os.path.isdir(checkpoint_info.path) or checkpoint_info.type == 'huggingface'):
            if getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None and vae_file is not None:
                sd_vae.apply_vae_config(shared.sd_model.sd_checkpoint_info.filename, vae_file, sd_model)
        if op == 'refiner' and shared.opts.diffusers_move_refiner:
            shared.log.debug('Moving refiner model to CPU')
            move_model(sd_model, devices.cpu)
        else:
            move_model(sd_model, devices.device)
        timer.record("move")

        if shared.opts.ipex_optimize:
            sd_model = sd_models_compile.ipex_optimize(sd_model)

        if ('Model' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend != 'none'):
            sd_model = sd_models_compile.compile_diffusers(sd_model)
        timer.record("compile")

        if shared.opts.enable_linfusion:
            from modules import linfusion
            linfusion.apply(sd_model)
            timer.record("linfusion")

    except Exception as e:
        shared.log.error(f"Load {op}: {e}")
        errors.display(e, "Model")

    if shared.opts.diffusers_offload_mode != 'balanced':
        devices.torch_gc(force=True)
    if sd_model is not None:
        script_callbacks.model_loaded_callback(sd_model)

    if debug_load:
        from modules import modelstats
        modelstats.analyze()

    shared.log.info(f"Load {op}: time={timer.summary()} native={get_native(sd_model)} memory={memory_stats()}")


class DiffusersTaskType(Enum):
    TEXT_2_IMAGE = 1
    IMAGE_2_IMAGE = 2
    INPAINTING = 3
    INSTRUCT = 4


def get_diffusers_task(pipe: diffusers.DiffusionPipeline) -> DiffusersTaskType:
    if pipe.__class__.__name__ in ["StableVideoDiffusionPipeline", "LEditsPPPipelineStableDiffusion", "LEditsPPPipelineStableDiffusionXL", "OmniGenPipeline"]:
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif pipe.__class__.__name__ == "StableDiffusionXLInstructPix2PixPipeline":
        return DiffusersTaskType.INSTRUCT
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING.values():
        return DiffusersTaskType.IMAGE_2_IMAGE
    elif pipe.__class__ in diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING.values():
        return DiffusersTaskType.INPAINTING
    else:
        return DiffusersTaskType.TEXT_2_IMAGE


def get_signature(cls):
    signature = inspect.signature(cls.__init__, follow_wrapped=True, eval_str=True)
    return signature.parameters


def get_call(cls):
    signature = inspect.signature(cls.__call__, follow_wrapped=True, eval_str=True)
    return signature.parameters


def switch_pipe(cls: diffusers.DiffusionPipeline, pipeline: diffusers.DiffusionPipeline = None, force = False, args = {}):
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
        signature = get_signature(cls)
        possible = signature.keys()
        if not force and isinstance(pipeline, cls) and args == {}:
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
                if signature[item].default != inspect._empty: # has default value so we dont have to worry about it # pylint: disable=protected-access
                    continue
                if item not in components_used:
                    shared.log.warning(f'Pipeling switch: missing component={item} type={signature[item].annotation}')
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
    if pipe is not None and shared.sd_model_type == 'sdxl' and hasattr(pipe, 'config') and 'requires_aesthetics_score' in pipe.config and hasattr(pipe, '_internal_dict'):
        debug_process(f'Pipeline clean: {pipe.__class__.__name__}')
        # diffusers adds requires_aesthetics_score with img2img and complains if requires_aesthetics_score exist in txt2img
        internal_dict = dict(pipe._internal_dict) # pylint: disable=protected-access
        internal_dict.pop('requires_aesthetics_score', None)
        del pipe._internal_dict
        pipe.register_to_config(**internal_dict)


def set_diffuser_pipe(pipe, new_pipe_type):
    exclude = [
        'StableDiffusionReferencePipeline',
        'StableDiffusionAdapterPipeline',
        'AnimateDiffPipeline',
        'AnimateDiffSDXLPipeline',
        'OmniGenPipeline',
        'StableDiffusion3ControlNetPipeline',
        'InstantIRPipeline',
        'FluxFillPipeline',
        'FluxControlPipeline',
        'StableVideoDiffusionPipeline',
    ]

    has_errors = False
    if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
        clean_diffuser_pipe(pipe)

    if get_diffusers_task(pipe) == new_pipe_type:
        return pipe

    # skip specific pipelines
    cls = pipe.__class__.__name__
    if cls in exclude:
        return pipe
    if 'Onnx' in cls:
        return pipe

    new_pipe = None
    # in some cases we want to reset the pipeline to parent as they dont have their own variants
    if new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE or new_pipe_type == DiffusersTaskType.INPAINTING:
        if cls == 'StableDiffusionPAGPipeline':
            pipe = switch_pipe(diffusers.StableDiffusionPipeline, pipe)
        if cls == 'StableDiffusionXLPAGPipeline':
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

    if new_pipe is None:
        if hasattr(pipe, 'config'): # real pipeline which can be auto-switched
            try:
                if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
                    new_pipe = diffusers.AutoPipelineForText2Image.from_pipe(pipe)
                elif new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
                    new_pipe = diffusers.AutoPipelineForImage2Image.from_pipe(pipe)
                elif new_pipe_type == DiffusersTaskType.INPAINTING:
                    new_pipe = diffusers.AutoPipelineForInpainting.from_pipe(pipe)
                else:
                    shared.log.error(f'Pipeline class change failed: type={new_pipe_type} pipeline={cls}')
                    return pipe
            except Exception as e: # pylint: disable=unused-variable
                shared.log.warning(f'Pipeline class change failed: type={new_pipe_type} pipeline={cls} {e}')
                has_errors = True
        if not hasattr(pipe, 'config') or has_errors:
            try: # maybe a wrapper pipeline so just change the class
                if new_pipe_type == DiffusersTaskType.TEXT_2_IMAGE:
                    pipe.__class__ = diffusers.pipelines.auto_pipeline._get_task_class(diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING, cls) # pylint: disable=protected-access
                    new_pipe = pipe
                elif new_pipe_type == DiffusersTaskType.IMAGE_2_IMAGE:
                    pipe.__class__ = diffusers.pipelines.auto_pipeline._get_task_class(diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING, cls) # pylint: disable=protected-access
                    new_pipe = pipe
                elif new_pipe_type == DiffusersTaskType.INPAINTING:
                    pipe.__class__ = diffusers.pipelines.auto_pipeline._get_task_class(diffusers.pipelines.auto_pipeline.AUTO_INPAINT_PIPELINES_MAPPING, cls) # pylint: disable=protected-access
                    new_pipe = pipe
                else:
                    shared.log.error(f'Pipeline class set failed: type={new_pipe_type} pipeline={cls}')
                    return pipe
            except Exception as e: # pylint: disable=unused-variable
                shared.log.warning(f'Pipeline class set failed: type={new_pipe_type} pipeline={cls} {e}')
                has_errors = True

    # if pipe.__class__ == new_pipe.__class__:
    #    return pipe
    new_pipe.sd_checkpoint_info = sd_checkpoint_info
    new_pipe.sd_model_checkpoint = sd_model_checkpoint
    new_pipe.embedding_db = embedding_db
    new_pipe.sd_model_hash = sd_model_hash
    new_pipe.has_accelerate = has_accelerate
    new_pipe.current_attn_name = current_attn_name
    new_pipe.default_scheduler = default_scheduler
    if image_encoder is not None:
        new_pipe.image_encoder = image_encoder
    if feature_extractor is not None:
        new_pipe.feature_extractor = feature_extractor
    if new_pipe.__class__.__name__ == 'FluxPipeline':
        new_pipe.register_modules(image_encoder = image_encoder)
        new_pipe.register_modules(feature_extractor = feature_extractor)
    new_pipe.is_sdxl = getattr(pipe, 'is_sdxl', False) # a1111 compatibility item
    new_pipe.is_sd2 = getattr(pipe, 'is_sd2', False)
    new_pipe.is_sd1 = getattr(pipe, 'is_sd1', True)
    if hasattr(new_pipe, 'watermark'):
        new_pipe.watermark = NoWatermark()
    add_noise_pred_to_diffusers_callback(new_pipe)

    if hasattr(new_pipe, 'pipe'): # also handle nested pipelines
        new_pipe.pipe = set_diffuser_pipe(new_pipe.pipe, new_pipe_type)
        add_noise_pred_to_diffusers_callback(new_pipe.pipe)

    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    shared.log.debug(f"Pipeline class change: original={cls} target={new_pipe.__class__.__name__} device={pipe.device} fn={fn}") # pylint: disable=protected-access
    pipe = new_pipe
    return pipe


def set_diffusers_attention(pipe):
    import diffusers.models.attention_processor as p

    def set_attn(pipe, attention):
        if attention is None:
            return
        if not hasattr(pipe, "_internal_dict"):
            return
        modules = [getattr(pipe, n, None) for n in pipe._internal_dict.keys()] # pylint: disable=protected-access
        modules = [m for m in modules if isinstance(m, torch.nn.Module) and hasattr(m, "set_attn_processor")]
        for module in modules:
            if module.__class__.__name__ in ['SD3Transformer2DModel']:
                module.set_attn_processor(p.JointAttnProcessor2_0())
            elif module.__class__.__name__ in ['FluxTransformer2DModel']:
                module.set_attn_processor(p.FluxAttnProcessor2_0())
            elif module.__class__.__name__ in ['HunyuanDiT2DModel']:
                module.set_attn_processor(p.HunyuanAttnProcessor2_0())
            elif module.__class__.__name__ in ['AuraFlowTransformer2DModel']:
                module.set_attn_processor(p.AuraFlowAttnProcessor2_0())
            elif 'KandinskyCombinedPipeline' in pipe.__class__.__name__:
                pass
            elif 'Transformer' in module.__class__.__name__:
                pass # unknown transformer so probably dont want to force attention processor
            else:
                module.set_attn_processor(attention)

    # if hasattr(pipe, 'pipe'):
    #    set_diffusers_attention(pipe.pipe)

    if 'ControlNet' in pipe.__class__.__name__: # do not replace attention in ControlNet pipelines
        return
    shared.log.debug(f'Setting model: attention="{shared.opts.cross_attention_optimization}"')
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

    pipe.current_attn_name = shared.opts.cross_attention_optimization


def add_noise_pred_to_diffusers_callback(pipe):
    if not hasattr(pipe, "_callback_tensor_inputs"):
        return pipe
    if pipe.__class__.__name__.startswith("StableDiffusion"):
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    elif pipe.__class__.__name__.startswith("StableCascade"):
        pipe.prior_pipe._callback_tensor_inputs.append("predicted_image_embedding") # pylint: disable=protected-access
    elif hasattr(pipe, "scheduler") and "flow" in pipe.scheduler.__class__.__name__.lower():
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    elif hasattr(pipe, "default_scheduler") and "flow" in pipe.default_scheduler.__class__.__name__.lower():
        pipe._callback_tensor_inputs.append("noise_pred") # pylint: disable=protected-access
    return pipe


def get_native(pipe: diffusers.DiffusionPipeline):
    if hasattr(pipe, "vae") and hasattr(pipe.vae.config, "sample_size"):
        size = pipe.vae.config.sample_size # Stable Diffusion
    elif hasattr(pipe, "movq") and hasattr(pipe.movq.config, "sample_size"):
        size = pipe.movq.config.sample_size # Kandinsky
    elif hasattr(pipe, "unet") and hasattr(pipe.unet.config, "sample_size"):
        size = pipe.unet.config.sample_size
    else:
        size = 0
    return size


def load_model(checkpoint_info=None, already_loaded_state_dict=None, timer=None, op='model'):
    from ldm.util import instantiate_from_config
    from modules import lowvram, sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint(op=op)
    if checkpoint_info is None:
        return
    if op == 'model' or op == 'dict':
        if (model_data.sd_model is not None) and (getattr(model_data.sd_model, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_model.sd_checkpoint_info.hash): # trying to load the same model
            return
    else:
        if (model_data.sd_refiner is not None) and (getattr(model_data.sd_refiner, 'sd_checkpoint_info', None) is not None) and (checkpoint_info.hash == model_data.sd_refiner.sd_checkpoint_info.hash): # trying to load the same model
            return
    shared.log.debug(f'Load {op}: name={checkpoint_info.filename} dict={already_loaded_state_dict is not None}')
    if timer is None:
        timer = Timer()
    current_checkpoint_info = None
    if op == 'model' or op == 'dict':
        if model_data.sd_model is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            current_checkpoint_info = getattr(model_data.sd_model, 'sd_checkpoint_info', None)
            unload_model_weights(op=op)
    else:
        if model_data.sd_refiner is not None:
            sd_hijack.model_hijack.undo_hijack(model_data.sd_refiner)
            current_checkpoint_info = getattr(model_data.sd_refiner, 'sd_checkpoint_info', None)
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
        shared.log.error(f'Load {op}: path="{checkpoint_info.filename}"')
        if current_checkpoint_info is not None:
            shared.log.info(f'Load {op}: previous="{current_checkpoint_info.filename}" restore')
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
    shared.log.info(f'Model load finished: {memory_stats()}')


def reload_text_encoder(initial=False):
    if initial and (shared.opts.sd_text_encoder is None or shared.opts.sd_text_encoder == 'None'):
        return # dont unload
    signature = get_signature(shared.sd_model)
    t5 = [k for k, v in signature.items() if 'T5EncoderModel' in str(v)]
    if hasattr(shared.sd_model, 'text_encoder') and 'vit' in shared.opts.sd_text_encoder.lower():
        from modules.model_te import set_clip
        set_clip(pipe=shared.sd_model)
    elif len(t5) > 0:
        from modules.model_te import set_t5
        shared.log.debug(f'Load module: type=t5 path="{shared.opts.sd_text_encoder}" module="{t5[0]}"')
        set_t5(pipe=shared.sd_model, module=t5[0], t5=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)
    elif hasattr(shared.sd_model, 'text_encoder_3'):
        from modules.model_te import set_t5
        shared.log.debug(f'Load module: type=t5 path="{shared.opts.sd_text_encoder}" module="text_encoder_3"')
        set_t5(pipe=shared.sd_model, module='text_encoder_3', t5=shared.opts.sd_text_encoder, cache_dir=shared.opts.diffusers_dir)


def reload_model_weights(sd_model=None, info=None, reuse_dict=False, op='model', force=False, revision=None):
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
        shared.log.debug(f'Load {op} dict: target="{checkpoint_info.filename}" existing={sd_model is not None} info={info}')
    else:
        model_data.sd_dict = 'None'
        # shared.log.debug(f'Load {op}: target="{checkpoint_info.filename}" existing={sd_model is not None} info={info}')
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
            shared.log.info(f'Load {op}: reusing dictionary')
            sd_hijack.model_hijack.undo_hijack(sd_model)
        else:
            unload_model_weights(op=op)
            sd_model = None
    timer = Timer()
    # TODO model loader: implement model in-memory caching
    state_dict = get_checkpoint_state_dict(checkpoint_info, timer) if not shared.native else None
    checkpoint_config = sd_models_config.find_checkpoint_config(state_dict, checkpoint_info)
    timer.record("config")
    if sd_model is None or checkpoint_config != getattr(sd_model, 'used_config', None) or force:
        sd_model = None
        if not shared.native:
            load_model(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op)
            model_data.sd_dict = shared.opts.sd_model_dict
        else:
            load_diffuser(checkpoint_info, already_loaded_state_dict=state_dict, timer=timer, op=op, revision=revision)
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
    shared.log.info(f"Load {op} using fallback: model={checkpoint_info.title}")
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
    shared.log.info(f"Load {op}: time={timer.summary()}")
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
    if hasattr(sd_model, "_internal_dict"):
        keys = sd_model._internal_dict.keys() # pylint: disable=protected-access
    else:
        keys = get_signature(sd_model).keys()
    for module_name in keys: # pylint: disable=protected-access
        module = getattr(sd_model, module_name, None)
        if isinstance(module, torch.nn.Module):
            network_layer_name = getattr(module, "network_layer_name", None)
            module = remove_hook_from_module(module, recurse=True)
            if network_layer_name:
                module.network_layer_name = network_layer_name
    sd_model.has_accelerate = False


def clear_caches():
    shared.log.debug('Cache clear')
    if not shared.opts.lora_legacy:
        from modules.lora import networks
        networks.loaded_networks.clear()
        networks.previously_loaded_networks.clear()
        networks.lora_cache.clear()
    from modules import prompt_parser_diffusers
    prompt_parser_diffusers.cache.clear()


def unload_model_weights(op='model'):
    if shared.compiled_model_state is not None:
        shared.compiled_model_state.compiled_cache.clear()
        shared.compiled_model_state.req_cache.clear()
        shared.compiled_model_state.partitioned_modules.clear()
    if op == 'model' or op == 'dict':
        if model_data.sd_model:
            if not shared.native:
                from modules import sd_hijack
                move_model(model_data.sd_model, devices.cpu)
                sd_hijack.model_hijack.undo_hijack(model_data.sd_model)
            elif not ('Model' in shared.opts.cuda_compile and shared.opts.cuda_compile_backend == "openvino_fx"):
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
