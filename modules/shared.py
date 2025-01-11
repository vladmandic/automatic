from functools import lru_cache
import io
import os
import sys
import time
import json
import threading
import contextlib
from types import SimpleNamespace
from urllib.parse import urlparse
from enum import Enum
import psutil
import requests
import gradio as gr
import fasteners
import orjson
import diffusers
from rich.console import Console
from modules import errors, devices, shared_items, shared_state, cmd_args, theme, history, files_cache
from modules.paths import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir # pylint: disable=W0611
from modules.dml import memory_providers, default_memory_provider, directml_do_hijack
from modules.onnx_impl import initialize_onnx, execution_providers
from modules.memstats import memory_stats, ram_stats # pylint: disable=unused-import
from modules.ui_components import DropdownEditable
import modules.interrogate
import modules.memmon
import modules.styles
import modules.paths as paths
from installer import print_dict
from installer import log as central_logger # pylint: disable=E0611


errors.install([gr])
demo: gr.Blocks = None
api = None
log = central_logger
progress_print_out = sys.stdout
parser = cmd_args.parser
url = 'https://github.com/vladmandic/automatic'
cmd_opts, _ = parser.parse_known_args()
hide_dirs = {"visible": not cmd_opts.hide_ui_dir_config}
xformers_available = False
locking_available = True
clip_model = None
interrogator = modules.interrogate.InterrogateModels(os.path.join("models", "interrogate"))
sd_upscalers = []
detailers = []
face_restorers = []
yolo = None
tab_names = []
extra_networks = []
options_templates = {}
hypernetworks = {}
loaded_hypernetworks = []
settings_components = None
latent_upscale_default_mode = "None"
latent_upscale_modes = {
    "Latent Nearest": {"mode": "nearest", "antialias": False},
    "Latent Nearest-exact": {"mode": "nearest-exact", "antialias": False},
    "Latent Area": {"mode": "area", "antialias": False},
    "Latent Bilinear": {"mode": "bilinear", "antialias": False},
    "Latent Bicubic": {"mode": "bicubic", "antialias": False},
    "Latent Bilinear antialias": {"mode": "bilinear", "antialias": True},
    "Latent Bicubic antialias": {"mode": "bicubic", "antialias": True},
    # "Latent Linear": {"mode": "linear", "antialias": False}, # not supported for latents with channels=4
    # "Latent Trilinear": {"mode": "trilinear", "antialias": False}, # not supported for latents with channels=4
}
restricted_opts = {
    "samples_filename_pattern",
    "directories_filename_pattern",
    "outdir_samples",
    "outdir_txt2img_samples",
    "outdir_img2img_samples",
    "outdir_extras_samples",
    "outdir_control_samples",
    "outdir_grids",
    "outdir_txt2img_grids",
    "outdir_save",
    "outdir_init_images"
}
resize_modes = ["None", "Fixed", "Crop", "Fill", "Outpaint", "Context aware"]
compatibility_opts = ['clip_skip', 'uni_pc_lower_order_final', 'uni_pc_order']
console = Console(log_time=True, log_time_format='%H:%M:%S-%f')
dir_timestamps = {}
dir_cache = {}
max_workers = 8
if os.environ.get("SD_HFCACHEDIR", None) is not None:
    hfcache_dir = os.environ.get("SD_HFCACHEDIR")
if os.environ.get("HF_HUB_CACHE", None) is not None:
    hfcache_dir = os.environ.get("HF_HUB_CACHE")
elif os.environ.get("HF_HUB", None) is not None:
    hfcache_dir = os.path.join(os.environ.get("HF_HUB"), '.cache')
else:
    hfcache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
    os.environ["HF_HUB_CACHE"] = hfcache_dir
log.debug(f'Huggingface cache: folder="{hfcache_dir}"')


class Backend(Enum):
    ORIGINAL = 1
    DIFFUSERS = 2


state = shared_state.State()
if not hasattr(cmd_opts, "use_openvino"):
    cmd_opts.use_openvino = False


def readfile(filename, silent=False, lock=False):
    global locking_available # pylint: disable=global-statement
    data = {}
    lock_file = None
    locked = False
    if lock and locking_available:
        try:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock")
            lock_file.logger.disabled = True
            locked = lock_file.acquire_read_lock(blocking=True, timeout=3)
        except Exception as err:
            lock_file = None
            locking_available = False
            log.error(f'File read lock: file="{filename}" {err}')
            locked = False
    try:
        # if not os.path.exists(filename):
        #    return {}
        t0 = time.time()
        with open(filename, "rb") as file:
            b = file.read()
            data = orjson.loads(b) # pylint: disable=no-member
        # if type(data) is str:
        #    data = json.loads(data)
        t1 = time.time()
        if not silent:
            fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            log.debug(f'Read: file="{filename}" json={len(data)} bytes={os.path.getsize(filename)} time={t1-t0:.3f} fn={fn}')
    except FileNotFoundError as err:
        log.debug(f'Reading failed: {filename} {err}')
    except Exception as err:
        if not silent:
            log.error(f'Reading failed: {filename} {err}')
    try:
        if locking_available and lock_file is not None:
            lock_file.release_read_lock()
        if locked and os.path.exists(f"{filename}.lock"):
            os.remove(f"{filename}.lock")
    except Exception:
        locking_available = False
    return data


def writefile(data, filename, mode='w', silent=False, atomic=False):
    import tempfile
    global locking_available # pylint: disable=global-statement
    lock_file = None
    locked = False

    def default(obj):
        log.error(f'Save: file="{filename}" not a valid object: {obj}')
        return str(obj)

    try:
        t0 = time.time()
        # skipkeys=True, ensure_ascii=True, check_circular=True, allow_nan=True
        if type(data) == dict:
            output = json.dumps(data, indent=2, default=default)
        elif type(data) == list:
            output = json.dumps(data, indent=2, default=default)
        elif isinstance(data, object):
            simple = {}
            for k in data.__dict__:
                if data.__dict__[k] is not None:
                    simple[k] = data.__dict__[k]
            output = json.dumps(simple, indent=2, default=default)
        else:
            raise ValueError('not a valid object')
    except Exception as err:
        log.error(f'Save failed: file="{filename}" {err}')
        return
    try:
        if locking_available:
            lock_file = fasteners.InterProcessReaderWriterLock(f"{filename}.lock") if locking_available else None
            lock_file.logger.disabled = True
            locked = lock_file.acquire_write_lock(blocking=True, timeout=3) if lock_file is not None else False
    except Exception as err:
        locking_available = False
        lock_file = None
        log.error(f'File write lock: file="{filename}" {err}')
        locked = False
    try:
        if atomic:
            with tempfile.NamedTemporaryFile(mode=mode, encoding="utf8", delete=False, dir=os.path.dirname(filename)) as f:
                f.write(output)
                f.flush()
                os.fsync(f.fileno())
                os.replace(f.name, filename)
        else:
            with open(filename, mode=mode, encoding="utf8") as file:
                file.write(output)
        t1 = time.time()
        if not silent:
            log.debug(f'Save: file="{filename}" json={len(data)} bytes={len(output)} time={t1-t0:.3f}')
    except Exception as err:
        log.error(f'Save failed: file="{filename}" {err}')
    try:
        if locking_available and lock_file is not None:
            lock_file.release_write_lock()
        if locked and os.path.exists(f"{filename}.lock"):
            os.remove(f"{filename}.lock")
    except Exception:
        locking_available = False


# early select backend
default_backend = 'diffusers'
early_opts = readfile(cmd_opts.config, silent=True)
early_backend = early_opts.get('sd_backend', default_backend)
backend = Backend.DIFFUSERS if early_backend.lower() == 'diffusers' else Backend.ORIGINAL
if cmd_opts.backend is not None: # override with args
    backend = Backend.DIFFUSERS if cmd_opts.backend.lower() == 'diffusers' else Backend.ORIGINAL
if cmd_opts.use_openvino: # override for openvino
    backend = Backend.DIFFUSERS
    from modules.intel.openvino import get_device_list as get_openvino_device_list # pylint: disable=ungrouped-imports
elif cmd_opts.use_ipex or devices.has_xpu():
    from modules.intel.ipex import ipex_init
    ok, e = ipex_init()
    if not ok:
        log.error(f'IPEX initialization failed: {e}')
elif cmd_opts.use_directml:
    name = 'directml'
    from modules.dml import directml_init
    ok, e = directml_init()
    if not ok:
        log.error(f'DirectML initialization failed: {e}')
devices.backend = devices.get_backend(cmd_opts)
devices.device = devices.get_optimal_device()
cpu_memory = round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2)
mem_stat = memory_stats()
gpu_memory = mem_stat['gpu']['total'] if "gpu" in mem_stat else 0
native = backend == Backend.DIFFUSERS
if not files_cache.do_cache_folders:
    log.warning('File cache disabled: ')


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, section=None, refresh=None, folder=None, submit=None, comment_before='', comment_after=''):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = section
        self.refresh = refresh
        self.folder = folder
        self.comment_before = comment_before # HTML text that will be added after label in UI
        self.comment_after = comment_after # HTML text that will be added before label in UI
        self.submit = submit
        self.exclude = ['sd_model_checkpoint', 'sd_model_refiner', 'sd_vae', 'sd_unet', 'sd_text_encoder', 'sd_model_dict']

    def needs_reload_ui(self):
        return self

    def link(self, label, uri):
        self.comment_before += f"[<a href='{uri}' target='_blank'>{label}</a>]"
        return self

    def js(self, label, js_func):
        self.comment_before += f"[<a onclick='{js_func}(); return false'>{label}</a>]"
        return self

    def info(self, info):
        self.comment_after += f"<span class='info'>({info})</span>"
        return self

    def html(self, info):
        self.comment_after += f"<span class='info'>{info}</span>"
        return self

    def needs_restart(self):
        self.comment_after += " <span class='info'>(requires restart)</span>"
        return self

    def validate(self, opt, value):
        if opt in self.exclude:
            return True
        args = self.component_args if self.component_args is not None else {}
        if callable(args):
            try:
                args = args()
            except Exception:
                args = {}
        choices = args.get("choices", [])
        if callable(choices):
            try:
                choices = choices()
            except Exception:
                choices = []
        if len(choices) > 0:
            if not isinstance(value, list):
                value = [value]
            for v in value:
                if v not in choices:
                    log.debug(f'Setting validation: "{opt}"="{v}" default="{self.default}" choices={choices}')
                    # return False
        minimum = args.get("minimum", None)
        maximum = args.get("maximum", None)
        if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
            log.error(f'Setting validation: "{opt}"={value} default={self.default} minimum={minimum} maximum={maximum}')
            return False
        return True

    def __str__(self) -> str:
        args = self.component_args if self.component_args is not None else {}
        if callable(args):
            args = args()
        choices = args.get("choices", [])
        return f'OptionInfo: label="{self.label}" section="{self.section}" component="{self.component}" default="{self.default}" refresh="{self.refresh is not None}" change="{self.onchange is not None}" args={args} choices={choices}'


def options_section(section_identifier, options_dict):
    for v in options_dict.values():
        v.section = section_identifier
    return options_dict


def list_checkpoint_titles():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.checkpoint_titles()


list_checkpoint_tiles = list_checkpoint_titles # alias for legacy typo
default_checkpoint = list_checkpoint_titles()[0] if len(list_checkpoint_titles()) > 0 else "model.safetensors"


def is_url(string):
    parsed_url = urlparse(string)
    return all([parsed_url.scheme, parsed_url.netloc])


def reload_hypernetworks():
    from modules.hypernetworks import hypernetwork
    global hypernetworks # pylint: disable=W0603
    hypernetworks = hypernetwork.list_hypernetworks(opts.hypernetwork_dir)


def refresh_checkpoints():
    import modules.sd_models # pylint: disable=W0621
    return modules.sd_models.list_models()


def refresh_vaes():
    import modules.sd_vae # pylint: disable=W0621
    modules.sd_vae.refresh_vae_list()


def refresh_upscalers():
    import modules.modelloader # pylint: disable=W0621
    modules.modelloader.load_upscalers()


def list_samplers():
    import modules.sd_samplers # pylint: disable=W0621
    modules.sd_samplers.set_samplers()
    return modules.sd_samplers.all_samplers


def temp_disable_extensions():
    disable_safe = ['sd-webui-controlnet', 'multidiffusion-upscaler-for-automatic1111', 'a1111-sd-webui-lycoris', 'sd-webui-agent-scheduler', 'clip-interrogator-ext', 'stable-diffusion-webui-rembg', 'sd-extension-chainner', 'stable-diffusion-webui-images-browser']
    disable_diffusers = ['sd-webui-controlnet', 'multidiffusion-upscaler-for-automatic1111', 'a1111-sd-webui-lycoris', 'sd-webui-animatediff']
    disable_themes = ['sd-webui-lobe-theme', 'cozy-nest', 'sdnext-modernui']
    disable_original = []
    disabled = []
    if modules.shared.cmd_opts.theme is not None:
        theme_name = modules.shared.cmd_opts.theme
    else:
        theme_name = f'{modules.shared.opts.theme_type.lower()}/{modules.shared.opts.gradio_theme}'

    if theme_name == 'lobe':
        disable_themes.remove('sd-webui-lobe-theme')
    elif theme_name == 'cozy-nest' or theme_name == 'cozy':
        disable_themes.remove('cozy-nest')
    elif '/' not in theme_name: # set default themes per type
        if theme_name == 'standard' or theme_name == 'default':
            theme_name = 'standard/black-teal'
        if theme_name == 'modern':
            theme_name = 'modern/Default'
        if theme_name == 'gradio':
            theme_name = 'gradio/default'
        if theme_name == 'huggingface':
            theme_name = 'huggingface/blaaa'

    if theme_name.lower().startswith('standard') or theme_name.lower().startswith('default'):
        modules.shared.opts.data['theme_type'] = 'Standard'
        modules.shared.opts.data['gradio_theme'] = theme_name[9:]
    elif theme_name.lower().startswith('modern'):
        modules.shared.opts.data['theme_type'] = 'Modern'
        modules.shared.opts.data['gradio_theme'] = theme_name[7:]
        disable_themes.remove('sdnext-modernui')
    elif theme_name.lower().startswith('gradio'):
        modules.shared.opts.data['theme_type'] = 'None'
        modules.shared.opts.data['gradio_theme'] = theme_name
    elif theme_name.lower().startswith('huggingface'):
        modules.shared.opts.data['theme_type'] = 'None'
        modules.shared.opts.data['gradio_theme'] = theme_name
    else:
        modules.shared.log.error(f'UI theme invalid: theme="{theme_name}" available={["standard/*", "modern/*", "none/*"]} fallback="standard/black-teal"')
        modules.shared.opts.data['theme_type'] = 'Standard'
        modules.shared.opts.data['gradio_theme'] = 'black-teal'

    for ext in disable_themes:
        if ext.lower() not in opts.disabled_extensions:
            disabled.append(ext)
    if cmd_opts.safe:
        for ext in disable_safe:
            if ext.lower() not in opts.disabled_extensions:
                disabled.append(ext)
    if native:
        for ext in disable_diffusers:
            if ext.lower() not in opts.disabled_extensions:
                disabled.append(ext)
    if not native:
        for ext in disable_original:
            if ext.lower() not in opts.disabled_extensions:
                disabled.append(ext)
    if not opts.lora_legacy:
        disabled.append('Lora')
    cmd_opts.controlnet_loglevel = 'WARNING'
    return disabled


def get_default_modes():
    default_offload_mode = "none"
    if not (cmd_opts.lowvram or cmd_opts.medvram):
        if "gpu" in mem_stat:
            if gpu_memory <= 4:
                cmd_opts.lowvram = True
                default_offload_mode = "sequential"
                log.info(f"Device detect: memory={gpu_memory:.1f} optimization=lowvram")
            # elif gpu_memory <= 8:
            #     cmd_opts.medvram = True
            #     default_offload_mode = "model"
            #     log.info(f"Device detect: memory={gpu_memory:.1f} optimization=medvram")
            else:
                default_offload_mode = "balanced"
                log.info(f"Device detect: memory={gpu_memory:.1f} optimization=balanced")
    elif cmd_opts.medvram:
        default_offload_mode = "balanced"
    elif cmd_opts.lowvram:
        default_offload_mode = "sequential"

    if devices.backend == "directml": # Force BMM for DirectML instead of SDP
        default_cross_attention = "Dynamic Attention BMM" if native else "Sub-quadratic"
    elif devices.backend == "cpu":
        default_cross_attention = "Scaled-Dot-Product" if native else "Doggettx's"
    elif devices.backend == "mps":
        default_cross_attention = "Scaled-Dot-Product" if native else "Doggettx's"
    else: # cuda, rocm, zluda, ipex, openvino
        default_cross_attention = "Scaled-Dot-Product"

    if devices.backend == "rocm":
        default_sdp_options =  ['Memory attention', 'Math attention']
    elif devices.backend == "zluda":
        default_sdp_options =  ['Math attention', 'Dynamic attention']
    else:
        default_sdp_options = ['Flash attention', 'Memory attention', 'Math attention']
    if (cmd_opts.lowvram or cmd_opts.medvram) and ('Flash attention' not in default_sdp_options and 'Dynamic attention' not in default_sdp_options):
        default_sdp_options.append('Dynamic attention')

    return default_offload_mode, default_cross_attention, default_sdp_options


startup_offload_mode, startup_cross_attention, startup_sdp_options = get_default_modes()

options_templates.update(options_section(('sd', "Models & Loading"), {
    "sd_backend": OptionInfo(default_backend, "Execution backend", gr.Radio, {"choices": ["diffusers", "original"] }),
    "diffusers_pipeline": OptionInfo('Autodetect', 'Model pipeline', gr.Dropdown, lambda: {"choices": list(shared_items.get_pipelines()), "visible": native}),
    "sd_model_checkpoint": OptionInfo(default_checkpoint, "Base model", DropdownEditable, lambda: {"choices": list_checkpoint_titles()}, refresh=refresh_checkpoints),
    "sd_model_refiner": OptionInfo('None', "Refiner model", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_titles()}, refresh=refresh_checkpoints),
    "sd_unet": OptionInfo("None", "UNET model", gr.Dropdown, lambda: {"choices": shared_items.sd_unet_items()}, refresh=shared_items.refresh_unet_list),
    "latent_history": OptionInfo(16, "Latent history size", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),

    "offload_sep": OptionInfo("<h2>Model Offloading</h2>", "", gr.HTML),
    "diffusers_move_base": OptionInfo(False, "Move base model to CPU when using refiner", gr.Checkbox, {"visible": False }),
    "diffusers_move_unet": OptionInfo(False, "Move base model to CPU when using VAE", gr.Checkbox, {"visible": False }),
    "diffusers_move_refiner": OptionInfo(False, "Move refiner model to CPU when not in use", gr.Checkbox, {"visible": False }),
    "diffusers_extract_ema": OptionInfo(False, "Use model EMA weights when possible", gr.Checkbox, {"visible": False }),
    "diffusers_offload_mode": OptionInfo(startup_offload_mode, "Model offload mode", gr.Radio, {"choices": ['none', 'balanced', 'model', 'sequential']}),
    "diffusers_offload_min_gpu_memory": OptionInfo(0.25, "Balanced offload GPU low watermark", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01 }),
    "diffusers_offload_max_gpu_memory": OptionInfo(0.70, "Balanced offload GPU high watermark", gr.Slider, {"minimum": 0.1, "maximum": 1, "step": 0.01 }),
    "diffusers_offload_max_cpu_memory": OptionInfo(0.90, "Balanced offload CPU high watermark", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": False }),

    "advanced_sep": OptionInfo("<h2>Advanced Options</h2>", "", gr.HTML),
    "sd_checkpoint_autoload": OptionInfo(True, "Model autoload on start"),
    "sd_checkpoint_autodownload": OptionInfo(True, "Model auto-download on demand"),
    "stream_load": OptionInfo(False, "Model load using streams", gr.Checkbox),
    "diffusers_eval": OptionInfo(True, "Force model eval", gr.Checkbox, {"visible": False }),
    "diffusers_to_gpu": OptionInfo(False, "Load model directly to GPU"),
    "disable_accelerate": OptionInfo(False, "Disable accelerate", gr.Checkbox, {"visible": False }),
    "sd_model_dict": OptionInfo('None', "Use separate base dict", gr.Dropdown, lambda: {"choices": ['None'] + list_checkpoint_titles()}, refresh=refresh_checkpoints),
    "sd_checkpoint_cache": OptionInfo(0, "Cached models", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1, "visible": not native }),
}))

options_templates.update(options_section(('vae_encoder', "Variable Auto Encoder"), {
    "sd_vae": OptionInfo("Automatic", "VAE model", gr.Dropdown, lambda: {"choices": shared_items.sd_vae_items()}, refresh=shared_items.refresh_vae_list),
    "diffusers_vae_upcast": OptionInfo("default", "VAE upcasting", gr.Radio, {"choices": ['default', 'true', 'false']}),
    "no_half_vae": OptionInfo(False if not cmd_opts.use_openvino else True, "Full precision (--no-half-vae)"),
    "diffusers_vae_slicing": OptionInfo(True, "VAE slicing", gr.Checkbox, {"visible": native}),
    "diffusers_vae_tiling": OptionInfo(cmd_opts.lowvram or cmd_opts.medvram, "VAE tiling", gr.Checkbox, {"visible": native}),
    "diffusers_vae_tile_size": OptionInfo(1024, "VAE tile size", gr.Slider, {"minimum": 256, "maximum": 4096, "step": 8 }),
    "diffusers_vae_tile_overlap": OptionInfo(0.25, "VAE tile overlap", gr.Slider, {"minimum": 0, "maximum": 0.95, "step": 0.05 }),
    "sd_vae_sliced_encode": OptionInfo(False, "VAE sliced encode", gr.Checkbox, {"visible": not native}),
    "nan_skip": OptionInfo(False, "Skip Generation if NaN found in latents", gr.Checkbox),
    "rollback_vae": OptionInfo(False, "Attempt VAE roll back for NaN values"),
}))

options_templates.update(options_section(('text_encoder', "Text Encoder"), {
    "sd_text_encoder": OptionInfo('None', "Text encoder model", gr.Dropdown, lambda: {"choices": shared_items.sd_te_items()}, refresh=shared_items.refresh_te_list),
    "prompt_attention": OptionInfo("native", "Prompt attention parser", gr.Radio, {"choices": ["native", "compel", "xhinker", "a1111", "fixed"] }),
    "prompt_mean_norm": OptionInfo(False, "Prompt attention normalization", gr.Checkbox),
    "sd_textencoder_cache": OptionInfo(True, "Cache text encoder results", gr.Checkbox, {"visible": False}),
    "sd_textencoder_cache_size": OptionInfo(4, "Text encoder cache size", gr.Slider, {"minimum": 0, "maximum": 16, "step": 1}),
    "comma_padding_backtrack": OptionInfo(20, "Prompt padding", gr.Slider, {"minimum": 0, "maximum": 74, "step": 1, "visible": not native }),
    "diffusers_zeros_prompt_pad": OptionInfo(False, "Use zeros for prompt padding", gr.Checkbox),
    "diffusers_pooled": OptionInfo("default", "Diffusers SDXL pooled embeds", gr.Radio, {"choices": ['default', 'weighted']}),
}))

options_templates.update(options_section(('cuda', "Compute Settings"), {
    "math_sep": OptionInfo("<h2>Execution Precision</h2>", "", gr.HTML),
    "precision": OptionInfo("Autocast", "Precision type", gr.Radio, {"choices": ["Autocast", "Full"]}),
    "cuda_dtype": OptionInfo("Auto", "Device precision type", gr.Radio, {"choices": ["Auto", "FP32", "FP16", "BF16"]}),
    "no_half": OptionInfo(False if not cmd_opts.use_openvino else True, "Full precision (--no-half)", None, None, None),
    "upcast_sampling": OptionInfo(False if sys.platform != "darwin" else True, "Upcast sampling", gr.Checkbox, {"visible": not native}),
    "upcast_attn": OptionInfo(False, "Upcast attention layer", gr.Checkbox, {"visible": not native}),
    "cuda_cast_unet": OptionInfo(False, "Fixed UNet precision", gr.Checkbox, {"visible": not native}),

    "generator_sep": OptionInfo("<h2>Noise Options</h2>", "", gr.HTML),
    "diffusers_generator_device": OptionInfo("GPU", "Generator device", gr.Radio, {"choices": ["GPU", "CPU", "Unset"]}),

    "cross_attention_sep": OptionInfo("<h2>Cross Attention</h2>", "", gr.HTML),
    "cross_attention_optimization": OptionInfo(startup_cross_attention, "Attention optimization method", gr.Radio, lambda: {"choices": shared_items.list_crossattention(native)}),
    "sdp_options": OptionInfo(startup_sdp_options, "SDP options", gr.CheckboxGroup, {"choices": ['Flash attention', 'Memory attention', 'Math attention', 'Dynamic attention', 'Sage attention'], "visible": native}),
    "xformers_options": OptionInfo(['Flash attention'], "xFormers options", gr.CheckboxGroup, {"choices": ['Flash attention'] }),
    "dynamic_attention_slice_rate": OptionInfo(4, "Dynamic Attention slicing rate in GB", gr.Slider, {"minimum": 0.1, "maximum": gpu_memory, "step": 0.1, "visible": native}),
    "sub_quad_sep": OptionInfo("<h3>Sub-quadratic options</h3>", "", gr.HTML, {"visible": not native}),
    "sub_quad_q_chunk_size": OptionInfo(512, "Attention query chunk size", gr.Slider, {"minimum": 16, "maximum": 8192, "step": 8, "visible": not native}),
    "sub_quad_kv_chunk_size": OptionInfo(512, "Attention kv chunk size", gr.Slider, {"minimum": 0, "maximum": 8192, "step": 8, "visible": not native}),
    "sub_quad_chunk_threshold": OptionInfo(80, "Attention chunking threshold", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1, "visible": not native}),
}))

options_templates.update(options_section(('backends', "Backend Settings"), {
    "other_sep": OptionInfo("<h2>Torch Options</h2>", "", gr.HTML),
    "opt_channelslast": OptionInfo(False, "Channels last "),
    "cudnn_deterministic": OptionInfo(False, "Deterministic mode"),
    "cudnn_benchmark": OptionInfo(False, "Full-depth cuDNN benchmark"),
    "diffusers_fuse_projections": OptionInfo(False, "Fused projections"),
    "torch_expandable_segments": OptionInfo(False, "Expandable segments"),
    "cuda_mem_fraction": OptionInfo(0.0, "Memory limit", gr.Slider, {"minimum": 0, "maximum": 2.0, "step": 0.05}),
    "torch_gc_threshold": OptionInfo(70, "GC threshold", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}),
    "inference_mode": OptionInfo("no-grad", "Inference mode", gr.Radio, {"choices": ["no-grad", "inference-mode", "none"]}),
    "torch_malloc": OptionInfo("native", "Memory allocator", gr.Radio, {"choices": ['native', 'cudaMallocAsync'] }),

    "onnx_sep": OptionInfo("<h2>ONNX</h2>", "", gr.HTML),
    "onnx_execution_provider": OptionInfo(execution_providers.get_default_execution_provider().value, 'ONNX Execution Provider', gr.Dropdown, lambda: {"choices": execution_providers.available_execution_providers }),
    "onnx_cpu_fallback": OptionInfo(True, 'ONNX allow fallback to CPU'),
    "onnx_cache_converted": OptionInfo(True, 'ONNX cache converted models'),
    "onnx_unload_base": OptionInfo(False, 'ONNX unload base model when processing refiner'),

    "olive_sep": OptionInfo("<h2>Olive</h2>", "", gr.HTML),
    "olive_float16": OptionInfo(True, 'Olive use FP16 on optimization'),
    "olive_vae_encoder_float32": OptionInfo(False, 'Olive force FP32 for VAE Encoder'),
    "olive_static_dims": OptionInfo(True, 'Olive use static dimensions'),
    "olive_cache_optimized": OptionInfo(True, 'Olive cache optimized models'),

    "ipex_sep": OptionInfo("<h2>IPEX</h2>", "", gr.HTML, {"visible": devices.backend == "ipex"}),
    "ipex_optimize": OptionInfo([], "IPEX Optimize", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder", "Upscaler"], "visible": devices.backend == "ipex"}),

    "openvino_sep": OptionInfo("<h2>OpenVINO</h2>", "", gr.HTML, {"visible": cmd_opts.use_openvino}),
    "openvino_devices": OptionInfo([], "OpenVINO devices to use", gr.CheckboxGroup, {"choices": get_openvino_device_list() if cmd_opts.use_openvino else [], "visible": cmd_opts.use_openvino}), # pylint: disable=E0606
    "openvino_accuracy": OptionInfo("performance", "OpenVINO accuracy mode", gr.Radio, {"choices": ['performance', 'accuracy'], "visible": cmd_opts.use_openvino}),
    "openvino_disable_model_caching": OptionInfo(True, "OpenVINO disable model caching", gr.Checkbox, {"visible": cmd_opts.use_openvino}),
    "openvino_disable_memory_cleanup": OptionInfo(True, "OpenVINO disable memory cleanup after compile", gr.Checkbox, {"visible": cmd_opts.use_openvino}),

    "directml_sep": OptionInfo("<h2>DirectML</h2>", "", gr.HTML, {"visible": devices.backend == "directml"}),
    "directml_memory_provider": OptionInfo(default_memory_provider, 'DirectML memory stats provider', gr.Radio, {"choices": memory_providers, "visible": devices.backend == "directml"}),
    "directml_catch_nan": OptionInfo(False, "DirectML retry ops for NaN", gr.Checkbox, {"visible": devices.backend == "directml"}),
}))

options_templates.update(options_section(('quantization', "Quantization Settings"), {
    "bnb_sep": OptionInfo("<h2>BitsAndBytes</h2>", "", gr.HTML),
    "bnb_quantization": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder"], "visible": native}),
    "bnb_quantization_type": OptionInfo("nf4", "Quantization type", gr.Dropdown, {"choices": ['nf4', 'fp8', 'fp4'], "visible": native}),
    "bnb_quantization_storage": OptionInfo("uint8", "Backend storage", gr.Dropdown, {"choices": ["float16", "float32", "int8", "uint8", "float64", "bfloat16"], "visible": native}),
    "optimum_quanto_sep": OptionInfo("<h2>Optimum Quanto</h2>", "", gr.HTML),
    "optimum_quanto_weights": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder", "ControlNet"], "visible": native}),
    "optimum_quanto_weights_type": OptionInfo("qint8", "Quantization weights type", gr.Dropdown, {"choices": ['qint8', 'qfloat8_e4m3fn', 'qfloat8_e5m2', 'qint4', 'qint2'], "visible": native}),
    "optimum_quanto_activations_type": OptionInfo("none", "Quantization activations type ", gr.Dropdown, {"choices": ['none', 'qint8', 'qfloat8_e4m3fn', 'qfloat8_e5m2'], "visible": native}),
    "torchao_sep": OptionInfo("<h2>TorchAO</h2>", "", gr.HTML),
    "torchao_quantization": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder"], "visible": native}),
    "torchao_quantization_mode": OptionInfo("pre", "Quantization mode", gr.Dropdown, {"choices": ['pre', 'post'], "visible": native}),
    "torchao_quantization_type": OptionInfo("int8_weight_only", "Quantization type", gr.Dropdown, {"choices": ['int4_weight_only', 'int8_dynamic_activation_int4_weight', 'int8_weight_only', 'int8_dynamic_activation_int8_weight', 'float8_weight_only', 'float8_dynamic_activation_float8_weight', 'float8_static_activation_float8_weight'], "visible": native}),
    "nncf_sep": OptionInfo("<h2>NNCF</h2>", "", gr.HTML),
    "nncf_compress_weights": OptionInfo([], "Quantization enabled", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder", "ControlNet"], "visible": native}),
    "nncf_compress_weights_mode": OptionInfo("INT8", "Quantization type", gr.Dropdown, {"choices": ['INT8', 'INT8_SYM', 'INT4_ASYM', 'INT4_SYM', 'NF4'] if cmd_opts.use_openvino else ['INT8']}),
    "nncf_compress_weights_raito": OptionInfo(1.0, "Compress ratio", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": cmd_opts.use_openvino}),
    "nncf_quantize": OptionInfo([], "OpenVINO enabled", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder"], "visible": cmd_opts.use_openvino}),
    "nncf_quant_mode": OptionInfo("INT8", "OpenVINO mode", gr.Dropdown, {"choices": ['INT8', 'FP8_E4M3', 'FP8_E5M2'], "visible": cmd_opts.use_openvino}),
    "quant_shuffle_weights": OptionInfo(False, "Shuffle weights", gr.Checkbox, {"visible": native}),
}))

options_templates.update(options_section(('advanced', "Pipeline Modifiers"), {
    "token_merging_sep": OptionInfo("<h2>Token Merging</h2>", "", gr.HTML),
    "token_merging_method": OptionInfo("None", "Token merging enabled", gr.Radio, {"choices": ['None', 'ToMe', 'ToDo']}),
    "tome_ratio": OptionInfo(0.0, "ToMe token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}),
    "todo_ratio": OptionInfo(0.0, "ToDo token merging ratio", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05}),

    "freeu_sep": OptionInfo("<h2>FreeU</h2>", "", gr.HTML),
    "freeu_enabled": OptionInfo(False, "FreeU enabled"),
    "freeu_b1": OptionInfo(1.2, "1st stage backbone", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_b2": OptionInfo(1.4, "2nd stage backbone", gr.Slider, {"minimum": 1.0, "maximum": 2.0, "step": 0.01}),
    "freeu_s1": OptionInfo(0.9, "1st stage skip", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "freeu_s2": OptionInfo(0.2, "2nd stage skip", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),

    "pag_sep": OptionInfo("<h2>Perturbed-Attention Guidance</h2>", "", gr.HTML),
    "pag_apply_layers": OptionInfo("m0", "PAG layer names"),

    "hypertile_sep": OptionInfo("<h2>HyperTile</h2>", "", gr.HTML),
    "hypertile_unet_enabled": OptionInfo(False, "UNet Enabled"),
    "hypertile_hires_only": OptionInfo(False, "HiRes pass only"),
    "hypertile_unet_tile": OptionInfo(0, "UNet max tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),
    "hypertile_unet_min_tile": OptionInfo(0, "UNet min tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),
    "hypertile_unet_swap_size": OptionInfo(1, "UNet swap size", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "hypertile_unet_depth": OptionInfo(0, "UNet depth", gr.Slider, {"minimum": 0, "maximum": 4, "step": 1}),
    "hypertile_vae_enabled": OptionInfo(False, "VAE Enabled", gr.Checkbox),
    "hypertile_vae_tile": OptionInfo(128, "VAE tile size", gr.Slider, {"minimum": 0, "maximum": 1024, "step": 8}),
    "hypertile_vae_swap_size": OptionInfo(1, "VAE swap size", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),

    "hidiffusion_sep": OptionInfo("<h2>HiDiffusion</h2>", "", gr.HTML),
    "hidiffusion_raunet": OptionInfo(True, "Apply RAU-Net"),
    "hidiffusion_attn": OptionInfo(True, "Apply MSW-MSA"),
    "hidiffusion_steps": OptionInfo(8, "Aggressive at step", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "hidiffusion_t1": OptionInfo(-1, "Override T1 ratio", gr.Slider, {"minimum": -1, "maximum": 1.0, "step": 0.05}),
    "hidiffusion_t2": OptionInfo(-1, "Override T2 ratio", gr.Slider, {"minimum": -1, "maximum": 1.0, "step": 0.05}),

    "linfusion_sep": OptionInfo("<h2>Batch</h2>", "", gr.HTML),
    "enable_linfusion": OptionInfo(False, "Apply LinFusion distillation on load"),

    "inference_batch_sep": OptionInfo("<h2>Batch</h2>", "", gr.HTML),
    "sequential_seed": OptionInfo(True, "Batch mode uses sequential seeds"),
    "batch_frame_mode": OptionInfo(False, "Parallel process images in batch"),
}))

options_templates.update(options_section(('compile', "Model Compile"), {
    "cuda_compile_sep": OptionInfo("<h2>Model Compile</h2>", "", gr.HTML),
    "cuda_compile": OptionInfo([] if not cmd_opts.use_openvino else ["Model", "VAE"], "Compile Model", gr.CheckboxGroup, {"choices": ["Model", "VAE", "Text Encoder", "Upscaler"]}),
    "cuda_compile_backend": OptionInfo("none" if not cmd_opts.use_openvino else "openvino_fx", "Model compile backend", gr.Radio, {"choices": ['none', 'inductor', 'cudagraphs', 'aot_ts_nvfuser', 'hidet', 'migraphx', 'ipex', 'onediff', 'stable-fast', 'deep-cache', 'olive-ai', 'openvino_fx']}),
    "cuda_compile_mode": OptionInfo("default", "Model compile mode", gr.Radio, {"choices": ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs']}),
    "cuda_compile_fullgraph": OptionInfo(True if not cmd_opts.use_openvino else False, "Model compile fullgraph"),
    "cuda_compile_precompile": OptionInfo(False, "Model compile precompile"),
    "cuda_compile_verbose": OptionInfo(False, "Model compile verbose mode"),
    "cuda_compile_errors": OptionInfo(True, "Model compile suppress errors"),
    "deep_cache_interval": OptionInfo(3, "DeepCache cache interval", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
}))

options_templates.update(options_section(('system-paths', "System Paths"), {
    "models_paths_sep_options": OptionInfo("<h2>Models Paths</h2>", "", gr.HTML),
    "models_dir": OptionInfo('models', "Base path where all models are stored", folder=True),
    "ckpt_dir": OptionInfo(os.path.join(paths.models_path, 'Stable-diffusion'), "Folder with stable diffusion models", folder=True),
    "diffusers_dir": OptionInfo(os.path.join(paths.models_path, 'Diffusers'), "Folder with Huggingface models", folder=True),
    "hfcache_dir": OptionInfo(hfcache_dir, "Folder for Huggingface cache", folder=True),
    "vae_dir": OptionInfo(os.path.join(paths.models_path, 'VAE'), "Folder with VAE files", folder=True),
    "unet_dir": OptionInfo(os.path.join(paths.models_path, 'UNET'), "Folder with UNET files", folder=True),
    "te_dir": OptionInfo(os.path.join(paths.models_path, 'Text-encoder'), "Folder with Text encoder files", folder=True),
    "lora_dir": OptionInfo(os.path.join(paths.models_path, 'Lora'), "Folder with LoRA network(s)", folder=True),
    "styles_dir": OptionInfo(os.path.join(paths.data_path, 'styles.csv'), "File or Folder with user-defined styles", folder=True),
    "wildcards_dir": OptionInfo(os.path.join(paths.models_path, 'wildcards'), "Folder with user-defined wildcards", folder=True),
    "embeddings_dir": OptionInfo(os.path.join(paths.models_path, 'embeddings'), "Folder with textual inversion embeddings", folder=True),
    "hypernetwork_dir": OptionInfo(os.path.join(paths.models_path, 'hypernetworks'), "Folder with Hypernetwork models", folder=True),
    "control_dir": OptionInfo(os.path.join(paths.models_path, 'control'), "Folder with Control models", folder=True),
    "yolo_dir": OptionInfo(os.path.join(paths.models_path, 'yolo'), "Folder with Yolo models", folder=True),
    "codeformer_models_path": OptionInfo(os.path.join(paths.models_path, 'Codeformer'), "Folder with codeformer models", folder=True),
    "gfpgan_models_path": OptionInfo(os.path.join(paths.models_path, 'GFPGAN'), "Folder with GFPGAN models", folder=True),
    "esrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'ESRGAN'), "Folder with ESRGAN models", folder=True),
    "bsrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'BSRGAN'), "Folder with BSRGAN models", folder=True),
    "realesrgan_models_path": OptionInfo(os.path.join(paths.models_path, 'RealESRGAN'), "Folder with RealESRGAN models", folder=True),
    "scunet_models_path": OptionInfo(os.path.join(paths.models_path, 'SCUNet'), "Folder with SCUNet models", folder=True),
    "swinir_models_path": OptionInfo(os.path.join(paths.models_path, 'SwinIR'), "Folder with SwinIR models", folder=True),
    "ldsr_models_path": OptionInfo(os.path.join(paths.models_path, 'LDSR'), "Folder with LDSR models", folder=True),
    "clip_models_path": OptionInfo(os.path.join(paths.models_path, 'CLIP'), "Folder with CLIP models", folder=True),
    "other_paths_sep_options": OptionInfo("<h2>Other paths</h2>", "", gr.HTML),
    "openvino_cache_path": OptionInfo('cache', "Directory for OpenVINO cache", folder=True),
    "accelerate_offload_path": OptionInfo('cache/accelerate', "Directory for disk offload with Accelerate", folder=True),
    "onnx_cached_models_path": OptionInfo(os.path.join(paths.models_path, 'ONNX', 'cache'), "Folder with ONNX cached models", folder=True),
    "onnx_temp_dir": OptionInfo(os.path.join(paths.models_path, 'ONNX', 'temp'), "Directory for ONNX conversion and Olive optimization process", folder=True),
    "temp_dir": OptionInfo("", "Directory for temporary images; leave empty for default", folder=True),
    "clean_temp_dir_at_start": OptionInfo(True, "Cleanup non-default temporary directory when starting webui"),
}))

options_templates.update(options_section(('saving-images', "Image Options"), {
    "keep_incomplete": OptionInfo(True, "Keep incomplete images"),
    "samples_save": OptionInfo(True, "Save all generated images"),
    "samples_format": OptionInfo('jpg', 'File format', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2", "jxl"]}),
    "jpeg_quality": OptionInfo(90, "Image quality", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "img_max_size_mp": OptionInfo(1000, "Maximum image size (MP)", gr.Slider, {"minimum": 100, "maximum": 2000, "step": 1}),
    "webp_lossless": OptionInfo(False, "WebP lossless compression"),
    "save_selected_only": OptionInfo(True, "Save only saves selected image"),
    "include_mask": OptionInfo(False, "Include mask in outputs"),
    "samples_save_zip": OptionInfo(True, "Create ZIP archive"),
    "image_background": OptionInfo("#000000", "Resize background color", gr.ColorPicker, {}),

    "image_sep_metadata": OptionInfo("<h2>Metadata/Logging</h2>", "", gr.HTML),
    "image_metadata": OptionInfo(True, "Include metadata"),
    "save_txt": OptionInfo(False, "Create image info text file"),
    "save_log_fn": OptionInfo("", "Append image info JSON file", component_args=hide_dirs),
    "image_sep_grid": OptionInfo("<h2>Grid Options</h2>", "", gr.HTML),
    "grid_save": OptionInfo(True, "Save all generated image grids"),
    "grid_format": OptionInfo('jpg', 'File format', gr.Dropdown, {"choices": ["jpg", "png", "webp", "tiff", "jp2", "jxl"]}),
    "n_rows": OptionInfo(-1, "Row count", gr.Slider, {"minimum": -1, "maximum": 16, "step": 1}),
    "grid_background": OptionInfo("#000000", "Grid background color", gr.ColorPicker, {}),
    "font": OptionInfo("", "Font file"),
    "font_color": OptionInfo("#FFFFFF", "Font color", gr.ColorPicker, {}),

    "image_sep_browser": OptionInfo("<h2>Image Gallery</h2>", "", gr.HTML),
    "browser_cache": OptionInfo(True, "Use image gallery cache"),
    "browser_folders": OptionInfo("", "Additional image browser folders"),
    "browser_fixed_width": OptionInfo(False, "Use fixed width thumbnails"),
    "viewer_show_metadata": OptionInfo(True, "Show metadata in full screen image browser"),

    "save_sep_options": OptionInfo("<h2>Intermediate Image Saving</h2>", "", gr.HTML),
    "save_init_img": OptionInfo(False, "Save init images"),
    "save_images_before_highres_fix": OptionInfo(False, "Save image before hires"),
    "save_images_before_refiner": OptionInfo(False, "Save image before refiner"),
    "save_images_before_detailer": OptionInfo(False, "Save image before detailer"),
    "save_images_before_color_correction": OptionInfo(False, "Save image before color correction"),
    "save_mask": OptionInfo(False, "Save inpainting mask"),
    "save_mask_composite": OptionInfo(False, "Save inpainting masked composite"),
    "gradio_skip_video": OptionInfo(False, "Do not display video output in UI"),

    "image_sep_watermark": OptionInfo("<h2>Watermarking</h2>", "", gr.HTML),
    "image_watermark_enabled": OptionInfo(False, "Include invisible watermark"),
    "image_watermark": OptionInfo('', "Invisible watermark string"),
    "image_watermark_position": OptionInfo('none', 'Image watermark position', gr.Dropdown, {"choices": ["none", "top/left", "top/right", "bottom/left", "bottom/right", "center", "random"]}),
    "image_watermark_image": OptionInfo('', "Image watermark file"),
}))

options_templates.update(options_section(('saving-paths', "Image Paths"), {
    "saving_sep_images": OptionInfo("<h2>Save Options</h2>", "", gr.HTML),
    "save_images_add_number": OptionInfo(True, "Numbered filenames", component_args=hide_dirs),
    "use_original_name_batch": OptionInfo(True, "Batch uses original name"),
    "save_to_dirs": OptionInfo(False, "Save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("[date]", "Directory name pattern", component_args=hide_dirs),
    "samples_filename_pattern": OptionInfo("[seq]-[model_name]-[prompt_words]", "Images filename pattern", component_args=hide_dirs),
    "directories_max_prompt_words": OptionInfo(8, "Max words", gr.Slider, {"minimum": 1, "maximum": 99, "step": 1, **hide_dirs}),

    "outdir_sep_dirs": OptionInfo("<h2>Folders</h2>", "", gr.HTML),
    "outdir_samples": OptionInfo("", "Images folder", component_args=hide_dirs, folder=True),
    "outdir_txt2img_samples": OptionInfo("outputs/text", 'Folder for text generate', component_args=hide_dirs, folder=True),
    "outdir_img2img_samples": OptionInfo("outputs/image", 'Folder for image generate', component_args=hide_dirs, folder=True),
    "outdir_control_samples": OptionInfo("outputs/control", 'Folder for control generate', component_args=hide_dirs, folder=True),
    "outdir_extras_samples": OptionInfo("outputs/extras", 'Folder for processed images', component_args=hide_dirs, folder=True),
    "outdir_save": OptionInfo("outputs/save", "Folder for manually saved images", component_args=hide_dirs, folder=True),
    "outdir_video": OptionInfo("outputs/video", "Folder for videos", component_args=hide_dirs, folder=True),
    "outdir_init_images": OptionInfo("outputs/init-images", "Folder for init images", component_args=hide_dirs, folder=True),

    "outdir_sep_grids": OptionInfo("<h2>Grids</h2>", "", gr.HTML),
    "outdir_grids": OptionInfo("", "Grids folder", component_args=hide_dirs, folder=True),
    "outdir_txt2img_grids": OptionInfo("outputs/grids", 'Folder for txt2img grids', component_args=hide_dirs, folder=True),
    "outdir_img2img_grids": OptionInfo("outputs/grids", 'Folder for img2img grids', component_args=hide_dirs, folder=True),
    "outdir_control_grids": OptionInfo("outputs/grids", 'Folder for control grids', component_args=hide_dirs, folder=True),
}))

options_templates.update(options_section(('ui', "User Interface"), {
    "theme_type": OptionInfo("Standard", "Theme type", gr.Radio, {"choices": ["Modern", "Standard", "None"]}),
    "theme_style": OptionInfo("Auto", "Theme mode", gr.Radio, {"choices": ["Auto", "Dark", "Light"]}),
    "gradio_theme": OptionInfo("black-teal", "UI theme", gr.Dropdown, lambda: {"choices": theme.list_themes()}, refresh=theme.refresh_themes),
    "autolaunch": OptionInfo(False, "Autolaunch browser upon startup"),
    "font_size": OptionInfo(14, "Font size", gr.Slider, {"minimum": 8, "maximum": 32, "step": 1, "visible": True}),
    "aspect_ratios": OptionInfo("1:1, 4:3, 3:2, 16:9, 16:10, 21:9, 2:3, 3:4, 9:16, 10:16, 9:21", "Allowed aspect ratios"),
    "motd": OptionInfo(False, "Show MOTD"),
    "compact_view": OptionInfo(False, "Compact view"),
    "return_grid": OptionInfo(True, "Show grid in results"),
    "return_mask": OptionInfo(False, "Inpainting include greyscale mask in results"),
    "return_mask_composite": OptionInfo(False, "Inpainting include masked composite in results"),
    "disable_weights_auto_swap": OptionInfo(True, "Do not change selected model when reading generation parameters"),
    "send_seed": OptionInfo(True, "Send seed when sending prompt or image to other interface"),
    "send_size": OptionInfo(True, "Send size when sending prompt or image to another interface"),
    "quicksettings_list": OptionInfo(["sd_model_checkpoint"], "Quicksettings list", gr.Dropdown, lambda: {"multiselect":True, "choices": list(opts.data_labels.keys())}),
}))

options_templates.update(options_section(('live-preview', "Live Previews"), {
    "show_progress_every_n_steps": OptionInfo(1, "Live preview display period", gr.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
    "show_progress_type": OptionInfo("Approximate", "Live preview method", gr.Radio, {"choices": ["Simple", "Approximate", "TAESD", "Full VAE"]}),
    "live_preview_refresh_period": OptionInfo(500, "Progress update period", gr.Slider, {"minimum": 0, "maximum": 5000, "step": 25}),
    "live_preview_taesd_layers": OptionInfo(3, "TAESD decode layers", gr.Slider, {"minimum": 1, "maximum": 3, "step": 1}),
    "live_preview_downscale": OptionInfo(True, "Downscale high resolution live previews"),
    "logmonitor_show": OptionInfo(True, "Show log view"),
    "logmonitor_refresh_period": OptionInfo(5000, "Log view update period", gr.Slider, {"minimum": 0, "maximum": 30000, "step": 25}),
    "notification_audio_enable": OptionInfo(False, "Play a notification upon completion"),
    "notification_audio_path": OptionInfo("html/notification.mp3","Path to notification sound", component_args=hide_dirs, folder=True),
}))

options_templates.update(options_section(('sampler-params', "Sampler Settings"), {
    "show_samplers": OptionInfo([], "Show samplers in user interface", gr.CheckboxGroup, lambda: {"choices": [x.name for x in list_samplers()]}),
    'eta_noise_seed_delta': OptionInfo(0, "Noise seed delta (eta)", gr.Number, {"precision": 0}),
    "scheduler_eta": OptionInfo(1.0, "Noise multiplier (eta)", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "schedulers_solver_order": OptionInfo(0, "Solver order (where", gr.Slider, {"minimum": 0, "maximum": 5, "step": 1, "visible": False}),

    # managed from ui.py for backend original
    "schedulers_brownian_noise": OptionInfo(True, "Use Brownian noise", gr.Checkbox, {"visible": False}),
    "schedulers_discard_penultimate": OptionInfo(True, "Discard penultimate sigma", gr.Checkbox, {"visible": False}),
    "schedulers_use_karras": OptionInfo(True, "Use Karras sigmas", gr.Checkbox, {"visible": False}),

    "schedulers_use_loworder": OptionInfo(True, "Use simplified solvers in final steps", gr.Checkbox, {"visible": False}),
    "schedulers_prediction_type": OptionInfo("default", "Override model prediction type", gr.Radio, {"choices": ['default', 'epsilon', 'sample', 'v_prediction'], "visible": False}),
    "schedulers_sigma": OptionInfo("default", "Sigma algorithm", gr.Radio, {"choices": ['default', 'karras', 'exponential', 'polyexponential'], "visible": False}),

    # managed from ui.py for backend diffusers
    "schedulers_beta_schedule": OptionInfo("default", "Beta schedule", gr.Dropdown, {"choices": ['default', 'linear', 'scaled_linear', 'squaredcos_cap_v2'], "visible": False}),
    "schedulers_use_thresholding": OptionInfo(False, "Use dynamic thresholding", gr.Checkbox, {"visible": False}),
    "schedulers_timestep_spacing": OptionInfo("default", "Timestep spacing", gr.Dropdown, {"choices": ['default', 'linspace', 'leading', 'trailing'], "visible": False}),
    'schedulers_timesteps': OptionInfo('', "Timesteps", gr.Textbox, {"visible": False}),
    "schedulers_rescale_betas": OptionInfo(False, "Rescale betas with zero terminal SNR", gr.Checkbox, {"visible": False}),
    'schedulers_beta_start': OptionInfo(0, "Beta start", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001, "visible": native}),
    'schedulers_beta_end': OptionInfo(0, "Beta end", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.00001, "visible": native}),
    'schedulers_timesteps_range': OptionInfo(1000, "Timesteps range", gr.Slider, {"minimum": 250, "maximum": 4000, "step": 1, "visible": native}),
    'schedulers_shift': OptionInfo(0, "Sampler shift", gr.Slider, {"minimum": 0.1, "maximum": 10, "step": 0.1, "visible": native}),
    'schedulers_dynamic_shift': OptionInfo(True, "Sampler dynamic shift", gr.Checkbox, {"visible": native}),

    # managed from ui.py for backend original k-diffusion
    "always_batch_cond_uncond": OptionInfo(False, "Disable conditional batching", gr.Checkbox, {"visible": not native}),
    "enable_quantization": OptionInfo(True, "Use quantization", gr.Checkbox, {"visible": not native}),
    's_churn': OptionInfo(0.0, "Sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": not native}),
    's_min_uncond': OptionInfo(0.0, "Sigma negative guidance minimum ", gr.Slider, {"minimum": 0.0, "maximum": 4.0, "step": 0.01, "visible": not native}),
    's_tmin':  OptionInfo(0.0, "Sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": not native}),
    's_noise': OptionInfo(1.0, "Sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": not native}),
    's_min':  OptionInfo(0.0, "Sigma min",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": not native}),
    's_max':  OptionInfo(0.0, "Sigma max",  gr.Slider, {"minimum": 0.0, "maximum": 100.0, "step": 1.0, "visible": not native}),
    "schedulers_sep_compvis": OptionInfo("<h2>CompVis Config</h2>", "", gr.HTML, {"visible": not native}),
    'uni_pc_variant': OptionInfo("bh2", "UniPC variant", gr.Radio, {"choices": ["bh1", "bh2", "vary_coeff"], "visible": not native}),
    'uni_pc_skip_type': OptionInfo("time_uniform", "UniPC skip type", gr.Radio, {"choices": ["time_uniform", "time_quadratic", "logSNR"], "visible": not native}),
    "ddim_discretize": OptionInfo('uniform', "DDIM discretize img2img", gr.Radio, {"choices": ['uniform', 'quad'], "visible": not native}),
}))

options_templates.update(options_section(('postprocessing', "Postprocessing"), {
    'postprocessing_enable_in_main_ui': OptionInfo([], "Additional postprocessing operations", gr.Dropdown, lambda: {"multiselect":True, "choices": [x.name for x in shared_items.postprocessing_scripts()]}),
    'postprocessing_operation_order': OptionInfo([], "Postprocessing operation order", gr.Dropdown, lambda: {"multiselect":True, "choices": [x.name for x in shared_items.postprocessing_scripts()], "visible": False }),

    "postprocessing_sep_img2img": OptionInfo("<h2>Inpaint</h2>", "", gr.HTML),
    "img2img_color_correction": OptionInfo(False, "Apply color correction"),
    "mask_apply_overlay": OptionInfo(True, "Apply mask as overlay"),
    "img2img_background_color": OptionInfo("#ffffff", "Image transparent color fill", gr.ColorPicker, {}),
    "inpainting_mask_weight": OptionInfo(1.0, "Inpainting conditioning mask strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "initial_noise_multiplier": OptionInfo(1.0, "Noise multiplier for image processing", gr.Slider, {"minimum": 0.1, "maximum": 1.5, "step": 0.01, "visible": not native}),
    "img2img_extra_noise": OptionInfo(0.0, "Extra noise multiplier for img2img", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01, "visible": not native}),

    "postprocessing_sep_detailer": OptionInfo("<h2>Detailer</h2>", "", gr.HTML),
    "detailer_model": OptionInfo("Detailer", "Detailer model", gr.Radio, lambda: {"choices": [x.name() for x in detailers], "visible": False}),
    "detailer_classes": OptionInfo("", "Detailer classes", gr.Textbox, { "visible": False}),
    "detailer_conf": OptionInfo(0.6, "Min confidence", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_max": OptionInfo(2, "Max detected", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1, "visible": False}),
    "detailer_iou": OptionInfo(0.5, "Max overlap", gr.Slider, {"minimum": 0, "maximum": 1.0, "step": 0.05, "visible": False}),
    "detailer_min_size": OptionInfo(0.0, "Min object size", gr.Slider, {"minimum": 0.1, "maximum": 1, "step": 0.05, "visible": False}),
    "detailer_max_size": OptionInfo(1.0, "Max object size", gr.Slider, {"minimum": 0.1, "maximum": 1, "step": 0.05, "visible": False}),
    "detailer_padding": OptionInfo(20, "Item padding", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1, "visible": False}),
    "detailer_blur": OptionInfo(10, "Item edge blur", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1, "visible": False}),
    "detailer_strength": OptionInfo(0.5, "Detailer strength", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01, "visible": False}),
    "detailer_models": OptionInfo(['face-yolo8n'], "Detailer models", gr.Dropdown, lambda: {"multiselect":True, "choices": list(yolo.list), "visible": False}),
    "detailer_unload": OptionInfo(False, "Move detailer model to CPU when complete"),
    "detailer_augment": OptionInfo(True, "Detailer use model augment"),

    "postprocessing_sep_face_restore": OptionInfo("<h2>Face Restore</h2>", "", gr.HTML),
    "face_restoration_model": OptionInfo("None", "Face restoration", gr.Radio, lambda: {"choices": ['None'] + [x.name() for x in face_restorers]}),
    "code_former_weight": OptionInfo(0.2, "CodeFormer weight parameter", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),

    "postprocessing_sep_upscalers": OptionInfo("<h2>Upscaling</h2>", "", gr.HTML),
    "upscaler_unload": OptionInfo(False, "Unload upscaler after processing"),
    "upscaler_tile_size": OptionInfo(192, "Upscaler tile size", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "upscaler_tile_overlap": OptionInfo(8, "Upscaler tile overlap", gr.Slider, {"minimum": 0, "maximum": 64, "step": 1}),
}))

options_templates.update(options_section(('control', "Control Options"), {
    "control_max_units": OptionInfo(4, "Maximum number of units", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}),
    "control_tiles": OptionInfo("1x1, 1x2, 1x3, 1x4, 2x1, 2x1, 2x2, 2x3, 2x4, 3x1, 3x2, 3x3, 3x4, 4x1, 4x2, 4x3, 4x4", "Tiling options"),
    "control_move_processor": OptionInfo(False, "Processor move to CPU after use"),
    "control_unload_processor": OptionInfo(False, "Processor unload after use"),
}))

options_templates.update(options_section(('interrogate', "Interrogate"), {
    "interrogate_keep_models_in_memory": OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_return_ranks": OptionInfo(True, "Interrogate: include ranks of model tags matches in results"),
    "interrogate_clip_num_beams": OptionInfo(1, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(32, "Interrogate: minimum description length", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(192, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_skip_categories": OptionInfo(["artists", "movements", "flavors"], "Interrogate: skip categories", gr.CheckboxGroup, lambda: {"choices": modules.interrogate.category_types()}, refresh=modules.interrogate.category_types),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.65, "Interrogate: deepbooru score threshold", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha": OptionInfo(False, "Interrogate: deepbooru sort alphabetically"),
    "deepbooru_use_spaces": OptionInfo(False, "Use spaces for tags in deepbooru"),
    "deepbooru_escape": OptionInfo(True, "Escape brackets in deepbooru"),
    "deepbooru_filter_tags": OptionInfo("", "Filter out tags from deepbooru output"),
}))

options_templates.update(options_section(('huggingface', "Huggingface"), {
    "huggingface_sep": OptionInfo("<h2>Huggingface</h2>", "", gr.HTML),
    "diffuser_cache_config": OptionInfo(True, "Use cached model config when available"),
    "huggingface_token": OptionInfo('', 'HuggingFace token'),
    "diffusers_model_load_variant": OptionInfo("default", "Preferred Model variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "diffusers_vae_load_variant": OptionInfo("default", "Preferred VAE variant", gr.Radio, {"choices": ['default', 'fp32', 'fp16']}),
    "custom_diffusers_pipeline": OptionInfo('', 'Load custom Diffusers pipeline'),
}))

options_templates.update(options_section(('extra_networks', "Networks"), {
    "extra_networks_sep1": OptionInfo("<h2>Networks UI</h2>", "", gr.HTML),
    "extra_networks_show": OptionInfo(True, "UI show on startup"),
    "extra_networks": OptionInfo(["All"], "Available networks", gr.Dropdown, lambda: {"multiselect":True, "choices": ['All'] + [en.title for en in extra_networks]}),
    "extra_networks_sort": OptionInfo("Default", "Sort order", gr.Dropdown, {"choices": ['Default', 'Name [A-Z]', 'Name [Z-A]', 'Date [Newest]', 'Date [Oldest]', 'Size [Largest]', 'Size [Smallest]']}),
    "extra_networks_view": OptionInfo("gallery", "UI view", gr.Radio, {"choices": ["gallery", "list"]}),
    "extra_networks_card_cover": OptionInfo("sidebar", "UI position", gr.Radio, {"choices": ["cover", "inline", "sidebar"]}),
    "extra_networks_height": OptionInfo(0, "UI height (%)", gr.Slider, {"minimum": 0, "maximum": 100, "step": 1}), # set in ui_javascript
    "extra_networks_sidebar_width": OptionInfo(35, "UI sidebar width (%)", gr.Slider, {"minimum": 10, "maximum": 80, "step": 1}),
    "extra_networks_card_size": OptionInfo(160, "UI card size (px)", gr.Slider, {"minimum": 20, "maximum": 2000, "step": 1}),
    "extra_networks_card_square": OptionInfo(True, "UI disable variable aspect ratio"),
    "extra_networks_fetch": OptionInfo(True, "UI fetch network info on mouse-over"),
    "extra_network_skip_indexing": OptionInfo(False, "Build info on first access", gr.Checkbox),

    "extra_networks_model_sep": OptionInfo("<h2>Models</h2>", "", gr.HTML),
    "extra_network_reference": OptionInfo(False, "Use reference values when available", gr.Checkbox),

    "extra_networks_lora_sep": OptionInfo("<h2>LoRA</h2>", "", gr.HTML),
    "extra_networks_default_multiplier": OptionInfo(1.0, "Default strength", gr.Slider, {"minimum": 0.0, "maximum": 2.0, "step": 0.01}),
    "lora_preferred_name": OptionInfo("filename", "LoRA preferred name", gr.Radio, {"choices": ["filename", "alias"], "visible": False}),
    "lora_add_hashes_to_infotext": OptionInfo(False, "LoRA add hash info to metadata"),
    "lora_fuse_diffusers": OptionInfo(True, "LoRA fuse directly to model"),
    "lora_legacy": OptionInfo(not native, "LoRA load using legacy method"),
    "lora_force_diffusers": OptionInfo(False if not cmd_opts.use_openvino else True, "LoRA load using Diffusers method"),
    "lora_maybe_diffusers": OptionInfo(False, "LoRA load using Diffusers method for selected models"),
    "lora_apply_tags": OptionInfo(0, "LoRA auto-apply tags", gr.Slider, {"minimum": -1, "maximum": 32, "step": 1}),
    "lora_in_memory_limit": OptionInfo(0, "LoRA memory cache", gr.Slider, {"minimum": 0, "maximum": 24, "step": 1}),
    "lora_quant": OptionInfo("NF4","LoRA precision when quantized", gr.Radio, {"choices": ["NF4", "FP4"]}),

    "extra_networks_styles_sep": OptionInfo("<h2>Styles</h2>", "", gr.HTML),
    "extra_networks_styles": OptionInfo(True, "Show built-in styles"),

    "extra_networks_embed_sep": OptionInfo("<h2>Embeddings</h2>", "", gr.HTML),
    "diffusers_enable_embed": OptionInfo(True, "Enable embeddings support", gr.Checkbox, {"visible": native}),
    "diffusers_convert_embed": OptionInfo(False, "Auto-convert SD15 embeddings to SDXL", gr.Checkbox, {"visible": native}),

    "extra_networks_wildcard_sep": OptionInfo("<h2>Wildcards</h2>", "", gr.HTML),
    "wildcards_enabled": OptionInfo(True, "Enable file wildcards support"),
}))

options_templates.update(options_section((None, "Internal options"), {
    "diffusers_version": OptionInfo("", "Diffusers version", gr.Textbox, {"visible": False}),
    "disabled_extensions": OptionInfo([], "Disable these extensions"),
    "sd_checkpoint_hash": OptionInfo("", "SHA256 hash of the current checkpoint"),
    "tooltips": OptionInfo("UI Tooltips", "UI tooltips", gr.Radio, {"choices": ["None", "Browser default", "UI tooltips"], "visible": False}),
}))

options_templates.update(options_section((None, "Hidden options"), {
    "batch_cond_uncond": OptionInfo(True, "Do conditional and unconditional denoising in one batch", gr.Checkbox, {"visible": False}),
    "CLIP_stop_at_last_layers": OptionInfo(1, "Clip skip", gr.Slider, {"minimum": 1, "maximum": 8, "step": 1, "visible": False}),
    "dataset_filename_join_string": OptionInfo(" ", "Filename join string", gr.Textbox, { "visible": False }),
    "dataset_filename_word_regex": OptionInfo("", "Filename word regex", gr.Textbox, { "visible": False }),
    "diffusers_force_zeros": OptionInfo(False, "Force zeros for prompts when empty", gr.Checkbox, {"visible": False}),
    "disable_all_extensions": OptionInfo("none", "Disable all extensions (preserves the list of disabled extensions)", gr.Radio, {"choices": ["none", "user", "all"]}),
    "disable_nan_check": OptionInfo(True, "Disable NaN check", gr.Checkbox, {"visible": False}),
    "embeddings_templates_dir": OptionInfo("", "Embeddings train templates directory", gr.Textbox, { "visible": False }),
    "extra_networks_card_fit": OptionInfo("cover", "UI image contain method", gr.Radio, {"choices": ["contain", "cover", "fill"], "visible": False}),
    "grid_extended_filename": OptionInfo(True, "Add extended info to filename when saving grid", gr.Checkbox, {"visible": False}),
    "grid_save_to_dirs": OptionInfo(False, "Save grids to a subdirectory", gr.Checkbox, {"visible": False}),
    "hypernetwork_enabled": OptionInfo(False, "Enable Hypernetwork support", gr.Checkbox, {"visible": False}),
    "img2img_fix_steps": OptionInfo(False, "For image processing do exact number of steps as specified", gr.Checkbox, { "visible": False }),
    "interrogate_clip_dict_limit": OptionInfo(2048, "CLIP: maximum number of lines in text file", gr.Slider, { "visible": False }),
    "keyedit_delimiters": OptionInfo(r".,\/!?%^*;:{}=`~()", "Ctrl+up/down word delimiters", gr.Textbox, { "visible": False }),
    "keyedit_precision_attention": OptionInfo(0.1, "Ctrl+up/down precision when editing (attention:1.1)", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "keyedit_precision_extra": OptionInfo(0.05, "Ctrl+up/down precision when editing <extra networks:0.9>", gr.Slider, {"minimum": 0.01, "maximum": 0.2, "step": 0.001, "visible": False}),
    "live_preview_content": OptionInfo("Combined", "Live preview subject", gr.Radio, {"choices": ["Combined", "Prompt", "Negative prompt"], "visible": False}),
    "live_previews_enable": OptionInfo(True, "Show live previews", gr.Checkbox, {"visible": False}),
    "lora_functional": OptionInfo(False, "Use Kohya method for handling multiple LoRA", gr.Checkbox, { "visible": False }),
    "lyco_dir": OptionInfo(os.path.join(paths.models_path, 'LyCORIS'), "Folder with LyCORIS network(s)", gr.Text, {"visible": False}),
    "model_reuse_dict": OptionInfo(False, "Reuse loaded model dictionary", gr.Checkbox, {"visible": False}),
    "pad_cond_uncond": OptionInfo(True, "Pad prompt and negative prompt to be same length", gr.Checkbox, {"visible": False}),
    "pin_memory": OptionInfo(True, "Pin training dataset to memory", gr.Checkbox, { "visible": False }),
    "save_optimizer_state": OptionInfo(False, "Save resumable optimizer state when training", gr.Checkbox, { "visible": False }),
    "save_training_settings_to_txt": OptionInfo(True, "Save training settings to a text file", gr.Checkbox, { "visible": False }),
    "sd_disable_ckpt": OptionInfo(False, "Disallow models in ckpt format", gr.Checkbox, {"visible": False}),
    "sd_hypernetwork": OptionInfo("None", "Add hypernetwork to prompt", gr.Dropdown, { "choices": ["None"], "visible": False }),
    "sd_lora": OptionInfo("", "Add LoRA to prompt", gr.Textbox, {"visible": False}),
    "sd_vae_checkpoint_cache": OptionInfo(0, "Cached VAEs", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1, "visible": False}),
    "show_progress_grid": OptionInfo(True, "Show previews as a grid", gr.Checkbox, {"visible": False}),
    "show_progressbar": OptionInfo(True, "Show progressbar", gr.Checkbox, {"visible": False}),
    "training_enable_tensorboard": OptionInfo(False, "Enable tensorboard logging", gr.Checkbox, { "visible": False }),
    "training_image_repeats_per_epoch": OptionInfo(1, "Image repeats per epoch", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1, "visible": False }),
    "training_tensorboard_flush_every": OptionInfo(120, "Tensorboard flush period", gr.Number, { "visible": False }),
    "training_tensorboard_save_images": OptionInfo(False, "Save generated images within tensorboard", gr.Checkbox, { "visible": False }),
    "training_write_csv_every": OptionInfo(0, "Save loss CSV file every n steps", gr.Number, { "visible": False }),
    "ui_scripts_reorder": OptionInfo("", "UI scripts order", gr.Textbox, { "visible": False }),
    "unload_models_when_training": OptionInfo(False, "Move VAE and CLIP to RAM when training", gr.Checkbox, { "visible": False }),
    "upscaler_for_img2img": OptionInfo("None", "Default upscaler for image resize operations", gr.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers], "visible": False}, refresh=refresh_upscalers),
    "use_save_to_dirs_for_ui": OptionInfo(False, "Save images to a subdirectory when using Save button", gr.Checkbox, {"visible": False}),
    "use_upscaler_name_as_suffix": OptionInfo(True, "Use upscaler as suffix", gr.Checkbox, {"visible": False}),
}))

options_templates.update()


class Options:
    data = None
    data_labels = options_templates
    filename = None
    typemap = {int: float}

    def __init__(self):
        self.data = {k: v.default for k, v in self.data_labels.items()}

    def __setattr__(self, key, value): # pylint: disable=inconsistent-return-statements
        if self.data is not None:
            if key in self.data or key in self.data_labels:
                if cmd_opts.freeze:
                    log.warning(f'Settings are frozen: {key}')
                    return
                if cmd_opts.hide_ui_dir_config and key in restricted_opts:
                    log.warning(f'Settings key is restricted: {key}')
                    return
                self.data[key] = value
                return
        return super(Options, self).__setattr__(key, value) # pylint: disable=super-with-arguments

    def get(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super(Options, self).__getattribute__(item) # pylint: disable=super-with-arguments

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]
        if item in self.data_labels:
            return self.data_labels[item].default
        return super(Options, self).__getattribute__(item) # pylint: disable=super-with-arguments

    def set(self, key, value):
        """sets an option and calls its onchange callback, returning True if the option changed and False otherwise"""
        oldval = self.data.get(key, None)
        if oldval is None:
            oldval = self.data_labels[key].default
        if oldval == value:
            return False
        try:
            setattr(self, key, value)
        except RuntimeError:
            return False
        if self.data_labels[key].onchange is not None:
            try:
                self.data_labels[key].onchange()
            except Exception as err:
                log.error(f'Error in onchange callback: {key} {value} {err}')
                errors.display(err, 'Error in onchange callback')
                setattr(self, key, oldval)
                return False
        return True

    def get_default(self, key):
        """returns the default value for the key"""
        data_label = self.data_labels.get(key)
        return data_label.default if data_label is not None else None

    def save_atomic(self, filename=None, silent=False):
        if filename is None:
            filename = self.filename
        if cmd_opts.freeze:
            log.warning(f'Setting: fn="{filename}" save disabled')
            return
        try:
            # output = json.dumps(self.data, indent=2)
            diff = {}
            unused_settings = []

            if os.environ.get('SD_CONFIG_DEBUG', None) is not None:
                log.debug('Settings: user')
                for k, v in self.data.items():
                    log.trace(f'  Config: item={k} value={v} default={self.data_labels[k].default if k in self.data_labels else None}')
                log.debug('Settings: defaults')
                for k in self.data_labels.keys():
                    log.trace(f'  Setting: item={k} default={self.data_labels[k].default}')

            for k, v in self.data.items():
                if k in self.data_labels:
                    if type(v) is list:
                        diff[k] = v
                    if self.data_labels[k].default != v:
                        diff[k] = v
                else:
                    if k not in compatibility_opts:
                        diff[k] = v
                        if not k.startswith('uiux_'):
                            unused_settings.append(k)
            writefile(diff, filename, silent=silent)
            if len(unused_settings) > 0:
                log.debug(f"Settings: unused={unused_settings}")
        except Exception as err:
            log.error(f'Settings: fn="{filename}" {err}')

    def save(self, filename=None, silent=False):
        threading.Thread(target=self.save_atomic, args=(filename, silent)).start()

    def same_type(self, x, y):
        if x is None or y is None:
            return True
        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))
        return type_x == type_y

    def load(self, filename=None):
        if filename is None:
            filename = self.filename
        if not os.path.isfile(filename):
            log.debug(f'Settings: fn="{filename}" created')
            self.save(filename)
            return
        self.data = readfile(filename, lock=True)
        if self.data.get('quicksettings') is not None and self.data.get('quicksettings_list') is None:
            self.data['quicksettings_list'] = [i.strip() for i in self.data.get('quicksettings').split(',')]
        unknown_settings = []
        for k, v in self.data.items():
            info: OptionInfo = self.data_labels.get(k, None)
            if info is not None:
                if not info.validate(k, v):
                    self.data[k] = info.default
            if info is not None and not self.same_type(info.default, v):
                log.warning(f"Setting validation: {k}={v} ({type(v).__name__} expected={type(info.default).__name__})")
                self.data[k] = info.default
            if info is None and k not in compatibility_opts and not k.startswith('uiux_'):
                unknown_settings.append(k)
        if len(unknown_settings) > 0:
            log.warning(f"Setting validation: unknown={unknown_settings}")

    def onchange(self, key, func, call=True):
        item = self.data_labels.get(key)
        item.onchange = func
        if call:
            func()

    def dumpjson(self):
        d = {k: self.data.get(k, self.data_labels.get(k).default) for k in self.data_labels.keys()}
        metadata = {
            k: {
                "is_stored": k in self.data and self.data[k] != self.data_labels[k].default, # pylint: disable=unnecessary-dict-index-lookup
                "tab_name": v.section[0]
            } for k, v in self.data_labels.items()
        }
        return json.dumps({"values": d, "metadata": metadata})

    def add_option(self, key, info):
        self.data_labels[key] = info

    def reorder(self):
        """reorder settings so that all items related to section always go together"""
        section_ids = {}
        settings_items = self.data_labels.items()
        for _k, item in settings_items:
            if item.section not in section_ids:
                section_ids[item.section] = len(section_ids)
        self.data_labels = dict(sorted(settings_items, key=lambda x: section_ids[x[1].section]))

    def cast_value(self, key, value):
        """casts an arbitrary to the same type as this setting's value with key
        Example: cast_value("eta_noise_seed_delta", "12") -> returns 12 (an int rather than str)
        """
        if value is None:
            return None
        default_value = self.data_labels[key].default
        if default_value is None:
            default_value = getattr(self, key, None)
        if default_value is None:
            return None
        expected_type = type(default_value)
        if expected_type == bool and value == "False":
            value = False
        elif expected_type == type(value):
            pass
        else:
            value = expected_type(value)
        return value

profiler = None
opts = Options()
config_filename = cmd_opts.config
opts.load(config_filename)
cmd_opts = cmd_args.settings_args(opts, cmd_opts)
if cmd_opts.use_xformers:
    opts.data['cross_attention_optimization'] = 'xFormers'
opts.data['uni_pc_lower_order_final'] = opts.schedulers_use_loworder # compatibility
opts.data['uni_pc_order'] = max(2, opts.schedulers_solver_order) # compatibility
log.info(f'Engine: backend={backend} compute={devices.backend} device={devices.get_optimal_device_name()} attention="{opts.cross_attention_optimization}" mode={devices.inference_context.__name__}')
if not native:
    log.warning('Backend=original is in maintainance-only mode')
    opts.data['diffusers_offload_mode'] = 'none'

prompt_styles = modules.styles.StyleDatabase(opts)
reference_models = readfile(os.path.join('html', 'reference.json'))
cmd_opts.disable_extension_access = (cmd_opts.share or cmd_opts.listen or (cmd_opts.server_name or False)) and not cmd_opts.insecure
devices.args = cmd_opts
devices.opts = opts
devices.onnx = [opts.onnx_execution_provider]
devices.set_cuda_params()
if opts.onnx_cpu_fallback and 'CPUExecutionProvider' not in devices.onnx:
    devices.onnx.append('CPUExecutionProvider')
device = devices.device

batch_cond_uncond = opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram
mem_mon = modules.memmon.MemUsageMonitor("MemMon", devices.device)
history = history.History()
if devices.backend == "directml":
    directml_do_hijack()
elif devices.backend == "zluda":
    from modules.zluda import initialize_zluda
    initialize_zluda()
initialize_onnx()
try:
    log.info(f'Device: {print_dict(devices.get_gpu_info())}')
except Exception as ex:
    log.error(f'Device: {ex}')


class TotalTQDM: # compatibility with previous global-tqdm
    # import tqdm
    def __init__(self):
        pass
    def reset(self):
        pass
    def update(self):
        pass
    def updateTotal(self, new_total):
        pass
    def clear(self):
        pass
total_tqdm = TotalTQDM()


def restart_server(restart=True):
    if demo is None:
        return
    log.warning('Server shutdown requested')
    try:
        sys.tracebacklimit = 0
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stdout(stderr):
            demo.server.wants_restart = restart
            demo.server.should_exit = True
            demo.server.force_exit = True
            demo.close(verbose=False)
            demo.server.close()
            demo.fns = []
        time.sleep(1)
        sys.tracebacklimit = 100
        # os._exit(0)
    except (Exception, BaseException) as err:
        log.error(f'Server shutdown error: {err}')
    if restart:
        log.info('Server will restart')


def restore_defaults(restart=True):
    if os.path.exists(cmd_opts.config):
        log.info('Restoring server defaults')
        os.remove(cmd_opts.config)
    restart_server(restart)


def listdir(path):
    if not os.path.exists(path):
        return []
    mtime = os.path.getmtime(path)
    if path in dir_timestamps and mtime == dir_timestamps[path]:
        return dir_cache[path]
    else:
        dir_cache[path] = [os.path.join(path, f) for f in os.listdir(path)]
        dir_timestamps[path] = mtime
        return dir_cache[path]


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return
    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)
    for root, _dirs, files in os.walk(path, followlinks=True):
        for filename in files:
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext not in allowed_extensions:
                    continue
            yield os.path.join(root, filename)


def html_path(filename):
    return os.path.join(paths.script_path, "html", filename)


def html(filename):
    path = html_path(filename)
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()
    return ""


@lru_cache(maxsize=1)
def get_version():
    version = None
    if version is None:
        try:
            import subprocess
            res = subprocess.run('git log --pretty=format:"%h %ad" -1 --date=short', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            ver = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else '  '
            githash, updated = ver.split(' ')
            res = subprocess.run('git remote get-url origin', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            origin = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            res = subprocess.run('git rev-parse --abbrev-ref HEAD', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell=True, check=True)
            branch = res.stdout.decode(encoding = 'utf8', errors='ignore') if len(res.stdout) > 0 else ''
            version = {
                'app': 'sd.next',
                'updated': updated,
                'hash': githash,
                'url': origin.replace('\n', '') + '/tree/' + branch.replace('\n', '')
            }
        except Exception:
            version = { 'app': 'sd.next' }
    return version


def req(url_addr, headers = None, **kwargs):
    if headers is None:
        headers = { 'Content-type': 'application/json' }
    try:
        res = requests.get(url_addr, timeout=30, headers=headers, verify=False, allow_redirects=True, **kwargs)
    except Exception as err:
        log.error(f'HTTP request error: url={url_addr} {err}')
        res = { 'status_code': 500, 'text': f'HTTP request error: url={url_addr} {err}' }
        res = SimpleNamespace(**res)
    return res


sd_model: diffusers.DiffusionPipeline = None # dummy and overwritten by class
sd_refiner: diffusers.DiffusionPipeline = None # dummy and overwritten by class
sd_model_type: str = '' # dummy and overwritten by class
sd_refiner_type: str = '' # dummy and overwritten by class
sd_loaded: bool = False # dummy and overwritten by class
compiled_model_state = None
listfiles = listdir

from modules.modeldata import Shared # pylint: disable=ungrouped-imports
sys.modules[__name__].__class__ = Shared
