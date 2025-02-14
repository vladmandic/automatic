import os
import sys
import time
import contextlib
from functools import wraps
import torch
from modules import rocm
from modules.errors import log, display, install as install_traceback
from installer import install


debug = os.environ.get('SD_DEVICE_DEBUG', None) is not None
install_traceback() # traceback handler
opts = None # initialized in get_backend to avoid circular import
args = None # initialized in get_backend to avoid circular import
cuda_ok = torch.cuda.is_available() or (hasattr(torch, 'xpu') and torch.xpu.is_available())
inference_context = torch.no_grad
cpu = torch.device("cpu")

fp16_ok = None # set once by test_fp16
bf16_ok = None # set once by test_bf16

backend = None # set by get_backend
device = None # set by get_optimal_device
dtype = None # set by set_dtype
dtype_vae = None
dtype_unet = None
unet_needs_upcast = False # compatibility item
onnx = None
sdpa_original = None
sdpa_pre_dyanmic_atten = None
previous_oom = 0 # oom counter
if debug:
    log.info(f'Torch build config: {torch.__config__.show()}')
# set_cuda_sync_mode('block') # none/auto/spin/yield/block


def has_mps() -> bool:
    if sys.platform != "darwin":
        return False
    else:
        from modules import devices_mac # pylint: disable=ungrouped-imports
        return devices_mac.has_mps # pylint: disable=used-before-assignment


def has_xpu() -> bool:
    return bool(hasattr(torch, 'xpu') and torch.xpu.is_available())


def has_zluda() -> bool:
    if not cuda_ok:
        return False
    try:
        dev = torch.device("cuda")
        return torch.cuda.get_device_name(dev).endswith("[ZLUDA]")
    except Exception:
        return False


def get_backend(shared_cmd_opts):
    global args # pylint: disable=global-statement
    args = shared_cmd_opts
    if args.use_openvino:
        name = 'openvino'
    elif args.use_directml:
        name = 'directml'
    elif has_xpu():
        name = 'ipex'
    elif has_zluda():
        name = 'zluda'
    elif torch.cuda.is_available() and torch.version.cuda:
        name = 'cuda'
    elif torch.cuda.is_available() and torch.version.hip:
        name = 'rocm'
    elif sys.platform == 'darwin':
        name = 'mps'
    else:
        name = 'cpu'
    return name


def get_gpu_info():
    def get_driver():
        import subprocess
        if torch.cuda.is_available() and torch.version.cuda:
            try:
                result = subprocess.run('nvidia-smi --query-gpu=driver_version --format=csv,noheader', shell=True, check=False, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                version = result.stdout.decode(encoding="utf8", errors="ignore").strip()
                return version
            except Exception:
                return ''
        else:
            return ''

    def get_package_version(pkg: str):
        import pkg_resources
        spec = pkg_resources.working_set.by_key.get(pkg, None) # more reliable than importlib
        version = pkg_resources.get_distribution(pkg).version if spec is not None else ''
        return version

    if not torch.cuda.is_available():
        try:
            if backend == 'openvino':
                from modules.intel.openvino import get_openvino_device
                return {
                    'device': get_openvino_device(), # pylint: disable=used-before-assignment
                    'openvino': get_package_version("openvino"),
                }
            elif backend == 'directml':
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} n={torch.cuda.device_count()}',
                    'directml': get_package_version("torch-directml"),
                }
            else:
                return {}
        except Exception:
            return {}
    else:
        try:
            if backend == 'ipex':
                return {
                    'device': f'{torch.xpu.get_device_name(torch.xpu.current_device())} n={torch.xpu.device_count()}',
                    'ipex': get_package_version('intel-extension-for-pytorch'),
                }
            elif backend == 'cuda' or backend == 'zluda':
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} n={torch.cuda.device_count()} arch={torch.cuda.get_arch_list()[-1]} capability={torch.cuda.get_device_capability(device)}',
                    'cuda': torch.version.cuda,
                    'cudnn': torch.backends.cudnn.version(),
                    'driver': get_driver(),
                }
            elif backend == 'rocm':
                return {
                    'device': f'{torch.cuda.get_device_name(torch.cuda.current_device())} n={torch.cuda.device_count()}',
                    'hip': torch.version.hip,
                }
            else:
                return {
                    'device': 'unknown'
                }
        except Exception as ex:
            if debug:
                display(ex, 'Device exception')
            return { 'error': ex }


def extract_device_id(args, name): # pylint: disable=redefined-outer-name
    for x in range(len(args)):
        if name in args[x]:
            return args[x + 1]
    return None


def get_cuda_device_string():
    from modules.shared import cmd_opts
    if backend == 'ipex':
        if cmd_opts.device_id is not None:
            return f"xpu:{cmd_opts.device_id}"
        return "xpu"
    elif backend == 'directml' and torch.dml.is_available():
        if cmd_opts.device_id is not None:
            return f"privateuseone:{cmd_opts.device_id}"
        return torch.dml.get_device_string(torch.dml.default_device().index)
    else:
        if cmd_opts.device_id is not None:
            return f"cuda:{cmd_opts.device_id}"
        return "cuda"


def get_optimal_device_name():
    if cuda_ok or backend == 'directml':
        return get_cuda_device_string()
    if has_mps() and backend != 'openvino':
        return "mps"
    return "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


def get_device_for(task): # pylint: disable=unused-argument
    # if task in cmd_opts.use_cpu:
    #    log.debug(f'Forcing CPU for task: {task}')
    #    return cpu
    return get_optimal_device()


def torch_gc(force:bool=False, fast:bool=False, reason:str=None):
    def get_stats():
        mem_dict = memstats.memory_stats()
        gpu_dict = mem_dict.get('gpu', {})
        ram_dict = mem_dict.get('ram', {})
        oom = gpu_dict.get('oom', 0)
        ram = ram_dict.get('used', 0)
        if backend == "directml":
            gpu = torch.cuda.memory_allocated() / (1 << 30)
        else:
            gpu = gpu_dict.get('used', 0)
        used_gpu = round(100 * gpu / gpu_dict.get('total', 1)) if gpu_dict.get('total', 1) > 1 else 0
        used_ram = round(100 * ram / ram_dict.get('total', 1)) if ram_dict.get('total', 1) > 1 else 0
        return gpu, used_gpu, ram, used_ram, oom

    global previous_oom # pylint: disable=global-statement
    import gc
    from modules import timer, memstats
    from modules.shared import cmd_opts

    t0 = time.time()
    gpu, used_gpu, ram, _used_ram, oom = get_stats()
    threshold = 0 if (cmd_opts.lowvram and not cmd_opts.use_zluda) else opts.torch_gc_threshold
    collected = 0
    if reason is None and force:
        reason='force'
    if threshold == 0 or used_gpu >= threshold:
        force = True
        if reason is None:
            reason = 'threshold'
    if oom > previous_oom:
        previous_oom = oom
        log.warning(f'Torch GPU out-of-memory error: {memstats.memory_stats()}')
        force = True
        if reason is None:
            reason = 'oom'
    if force:
        # actual gc
        collected = gc.collect() if not fast else 0 # python gc
        if cuda_ok:
            try:
                with torch.cuda.device(get_cuda_device_string()):
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache() # cuda gc
                    torch.cuda.ipc_collect()
            except Exception:
                pass
    else:
        return gpu, ram
    t1 = time.time()
    timer.process.add('gc', t1 - t0)
    if fast:
        return gpu, ram

    new_gpu, new_used_gpu, new_ram, new_used_ram, oom = get_stats()
    before = { 'gpu': gpu, 'ram': ram }
    after = { 'gpu': new_gpu, 'ram': new_ram, 'oom': oom }
    utilization = { 'gpu': new_used_gpu, 'ram': new_used_ram }
    results = { 'gpu': round(gpu - new_gpu, 2), 'py': collected }
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    log.debug(f'GC: current={after} prev={before} load={utilization} gc={results} fn={fn} why={reason} time={t1-t0:.2f}')
    return new_gpu, new_ram


def set_cuda_sync_mode(mode):
    """
    Set the CUDA device synchronization mode: auto, spin, yield or block.
    auto: Chooses spin or yield depending on the number of available CPU cores.
    spin: Runs one CPU core per GPU at 100% to poll for completed operations.
    yield: Gives control to other threads between polling, if any are waiting.
    block: Lets the thread sleep until the GPU driver signals completion.
    """
    if mode == -1 or mode == 'none' or not cuda_ok:
        return
    try:
        import ctypes
        log.info(f'Torch CUDA sync: mode={mode}')
        torch.cuda.set_device(torch.device(get_optimal_device_name()))
        ctypes.CDLL('libcudart.so').cudaSetDeviceFlags({'auto': 0, 'spin': 1, 'yield': 2, 'block': 4}[mode])
    except Exception:
        pass


def set_cuda_memory_limit():
    if not cuda_ok or opts.cuda_mem_fraction == 0:
        return
    from modules.shared import cmd_opts
    try:
        torch_gc(force=True)
        mem = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(float(opts.cuda_mem_fraction), cmd_opts.device_id if cmd_opts.device_id is not None else 0)
        log.info(f'Torch CUDA memory limit: fraction={opts.cuda_mem_fraction:.2f} limit={round(opts.cuda_mem_fraction * mem / 1024 / 1024)} total={round(mem / 1024 / 1024)}')
    except Exception as e:
        log.warning(f'Torch CUDA memory limit: fraction={opts.cuda_mem_fraction:.2f} {e}')


def test_fp16():
    global fp16_ok # pylint: disable=global-statement
    if fp16_ok is not None:
        return fp16_ok
    if sys.platform == "darwin" or backend == 'openvino': # override
        fp16_ok = False
        return fp16_ok
    try:
        x = torch.tensor([[1.5,.0,.0,.0]]).to(device=device, dtype=torch.float16)
        layerNorm = torch.nn.LayerNorm(4, eps=0.00001, elementwise_affine=True, dtype=torch.float16, device=device)
        out = layerNorm(x)
        if out.dtype != torch.float16:
            raise RuntimeError('Torch FP16 test: dtype mismatch')
        if torch.all(torch.isnan(out)).item():
            raise RuntimeError('Torch FP16 test: NaN')
        fp16_ok = True
    except Exception as ex:
        log.warning(f'Torch FP16 test fail: {ex}')
        fp16_ok = False
    return fp16_ok


def test_bf16():
    global bf16_ok # pylint: disable=global-statement
    if bf16_ok is not None:
        return bf16_ok
    if opts.cuda_dtype != 'BF16': # don't override if the user sets it
        if sys.platform == "darwin" or backend == 'openvino' or backend == 'directml': # override
            bf16_ok = False
            return bf16_ok
        elif backend == 'rocm' or backend == 'zluda':
            agent = None
            if backend == 'rocm':
                agent = rocm.Agent(getattr(torch.cuda.get_device_properties(device), "gcnArchName", "gfx0000"))
            else:
                from modules.zluda_installer import default_agent
                agent = default_agent
            if agent is not None and agent.gfx_version < 0x1100 and agent.arch != rocm.MicroArchitecture.CDNA: # all cards before RDNA 3 except for CDNA cards
                bf16_ok = False
                return bf16_ok
    try:
        import torch.nn.functional as F
        image = torch.randn(1, 4, 32, 32).to(device=device, dtype=torch.bfloat16)
        out = F.interpolate(image, size=(64, 64), mode="nearest")
        if out.dtype != torch.bfloat16:
            raise RuntimeError('Torch BF16 test: dtype mismatch')
        if torch.all(torch.isnan(out)).item():
            raise RuntimeError('Torch BF16 test: NaN')
        bf16_ok = True
    except Exception as ex:
        log.warning(f'Torch BF16 test fail: {ex}')
        bf16_ok = False
    return bf16_ok


def set_cudnn_params():
    if cuda_ok:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
        except Exception:
            pass
        if torch.backends.cudnn.is_available():
            try:
                torch.backends.cudnn.deterministic = opts.cudnn_deterministic
                torch.use_deterministic_algorithms(opts.cudnn_deterministic)
                if opts.cudnn_deterministic:
                    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
                torch.backends.cudnn.benchmark = True
                if opts.cudnn_benchmark:
                    log.debug('Torch cuDNN: enable benchmark')
                    torch.backends.cudnn.benchmark_limit = 0
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass


def override_ipex_math():
    if backend == "ipex":
        try:
            torch.xpu.set_fp32_math_mode(mode=torch.xpu.FP32MathMode.TF32)
        except Exception:
            pass


def set_sdpa_params():
    try:
        if opts.cross_attention_optimization == "Scaled-Dot-Product":
            torch.backends.cuda.enable_flash_sdp('Flash attention' in opts.sdp_options)
            torch.backends.cuda.enable_mem_efficient_sdp('Memory attention' in opts.sdp_options)
            torch.backends.cuda.enable_math_sdp('Math attention' in opts.sdp_options)
            global sdpa_original # pylint: disable=global-statement
            if sdpa_original is not None:
                torch.nn.functional.scaled_dot_product_attention = sdpa_original
            else:
                sdpa_original = torch.nn.functional.scaled_dot_product_attention
            if backend == "rocm":
                if 'Flash attention' in opts.sdp_options:
                    try:
                        # https://github.com/huggingface/diffusers/discussions/7172
                        from flash_attn import flash_attn_func
                        sdpa_pre_flash_atten = torch.nn.functional.scaled_dot_product_attention
                        @wraps(sdpa_pre_flash_atten)
                        def sdpa_flash_atten(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                            if query.shape[-1] <= 128 and attn_mask is None and query.dtype != torch.float32:
                                return flash_attn_func(q=query.transpose(1, 2), k=key.transpose(1, 2), v=value.transpose(1, 2), dropout_p=dropout_p, causal=is_causal, softmax_scale=scale).transpose(1, 2)
                            else:
                                return sdpa_pre_flash_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                        torch.nn.functional.scaled_dot_product_attention = sdpa_flash_atten
                        log.debug('ROCm Flash Attention Hijacked')
                    except Exception as err:
                        log.error(f'ROCm Flash Attention failed: {err}')
            if 'Sage attention' in opts.sdp_options:
                try:
                    install('sageattention')
                    from sageattention import sageattn
                    sdpa_pre_sage_atten = torch.nn.functional.scaled_dot_product_attention
                    @wraps(sdpa_pre_sage_atten)
                    def sdpa_sage_atten(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
                        if query.shape[-1] in {128, 96, 64} and attn_mask is None and query.dtype != torch.float32:
                            return sageattn(q=query, k=key, v=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                        else:
                            return sdpa_pre_sage_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
                    torch.nn.functional.scaled_dot_product_attention = sdpa_sage_atten
                    log.debug('SDPA Sage Attention Hijacked')
                except Exception as err:
                    log.error(f'SDPA Sage Attention failed: {err}')
            if 'Dynamic attention' in opts.sdp_options:
                try:
                    global sdpa_pre_dyanmic_atten # pylint: disable=global-statement
                    sdpa_pre_dyanmic_atten = torch.nn.functional.scaled_dot_product_attention
                    from modules.sd_hijack_dynamic_atten import sliced_scaled_dot_product_attention
                    torch.nn.functional.scaled_dot_product_attention = sliced_scaled_dot_product_attention
                    log.debug('SDPA Dynamic Attention Hijacked')
                except Exception as err:
                    log.error(f'SDPA Dynamic Attention failed: {err}')
    except Exception:
        pass


def set_dtype():
    global dtype, dtype_vae, dtype_unet, unet_needs_upcast, inference_context # pylint: disable=global-statement
    test_fp16()
    test_bf16()
    if opts.cuda_dtype == 'Auto': # detect
        if bf16_ok:
            dtype = torch.bfloat16
            dtype_vae = torch.bfloat16
            dtype_unet = torch.bfloat16
        elif fp16_ok:
            dtype = torch.float16
            dtype_vae = torch.float16
            dtype_unet = torch.float16
        else:
            dtype = torch.float32
            dtype_vae = torch.float32
            dtype_unet = torch.float32
    elif opts.cuda_dtype == 'FP32':
        dtype = torch.float32
        dtype_vae = torch.float32
        dtype_unet = torch.float32
    elif opts.cuda_dtype == 'BF16':
        if not bf16_ok:
            log.warning(f'Torch device capability failed: device={device} dtype={torch.bfloat16}')
        dtype = torch.bfloat16
        dtype_vae = torch.bfloat16
        dtype_unet = torch.bfloat16
    elif opts.cuda_dtype == 'FP16':
        if not fp16_ok:
            log.warning(f'Torch device capability failed: device={device} dtype={torch.float16}')
        dtype = torch.float16
        dtype_vae = torch.float16
        dtype_unet = torch.float16

    if opts.no_half:
        dtype = torch.float32
        dtype_vae = torch.float32
        dtype_unet = torch.float32
        log.info(f'Torch override: no-half dtype={dtype}')
    if opts.no_half_vae:
        dtype_vae = torch.float32
        log.info(f'Torch override: no-half-vae dtype={dtype_vae}')
    unet_needs_upcast = opts.upcast_sampling
    if opts.inference_mode == 'inference-mode':
        inference_context = torch.inference_mode
    elif opts.inference_mode == 'none':
        inference_context = contextlib.nullcontext
    else:
        inference_context = torch.no_grad


def set_cuda_params():
    override_ipex_math()
    set_cuda_memory_limit()
    set_cudnn_params()
    set_sdpa_params()
    set_dtype()
    if backend == 'openvino':
        from modules.intel.openvino import get_device as get_raw_openvino_device
        device_name = get_raw_openvino_device()
    else:
        device_name = torch.device(get_optimal_device_name())
    log.info(f'Torch parameters: backend={backend} device={device_name} config={opts.cuda_dtype} dtype={dtype} context={inference_context.__name__} nohalf={opts.no_half} nohalfvae={opts.no_half_vae} upcast={opts.upcast_sampling} deterministic={opts.cudnn_deterministic} fp16={"pass" if fp16_ok else "fail"} bf16={"pass" if bf16_ok else "fail"} optimization="{opts.cross_attention_optimization}"')


def cond_cast_unet(tensor):
    return tensor.to(dtype_unet) if unet_needs_upcast else tensor


def cond_cast_float(tensor):
    return tensor.float() if unet_needs_upcast else tensor


def randn(seed, shape=None):
    torch.manual_seed(seed)
    if backend == 'ipex':
        torch.xpu.manual_seed_all(seed)
    if shape is None:
        return None
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    elif opts.diffusers_generator_device == "CPU":
        return torch.randn(shape, device=cpu)
    else:
        return torch.randn(shape, device=device)


def randn_without_seed(shape):
    if device.type == 'mps':
        return torch.randn(shape, device=cpu).to(device)
    return torch.randn(shape, device=device)


def autocast(disable=False):
    if disable or dtype == torch.float32:
        return contextlib.nullcontext()
    if backend == 'directml':
        return torch.dml.amp.autocast(dtype)
    if cuda_ok:
        return torch.autocast("cuda")
    else:
        return torch.autocast("cpu")


def without_autocast(disable=False):
    if disable:
        return contextlib.nullcontext()
    if backend == 'directml':
        return torch.dml.amp.autocast(enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext() # pylint: disable=unexpected-keyword-arg
    if cuda_ok:
        return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()
    else:
        return torch.autocast("cpu", enabled=False) if torch.is_autocast_enabled() else contextlib.nullcontext()


class NansException(Exception):
    pass


def test_for_nans(x, where):
    if opts.disable_nan_check:
        return
    if not torch.all(torch.isnan(x)).item():
        return
    if where == "unet":
        message = "A tensor with all NaNs was produced in Unet."
        if not opts.no_half:
            message += " This could be either because there's not enough precision to represent the picture, or because your video card does not support half type. Try setting the \"Upcast cross attention layer to float32\" option in Settings > Stable Diffusion or using the --no-half commandline argument to fix this."
    elif where == "vae":
        message = "A tensor with all NaNs was produced in VAE."
        if not opts.no_half and not opts.no_half_vae:
            message += " This could be because there's not enough precision to represent the picture. Try adding --no-half-vae commandline argument to fix this."
    else:
        message = "A tensor with all NaNs was produced."
    message += " Use --disable-nan-check commandline argument to disable this check."
    raise NansException(message)


def normalize_device(dev):
    if torch.device(dev).type in {"cpu", "mps", "meta"}:
        return torch.device(dev)
    if torch.device(dev).index is None:
        return torch.device(str(dev), index=0)
    return torch.device(dev)


def same_device(d1, d2):
    if d1.type != d2.type:
        return False
    return normalize_device(d1) == normalize_device(d2)
