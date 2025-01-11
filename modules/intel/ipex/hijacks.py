import os
from functools import wraps
from contextlib import nullcontext
import torch
import numpy as np
from modules import devices, errors

device_supports_fp64 = torch.xpu.has_fp64_dtype() if hasattr(torch.xpu, "has_fp64_dtype") else torch.xpu.get_device_properties("xpu").has_fp64

# pylint: disable=protected-access, missing-function-docstring, line-too-long, unnecessary-lambda, no-else-return

class DummyDataParallel(torch.nn.Module): # pylint: disable=missing-class-docstring, unused-argument, too-few-public-methods
    def __new__(cls, module, device_ids=None, output_device=None, dim=0): # pylint: disable=unused-argument
        if isinstance(device_ids, list) and len(device_ids) > 1:
            errors.log.error("IPEX backend doesn't support DataParallel on multiple XPU devices")
        return module.to(devices.device)

def return_null_context(*args, **kwargs): # pylint: disable=unused-argument
    return nullcontext()

@property
def is_cuda(self):
    return self.device.type == 'xpu' or self.device.type == 'cuda'

def check_device(device):
    return bool((isinstance(device, torch.device) and device.type == "cuda") or (isinstance(device, str) and "cuda" in device) or isinstance(device, int))

def return_xpu(device):
    return f"xpu:{device.split(':')[-1]}" if isinstance(device, str) and ":" in device else f"xpu:{device}" if isinstance(device, int) else torch.device(f"xpu:{device.index}" if device.index is not None else "xpu") if isinstance(device, torch.device) else "xpu"


# Autocast
original_autocast_init = torch.amp.autocast_mode.autocast.__init__
@wraps(torch.amp.autocast_mode.autocast.__init__)
def autocast_init(self, device_type, dtype=None, enabled=True, cache_enabled=None):
    if device_type == "cuda" or device_type == "xpu":
        if dtype is None:
            dtype = devices.dtype
        return original_autocast_init(self, device_type="xpu", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return original_autocast_init(self, device_type=device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

# Latent Antialias CPU Offload:
original_interpolate = torch.nn.functional.interpolate
@wraps(torch.nn.functional.interpolate)
def interpolate(tensor, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False): # pylint: disable=too-many-arguments
    if mode in {'bicubic', 'bilinear'}:
        return_device = tensor.device
        return_dtype = tensor.dtype
        return original_interpolate(tensor.to("cpu", dtype=torch.float32), size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias).to(return_device, dtype=return_dtype)
    else:
        return original_interpolate(tensor, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)


# Diffusers Float64 (Alchemist GPUs doesn't support 64 bit):
original_from_numpy = torch.from_numpy
@wraps(torch.from_numpy)
def from_numpy(ndarray):
    if ndarray.dtype == float:
        return original_from_numpy(ndarray.astype('float32'))
    else:
        return original_from_numpy(ndarray)

original_as_tensor = torch.as_tensor
@wraps(torch.as_tensor)
def as_tensor(data, dtype=None, device=None):
    if check_device(device):
        device = return_xpu(device)
    if isinstance(data, np.ndarray) and data.dtype == float and not (
        (isinstance(device, torch.device) and device.type == "cpu") or (isinstance(device, str) and "cpu" in device)):
        return original_as_tensor(data, dtype=torch.float32, device=device)
    else:
        return original_as_tensor(data, dtype=dtype, device=device)


if device_supports_fp64 and os.environ.get('IPEX_FORCE_ATTENTION_SLICE', None) is None:
    original_torch_bmm = torch.bmm
    original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
else:
    # 32 bit attention workarounds for Alchemist:
    try:
        from .attention import torch_bmm_32_bit as original_torch_bmm
        from .attention import scaled_dot_product_attention_32_bit as original_scaled_dot_product_attention
    except Exception: # pylint: disable=broad-exception-caught
        original_torch_bmm = torch.bmm
        original_scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention


# Data Type Errors:
@wraps(torch.bmm)
def torch_bmm(input, mat2, *, out=None):
    if input.dtype != mat2.dtype:
        mat2 = mat2.to(input.dtype)
    return original_torch_bmm(input, mat2, out=out)

@wraps(torch.nn.functional.scaled_dot_product_attention)
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    if query.dtype != key.dtype:
        key = key.to(dtype=query.dtype)
    if query.dtype != value.dtype:
        value = value.to(dtype=query.dtype)
    if attn_mask is not None and query.dtype != attn_mask.dtype:
        attn_mask = attn_mask.to(dtype=query.dtype)
    return original_scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)

# Diffusers FreeU
original_fft_fftn = torch.fft.fftn
@wraps(torch.fft.fftn)
def fft_fftn(input, s=None, dim=None, norm=None, *, out=None):
    return_dtype = input.dtype
    return original_fft_fftn(input.to(dtype=torch.float32), s=s, dim=dim, norm=norm, out=out).to(dtype=return_dtype)

# Diffusers FreeU
original_fft_ifftn = torch.fft.ifftn
@wraps(torch.fft.ifftn)
def fft_ifftn(input, s=None, dim=None, norm=None, *, out=None):
    return_dtype = input.dtype
    return original_fft_ifftn(input.to(dtype=torch.float32), s=s, dim=dim, norm=norm, out=out).to(dtype=return_dtype)

# A1111 FP16
original_functional_group_norm = torch.nn.functional.group_norm
@wraps(torch.nn.functional.group_norm)
def functional_group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)

# A1111 BF16
original_functional_layer_norm = torch.nn.functional.layer_norm
@wraps(torch.nn.functional.layer_norm)
def functional_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    if weight is not None and input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and weight is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

# Training
original_functional_linear = torch.nn.functional.linear
@wraps(torch.nn.functional.linear)
def functional_linear(input, weight, bias=None):
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_linear(input, weight, bias=bias)

original_functional_conv1d = torch.nn.functional.conv1d
@wraps(torch.nn.functional.conv1d)
def functional_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_conv1d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

original_functional_conv2d = torch.nn.functional.conv2d
@wraps(torch.nn.functional.conv2d)
def functional_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

# LTX Video
original_functional_conv3d = torch.nn.functional.conv3d
@wraps(torch.nn.functional.conv3d)
def functional_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.dtype != weight.data.dtype:
        input = input.to(dtype=weight.data.dtype)
    if bias is not None and bias.data.dtype != weight.data.dtype:
        bias.data = bias.data.to(dtype=weight.data.dtype)
    return original_functional_conv3d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)

# SwinIR BF16:
original_functional_pad = torch.nn.functional.pad
@wraps(torch.nn.functional.pad)
def functional_pad(input, pad, mode='constant', value=None):
    if mode == 'reflect' and input.dtype == torch.bfloat16:
        return original_functional_pad(input.to(torch.float32), pad, mode=mode, value=value).to(dtype=torch.bfloat16)
    else:
        return original_functional_pad(input, pad, mode=mode, value=value)


original_torch_tensor = torch.tensor
@wraps(torch.tensor)
def torch_tensor(data, *args, dtype=None, device=None, **kwargs):
    if check_device(device):
        device = return_xpu(device)
    if not device_supports_fp64:
        if (isinstance(device, torch.device) and device.type == "xpu") or (isinstance(device, str) and "xpu" in device):
            if dtype == torch.float64:
                dtype = torch.float32
            elif dtype is None and (hasattr(data, "dtype") and (data.dtype == torch.float64 or data.dtype == float)):
                dtype = torch.float32
    return original_torch_tensor(data, *args, dtype=dtype, device=device, **kwargs)

original_Tensor_to = torch.Tensor.to
@wraps(torch.Tensor.to)
def Tensor_to(self, device=None, *args, **kwargs):
    if check_device(device):
        return original_Tensor_to(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_to(self, device, *args, **kwargs)

original_Tensor_cuda = torch.Tensor.cuda
@wraps(torch.Tensor.cuda)
def Tensor_cuda(self, device=None, *args, **kwargs):
    if check_device(device):
        return original_Tensor_cuda(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_cuda(self, device, *args, **kwargs)

original_Tensor_pin_memory = torch.Tensor.pin_memory
@wraps(torch.Tensor.pin_memory)
def Tensor_pin_memory(self, device=None, *args, **kwargs):
    if device is None:
        device = "xpu"
    if check_device(device):
        return original_Tensor_pin_memory(self, return_xpu(device), *args, **kwargs)
    else:
        return original_Tensor_pin_memory(self, device, *args, **kwargs)

original_UntypedStorage_init = torch.UntypedStorage.__init__
@wraps(torch.UntypedStorage.__init__)
def UntypedStorage_init(*args, device=None, **kwargs):
    if check_device(device):
        return original_UntypedStorage_init(*args, device=return_xpu(device), **kwargs)
    else:
        return original_UntypedStorage_init(*args, device=device, **kwargs)

original_UntypedStorage_cuda = torch.UntypedStorage.cuda
@wraps(torch.UntypedStorage.cuda)
def UntypedStorage_cuda(self, device=None, *args, **kwargs):
    if check_device(device):
        return original_UntypedStorage_cuda(self, return_xpu(device), *args, **kwargs)
    else:
        return original_UntypedStorage_cuda(self, device, *args, **kwargs)

original_torch_empty = torch.empty
@wraps(torch.empty)
def torch_empty(*args, device=None, **kwargs):
    if check_device(device):
        return original_torch_empty(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_empty(*args, device=device, **kwargs)

original_torch_randn = torch.randn
@wraps(torch.randn)
def torch_randn(*args, device=None, dtype=None, **kwargs):
    if dtype is bytes:
        dtype = None
    if check_device(device):
        return original_torch_randn(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_randn(*args, device=device, **kwargs)

original_torch_ones = torch.ones
@wraps(torch.ones)
def torch_ones(*args, device=None, **kwargs):
    if check_device(device):
        return original_torch_ones(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_ones(*args, device=device, **kwargs)

original_torch_zeros = torch.zeros
@wraps(torch.zeros)
def torch_zeros(*args, device=None, **kwargs):
    if check_device(device):
        return original_torch_zeros(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_zeros(*args, device=device, **kwargs)

original_torch_full = torch.full
@wraps(torch.full)
def torch_full(*args, device=None, **kwargs):
    if check_device(device):
        return original_torch_full(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_full(*args, device=device, **kwargs)

original_torch_linspace = torch.linspace
@wraps(torch.linspace)
def torch_linspace(*args, device=None, **kwargs):
    if check_device(device):
        return original_torch_linspace(*args, device=return_xpu(device), **kwargs)
    else:
        return original_torch_linspace(*args, device=device, **kwargs)

original_torch_Generator = torch.Generator
@wraps(torch.Generator)
def torch_Generator(device=None):
    if check_device(device):
        return original_torch_Generator(return_xpu(device))
    else:
        return original_torch_Generator(device)

original_torch_load = torch.load
@wraps(torch.load)
def torch_load(f, map_location=None, *args, **kwargs):
    if map_location is None:
        map_location = "xpu"
    if check_device(map_location):
        return original_torch_load(f, *args, map_location=return_xpu(map_location), **kwargs)
    else:
        return original_torch_load(f, *args, map_location=map_location, **kwargs)


# Hijack Functions:
def ipex_hijacks(legacy=True):
    if legacy and float(torch.__version__[:3]) < 2.5:
        torch.nn.functional.interpolate = interpolate
    torch.tensor = torch_tensor
    torch.Tensor.to = Tensor_to
    torch.Tensor.cuda = Tensor_cuda
    torch.Tensor.pin_memory = Tensor_pin_memory
    torch.UntypedStorage.__init__ = UntypedStorage_init
    torch.UntypedStorage.cuda = UntypedStorage_cuda
    torch.empty = torch_empty
    torch.randn = torch_randn
    torch.ones = torch_ones
    torch.zeros = torch_zeros
    torch.full = torch_full
    torch.linspace = torch_linspace
    torch.Generator = torch_Generator
    torch.load = torch_load

    torch.backends.cuda.sdp_kernel = return_null_context
    torch.nn.DataParallel = DummyDataParallel
    torch.UntypedStorage.is_cuda = is_cuda
    torch.amp.autocast_mode.autocast.__init__ = autocast_init

    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    torch.nn.functional.group_norm = functional_group_norm
    torch.nn.functional.layer_norm = functional_layer_norm
    torch.nn.functional.linear = functional_linear
    torch.nn.functional.conv1d = functional_conv1d
    torch.nn.functional.conv2d = functional_conv2d
    torch.nn.functional.conv3d = functional_conv3d
    torch.nn.functional.pad = functional_pad

    torch.bmm = torch_bmm
    torch.fft.fftn = fft_fftn
    torch.fft.ifftn = fft_ifftn
    if not device_supports_fp64:
        torch.from_numpy = from_numpy
        torch.as_tensor = as_tensor
