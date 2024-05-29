import os
import sys
from typing import Union
import torch
from torch._prims_common import DeviceLikeType
import onnxruntime as ort
from modules import shared, devices


PLATFORM = sys.platform
do_nothing = lambda _: None # pylint: disable=unnecessary-lambda-assignment


def _join_rocm_home(*paths) -> str:
    from torch.utils.cpp_extension import ROCM_HOME
    return os.path.join(ROCM_HOME, *paths)


def is_zluda(device: DeviceLikeType):
    device = torch.device(device)
    return torch.cuda.get_device_name(device).endswith("[ZLUDA]")


def test(device: DeviceLikeType) -> Union[Exception, None]:
    device = torch.device(device)
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        assert out.sum().is_nonzero()
        return None
    except Exception as e:
        return e


def initialize_zluda():
    device = devices.get_optimal_device()
    if not devices.cuda_ok or not is_zluda(device):
        return

    torch.version.hip = "5.7"
    sys.platform = ""
    from torch.utils import cpp_extension
    sys.platform = PLATFORM
    cpp_extension.IS_WINDOWS = PLATFORM == "win32"
    cpp_extension.IS_MACOS = False
    cpp_extension.IS_LINUX = sys.platform.startswith('linux')
    cpp_extension._join_rocm_home = _join_rocm_home # pylint: disable=protected-access

    if cpp_extension.IS_WINDOWS:
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_flash_sdp = do_nothing
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_math_sdp = do_nothing
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp = do_nothing
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp = do_nothing
        shared.opts.sdp_options = ['Math attention']

        # ONNX Runtime is not supported
        ort.capi._pybind_state.get_available_providers = lambda: [v for v in ort.get_available_providers() if v != 'CUDAExecutionProvider'] # pylint: disable=protected-access
        ort.get_available_providers = ort.capi._pybind_state.get_available_providers # pylint: disable=protected-access
        if shared.opts.onnx_execution_provider == 'CUDAExecutionProvider':
            shared.opts.onnx_execution_provider = 'CPUExecutionProvider'

        devices.device_codeformer = devices.cpu

        result = test(device)
        if result is not None:
            shared.log.warning(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={torch.cuda.get_device_name(device)}')
            shared.log.error(result)
            torch.cuda.is_available = lambda: False
            devices.cuda_ok = False
            devices.backend = 'cpu'
            devices.device = devices.device_esrgan = devices.device_gfpgan = devices.device_interrogate = devices.cpu
