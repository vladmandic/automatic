import platform
import torch
from modules import shared, devices


def test(device: torch.device):
    try:
        ten1 = torch.randn((2, 4,), device=device)
        ten2 = torch.randn((4, 8,), device=device)
        out = torch.mm(ten1, ten2)
        return out.sum().is_nonzero()
    except Exception:
        return False


def initialize_zluda():
    device = devices.get_optimal_device()
    if platform.system() == "Windows" and devices.cuda_ok and torch.cuda.get_device_name(device).endswith("[ZLUDA]"):
        torch.backends.cudnn.enabled = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        shared.opts.sdp_options = ['Math attention']
        devices.device_codeformer = devices.cpu

        if not test(device):
            shared.log.error(f'ZLUDA device failed to pass basic operation test: index={device.index}, device_name={torch.cuda.get_device_name(device)}')
            torch.cuda.is_available = lambda: False
            devices.cuda_ok = False
            devices.backend = 'cpu'
            devices.device = devices.device_esrgan = devices.device_gfpgan = devices.device_interrogate = devices.cpu
