import platform
import torch
from modules import shared, devices


def initialize_zluda():
    if platform.system() == "Windows" and devices.cuda_ok and torch.cuda.get_device_name(devices.get_optimal_device()).endswith("[ZLUDA]"):
        shared.log.warning("Detected ZLUDA device. Currently, ZLUDA support is experimental and unstable.")
        torch.backends.cudnn.enabled = shared.opts.zluda_enable_cudnn
        if torch.backends.cudnn.enabled:
            shared.log.warning("cuDNN with ZLUDA won't work at this moment. Please wait for future update.")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        shared.opts.sdp_options = ['Math attention']
