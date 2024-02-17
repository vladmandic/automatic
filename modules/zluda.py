import platform
import torch
from modules import shared, devices


def initialize_zluda():
    if platform.system() == "Windows" and devices.cuda_ok and torch.cuda.get_device_name(devices.get_optimal_device()).endswith("[ZLUDA]"):
        shared.log.warning("Detected ZLUDA device. Currently, ZLUDA support is experimental and unstable.")
        torch.backends.cudnn.enabled = shared.opts.zluda_enable_cudnn
        if torch.backends.cudnn.enabled:
            shared.log.warning("cuDNN with ZLUDA won't work at this moment. Please wait for future update.")
        if shared.opts.cross_attention_optimization == "Scaled-Dot-Product":
            shared.log.warning("ZLUDA does not support Scaled-Dot-Product attention. Please consider changing it to Batch matrix-matrix.")
