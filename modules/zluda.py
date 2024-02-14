import platform
import torch
from modules import shared, devices


def initialize_zluda():
    if platform.system() == "Windows" and devices.cuda_ok and torch.cuda.get_device_name(devices.get_optimal_device()).endswith("[ZLUDA]"):
        torch.backends.cudnn.enabled = False
        shared.opts.cross_attention_optimization = "Batch matrix-matrix"
