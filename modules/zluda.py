import platform
import torch
from diffusers.models.attention_processor import AttnProcessor
from modules import shared, devices


def initialize_zluda():
    if platform.system() == "Windows" and devices.cuda_ok and torch.cuda.get_device_name(devices.get_optimal_device()).endswith("[ZLUDA]"):
        shared.log.warning("Detected ZLUDA device. Currently, ZLUDA support is experimental and unstable.")
        torch.backends.cudnn.enabled = False
        if shared.opts.cross_attention_optimization == "Scaled-Dot-Product":
            shared.opts.cross_attention_optimization = "Batch matrix-matrix"
        if shared.opts.zluda_force_sync:
            patch_attention_processor(AttnProcessor)


def patch_attention_processor(cls):
    forward = cls.__call__
    def patched(self, *args, **kwargs):
        R = forward(self, *args, **kwargs)
        torch.cuda.synchronize()
        return R
    cls.__call__ = patched
