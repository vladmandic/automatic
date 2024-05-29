from typing import Optional
import torch
from .utils import rDevice, get_device


class Device:
    idx: int

    def __enter__(self, device: Optional[rDevice]=None):
        torch.dml.context_device = get_device(device)
        self.idx = torch.dml.context_device.index

    def __init__(self, device: Optional[rDevice]=None) -> torch.device: # pylint: disable=return-in-init
        self.idx = get_device(device).index

    def __exit__(self, t, v, tb):
        torch.dml.context_device = None
