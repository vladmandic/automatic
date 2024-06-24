import os
import sys
import torch


_topk = torch.topk
def topk(tensor: torch.Tensor, *args, **kwargs):
    device = tensor.device
    values, indices = _topk(tensor.cpu(), *args, **kwargs)
    return torch.return_types.topk((values.to(device), indices.to(device),))


def _join_rocm_home(*paths) -> str:
    from torch.utils.cpp_extension import ROCM_HOME
    return os.path.join(ROCM_HOME, *paths)


def do_hijack():
    torch.version.hip = "5.7"
    torch.topk = topk
    platform = sys.platform
    sys.platform = ""
    from torch.utils import cpp_extension
    sys.platform = platform
    cpp_extension.IS_WINDOWS = platform == "win32"
    cpp_extension.IS_MACOS = False
    cpp_extension.IS_LINUX = platform.startswith('linux')
    cpp_extension._join_rocm_home = _join_rocm_home # pylint: disable=protected-access
