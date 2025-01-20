import torch
from modules import rocm


_topk = torch.topk
def topk(input: torch.Tensor, *args, **kwargs): # pylint: disable=redefined-builtin
    device = input.device
    values, indices = _topk(input.cpu(), *args, **kwargs)
    return torch.return_types.topk((values.to(device), indices.to(device),))


def do_hijack():
    torch.version.hip = rocm.version
    torch.topk = topk
