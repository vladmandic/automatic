import torch
from modules import rocm


_topk = torch.topk
def topk(tensor: torch.Tensor, *args, **kwargs):
    device = tensor.device
    values, indices = _topk(tensor.cpu(), *args, **kwargs)
    return torch.return_types.topk((values.to(device), indices.to(device),))


def jit_script(f, *_, **__): # experiment / provide dummy graph
    f.graph = torch._C.Graph() # pylint: disable=protected-access
    return f


def do_hijack():
    torch.version.hip = rocm.version
    torch.topk = topk
    torch.jit.script = jit_script
