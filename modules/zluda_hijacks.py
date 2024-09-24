import torch
from modules import rocm


_topk = torch.topk
def topk(input: torch.Tensor, *args, **kwargs): # pylint: disable=redefined-builtin
    device = input.device
    values, indices = _topk(input.cpu(), *args, **kwargs)
    return torch.return_types.topk((values.to(device), indices.to(device),))


_fft_fftn = torch.fft.fftn
def fft_fftn(input: torch.Tensor, *args, **kwargs) -> torch.Tensor: # pylint: disable=redefined-builtin
    return _fft_fftn(input.cpu(), *args, **kwargs).to(input.device)


_fft_ifftn = torch.fft.ifftn
def fft_ifftn(input: torch.Tensor, *args, **kwargs) -> torch.Tensor: # pylint: disable=redefined-builtin
    return _fft_ifftn(input.cpu(), *args, **kwargs).to(input.device)


def jit_script(f, *_, **__): # experiment / provide dummy graph
    f.graph = torch._C.Graph() # pylint: disable=protected-access
    return f


def do_hijack():
    torch.version.hip = rocm.version
    torch.topk = topk
    torch.fft.fftn = fft_fftn
    torch.fft.ifftn = fft_ifftn
    torch.jit.script = jit_script
