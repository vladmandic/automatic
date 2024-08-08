import torch


_topk = torch.topk
def topk(tensor: torch.Tensor, *args, **kwargs):
    device = tensor.device
    values, indices = _topk(tensor.cpu(), *args, **kwargs)
    return torch.return_types.topk((values.to(device), indices.to(device),))


def do_hijack():
    torch.version.hip = "5.7"
    torch.topk = topk
