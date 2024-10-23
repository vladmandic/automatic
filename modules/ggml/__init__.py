from pathlib import Path
import torch
import gguf
from .gguf_utils import TORCH_COMPATIBLE_QTYPES
from .gguf_tensor import GGMLTensor


def load_gguf_state_dict(path: str, compute_dtype: torch.dtype) -> dict[str, GGMLTensor]:
    sd: dict[str, GGMLTensor] = {}
    stats = {}
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        torch_tensor = torch.from_numpy(tensor.data)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            torch_tensor = torch_tensor.view(*shape)
        sd[tensor.name] = GGMLTensor(torch_tensor, ggml_quantization_type=tensor.tensor_type, tensor_shape=shape, compute_dtype=compute_dtype)
        if tensor.tensor_type.name not in stats:
            stats[tensor.tensor_type.name] = 0
        stats[tensor.tensor_type.name] += 1
    return sd, stats
