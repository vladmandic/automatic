from pathlib import Path
import torch
import gguf
from .gguf_utils import TORCH_COMPATIBLE_QTYPES
from .gguf_tensor import GGMLTensor


def load_gguf(path: str, compute_dtype: torch.dtype) -> dict[str, GGMLTensor]:
    sd: dict[str, GGMLTensor] = {}
    reader = gguf.GGUFReader(path)
    for tensor in reader.tensors:
        torch_tensor = torch.from_numpy(tensor.data)
        shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        if tensor.tensor_type in TORCH_COMPATIBLE_QTYPES:
            torch_tensor = torch_tensor.view(*shape)
        sd[tensor.name] = GGMLTensor(torch_tensor, ggml_quantization_type=tensor.tensor_type, tensor_shape=shape, compute_dtype=compute_dtype)
    return sd


def load_model(path: str, dtype: torch.dtype) -> torch.nn.Module:
    state_dict = load_gguf(path, compute_dtype=dtype)
    # TODO create torch.nn.Modules, etc...
    # state_dict = state_dict.get("state_dict") or state_dict
    for k, v in state_dict.items():
        print(k, type(v))
