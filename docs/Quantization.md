# Quantization

Quantization is a process of:
- storage-optimization  
  reducing the memory footprint of the model by reducing the precision of parameters in a model  
- compute-optimization  
  speed up the inference process by providing optimized kernels for native execution in quantized precision  

For storage-only quantization, the model is quantized to lower precision but the operations are still performed in the original precision which means that each operation needs to be upcasted to the original precision before execution resulting in a performance overhead.

## Using Quantized Models

Quantization can be done in multiple ways:
on-the-fly- by quantizing on-the-fly during model load  
  available by selecting *settings -> quantization* for some quantization types  
- by quantizing immediately after model load  
  available by selecting *settings -> quantization* for all quantization types  
- by simply loading a pre-quantized model  
  quantization type will be auto-determined at the start of the load  
- during model training itself  
  out-of-scope for this document  

## Quantization Engines

!!! tip

    If you're on Windows with a compatible GPU, you may try WSL2 for broader feature compatibiliy  
    See [WSL Wiki](https://github.com/vladmandic/automatic/wiki/WSL) for more details

### BitsAndBytes

Typical models pre-quantized with `bitsandbytes` would have look like `*nf4.safetensors` or `*fp8.safetensors`

!!! note

    BnB is the only quantization method that allows for usage of balanced offload as well as quantization on-the-fly during load, thus it is considered most versatile choice, but it is not available on all platforms.

Limitations:
- default `bitsandbytes` package only supports nVidia GPUs  
  some quantization types require newer GPU with supported CUDA ops: e.g. *nVidia Turing* GPUs or newer  
- `bitsandbytes` relies on `triton` packages which are not available on windows unless manually compiled/installed  
  without them, performance is significantly reduced  
- for AMD/ROCm: [link](https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=AMD+ROCm#amd-gpu) 
- for Intel/IPEX: [link](https://huggingface.co/docs/bitsandbytes/main/en/installation?backend=Intel+CPU+%2B+GPU#multi-backend)  

### Optimum-Quanto

Typical models pre-quantized with `optimum.quanto` would have look like `*qint.safetensors`.

!!! note

    OQ is highly efficient with its qint8/qint4 quantization types, but its usage is limited to specific platforms and cannot be used with broad offloading methods

Limitations:
- requires `torch==2.4.0`  
  if you're running older torch, you can try upgrading it or running sdnext with `--reinstall` flag  
- not compatible with balanced offload  
- not supported on Intel Arc/IPEX since IPEX is still based on Torch 2.3  
- not supported with Zluda since Zluda does not support torch 2.4  

### GGUF

**GGUF** is a binary file format used to package pre-quantized models.  

**GGUF** is originally desiged by `llama.cpp` project and intended to be used with its **GGML** execution runtime.  
However, without GGML, GGUF provides storage-only quantization which means that every operation needs to be upcast to current device precision before execution (typically FP16 or BF16) which comes with a significant performance overhead.

!!! warning

    Right now, all popular T2I inference UIs *(SD.Next, Forge, ComfyUI, InvokeAI etc.)* are using GGUF as storage-only and as such usage of GGUF is not recommended!  

- `gguf` supports wide range of quantization types and is not platform or GPU dependent  
- `gguf` does not provide native GPU kernels which means that `gguf` is purely a *storage optimization*  
- `gguf` reduces model size and memory usage, but it does slow down model inference since all quantized weights are de-quantized on-the-fly  

Limitations:
- `gguf` is not compatible with model offloading as it would trigger de-quantization  
- *note*: only supported component in `gguf` binary format is UNET/Transformer  
  you cannot load all-in-one single-file GGUF model

### NNCF

NNCF provides full cross-platform storage-only quantization (referred to as model compression)  
with optional platform-specific compute-optimization (available only on OpenVINO platform)  

!!! note

    Advantage of **NNCF** is that it does work on any platform: if you're having issues with `optimum-quanto` or `bitsandbytes`, try it out!  

- broad platform and GPU support  
- enable in *Settings -> Compute -> Compress model weights with NNCF*  
- see [NNCF Wiki](https://github.com/vladmandic/automatic/wiki/NNCF-Compression) for more details  

## Errors

!!! caution

    Using incompatible configurations will result in errors during model load:

- BitsAndBytes nf4 quantization is not compatible with sequential offload  
  > Error: Blockwise quantization only supports 16/32-bit floats  
- Quanto qint quantization is not compatible with balanced offload  
  > Error: QBytesTensor.__new__() missing 5 required positional arguments  
- Quanto qint quantization is not compatible with sequential offload  
  > Error: Expected all tensors to be on the same device  
