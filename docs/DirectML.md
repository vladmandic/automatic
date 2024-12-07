# DirectML

SD.Next includes support for PyTorch-DirectML.

## How to

Add `--use-directml` on commandline arguments.

For details, go to [Installation](https://github.com/vladmandic/automatic/wiki/Installation#general-installation).

## Performance

The performance is quite bad compared to ROCm.

If you are familiar with Linux system, we recommend ROCm.

## FAQ

### DirectML does not collect garbage memory

PyTorch-DirectML does not access graphics memory by indexing. Because PyTorch-DirectML's tensor implementation extends [OpaqueTensorImpl](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/OpaqueTensorImpl.h), we cannot access the actual storage of a tensor.

### An error occurs with no error message

If you met `RuntimeError` with no error message (or empty), please report us via GitHub issue or Discord. (please check whether there's a duplicated issue)

### It does not work properly with FP16

If it works with FP32, please report us via GitHub issue or Discord. (please check whether there's a duplicated issue)

### The terminal is suddenly frozen during generation

Please report us via GitHub issue or Discord. (please check whether there's a duplicated issue)

## Olive (experimental support)

Refer to [ONNX Runtime](https://github.com/vladmandic/automatic/wiki/ONNX-Runtime-&-Olive)
