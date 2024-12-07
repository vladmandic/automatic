# Benchmark

To run standardized benchmark, you can use UI -> System -> Benchmark feature
or via CLI using `cli/run-benchmark.py` script.

It runs identical tests, but often CLI is faster due to lower overhead.

## Environment

- Hardware: nVidia RTX 4090 with i9-13900KF
- Packages: Torch 2.1.0 with CUDA 12.1 and cuDNN 8.9
- Params: model=SD15 | batch-size=4 | batch-count=4 | steps=50 | resolution=512px | sampler=Euler A

## Results

Basic tests using UI:

|||Diffusers||Original|||
|---|---|---|---|---|---|---|
|Precision|Params|SDP|xFormers|SDP|xFormers|None|
|FP32|Default|33.0|20.0||||
|BF16|Default|73.0|45.5||||
|FP16|Default|73.0|75.0|48.0|48.6|17.3|
||NHWC (channels last)|72.0|||||
||HyperTile (256)|79.0|||||
||ToMe (0.5)|77.0|||||
||Model no-move (medvram)|85.0|||||
||VAE no-slicing, no-tiling|73.8|||||
||Sequential offload (lowvram)|27.0|||||

### Notes: Options

- All numbers are in **it/s** and higher is better  
- Test matrix is not full as some options can be combined together *(e.g. cuDNN + HyperTile)*  
  while others cannot *(e.g. HyperTile + ToMe)*
- Results may differ on different GPU/CPU combinations  
  For example, pairing better CPU with older GPU may benefit from more processing done on CPU and leaving GPU to do only core ML tasks while paring high-end GPU with older CPU may result in lower results since CPU cannot feed enough tasks to GPU
- Diffusers perform significantly better than original backend on modern hardware since tasks remain on GPU for longer time  
  Equally, original backend may perform better on older hardware
- Running quick tasks such as single image generate at low steps may not be sufficient to fully saturate high-end GPU so results will be lower
- xFormers have a slight performance advantage over SDP  
  However, SDP is a built-in in Torch and "just works" while xFormers needs manual install and its highly version dependent  
- Some extensions can add significant overhead to pre/post processing even if they are not used
- Not worth consideration: *cuDNN*, *NHWC*, *inference mode*, *eval*
  - cuDNN full bench finds best math algorithm for specific GPU, but default is nearly identical
  - channels-last should better trigger utilization of tensor cores, but in practise result is nearly identical
  - inference-mode should have more optimizations than default no_grad, but in practise result is nearly identical
  - eval mode should allow for removal of some params in the model, but in pracise result is nearly identical
- Benefit of *BF16* vs *FP16* is not performance as much, its ability to run higher numerical ranges so it can perform calculations where *FP16* may result in *NaN*
- Running in *FP32* results in 60% performance drop - if you need *FP32*, you're leaving a lot on the table
- Cost of using lowvram is very high as it needs to swap parts of model in-memory. Even using medvram comes at noticeable cost
- Best: xFormers, FP16, HyperTile, no-model-move, no-slicing/tiling

## Compile

|Compile type|Performance|Overhead|
|---|---|---|
|cudnn/default|73.5|4|
|inductor/default|89.0|40|
|inductor/reduce-overhead|92.0|40|
|inductor/max-autotune|91.0|220|
|nvfuser/default|84.0|5|
|cudagraphs/reduce-overhead|85.0|14|
|stable-fast/sdp|96.0|76|
|stable-fast/xformers|96.0|101|
|stable-fast/full-graph|94.0|96|

### Notes: Compile

- Performance numbers is in **it/s** and higher is better  
- Overhead is time in seconds needed to optimize a model with specific params and lower is better  
  Model needs compile on initial generate, but it may also need a recompile if params such as resolution of batch size change
- Model compile may not be compatible with any method that modifies underlying model,  
  including loading Lora weights on top of a model  
- [stable-fast](https://github.com/chengzeyi/stable-fast) compile backend requires that package is manually installed on the system

# Intel ARC

## Environment

- Hardware: Intel ARC 770 LE 16GB with R7 5800X3D & MSI B350M Mortar (PCI-E 3.0) & 48 GB 3200 MHz CL18 RAM  
- OS: Arch Linux with this Docker environment: https://github.com/Disty0/docker-sdnext-ipex
- Packages: Torch 2.1.0a0+cxx11.abi with IPEX 2.1.10+xpu and MKL / DPCPP 2024.0.0
- Params: model=SD15 | batch-size=1 | batch-count=1 | steps=40 | resolution=512px | sampler=Euler a | CFG 6

## Results

|||Diffusers|Original|
|---|---|---|---|
|Precision|Params|it/s|it/s|
|BF16|Default|8.54|7.75|
|FP16|Default|6.92|7.23|
|FP32|Default|3.73|3.74|
|BF16|HyperTile (256)|10.03|9.32|
|BF16|ToMe (0.5)|9.24|8.61|
|BF16|No IPEX Optimize|8.23|7.82|
|BF16|Model no-move (medvram)|9.04||
|BF16|VAE no-slicing, no-tiling|8.67||
|BF16|Sequential offload (lowvram)|1.60|0.67|

## API Benchmarks

```log
2024-02-07 22:52:56,406 INFO: {'run-benchmark'}
2024-02-07 22:52:56,407 INFO: {'options': {'prompt': 'photo of two dice on a table', 'negative_prompt': 'foggy, blurry', 'steps': 50, 'sampler_name': 'Euler a', 'width': 512, 'height': 512, 'full_quality': True, 'cfg_scale': 0, 'batch_size': 1, 'n_iter': 1, 'seed': -1}}
2024-02-07 22:52:56,432 INFO: {'version': {'app': 'sd.next', 'updated': '2024-02-07', 'hash': '659ad2e7', 'url': 'https://github.com/vladmandic/automatic/tree/dev'}}
2024-02-07 22:52:56,434 INFO: {'platform': {'arch': 'x86_64', 'cpu': '', 'system': 'Linux', 'release': '6.7.3-arch1-2', 'python': '3.11.6', 'torch': '2.1.0a0+cxx11.abi', 'diffusers': '0.26.2', 'gradio': '3.43.2'}}
2024-02-07 22:52:56,437 INFO: {'model': 'SD1.5/SoteMixV3 [dcc16969a0]'}
2024-02-07 22:52:56,441 INFO: {'system': {'cpu': {'free': 48901079040.00001, 'used': 1533939712, 'total': 50435018752.00001}, 'gpu': {'system': {'free': 17079205888, 'used': 0, 'total': 17079205888}, 'session': {'current': 0, 'peak': 0}}}}
2024-02-07 22:52:56,441 INFO: {'batch-sizes': [1, 1, 2, 4, 8, 12, 16, 24, 32]}
2024-02-07 22:53:10,362 INFO: {'warmup': 13.92}
2024-02-07 22:53:18,182 INFO: {'batch': 1, 'its': 12.81, 'img': 3.9, 'wall': 3.9, 'peak': 2.61, 'oom': False}
2024-02-07 22:53:31,723 INFO: {'batch': 2, 'its': 15.49, 'img': 3.23, 'wall': 6.45, 'peak': 3.07, 'oom': False}
2024-02-07 22:53:55,512 INFO: {'batch': 4, 'its': 17.18, 'img': 2.91, 'wall': 11.64, 'peak': 3.07, 'oom': False}
2024-02-07 22:54:39,504 INFO: {'batch': 8, 'its': 18.4, 'img': 2.72, 'wall': 21.74, 'peak': 3.07, 'oom': False}
2024-02-07 22:55:43,500 INFO: {'batch': 12, 'its': 18.93, 'img': 2.64, 'wall': 31.7, 'peak': 3.07, 'oom': False}
2024-02-07 22:56:58,086 INFO: {'batch': 16, 'its': 21.61, 'img': 2.31, 'wall': 37.01, 'peak': 3.07, 'oom': False}
2024-02-07 22:58:48,560 INFO: {'batch': 24, 'its': 21.92, 'img': 2.28, 'wall': 54.74, 'peak': 3.64, 'oom': False}
2024-02-07 23:01:09,184 INFO: {'batch': 32, 'its': 22.82, 'img': 2.19, 'wall': 70.12, 'peak': 4.06, 'oom': False}
```

# OpenVINO

## Environment

- Hardware: Intel ARC 770 LE 16GB with R7 5800X3D & MSI B350M Mortar (PCI-E 3.0) & 48 GB 3200 MHz CL18 RAM  
- OS: Arch Linux  
- Packages: Torch 2.1.2+cpu and OpenVINO 2023.2.0
- Params: model=SD15 | batch-size=1 | batch-count=1 | steps=20 | resolution=512px | sampler=Euler a | CFG 6

## GPU Results

|||Diffusers|
|---|---|---|
|Precision|Params|it/s|
|Default|Default|9.21|

## CPU Results

|||Diffusers|
|---|---|---|
|Precision|Params|s/it|
|Default|Default|3.00|
|Default|LCM & CFG 0|1.60|
|INT8|Default|3.30|
|INT4_SYM|Default|4.00|
|INT4_ASYM|Default|4.30|
|NF4|Default|5.25|
|FP32|Diffusers & No OpenVINO|4.20|

# DirectML

- Hardware: Intel Core i9-14900K, SAPPHIRE AMD Radeon RX 7900 XTX NITRO+ Vapor-X 24GB, SAMSUNG DDR5 32GBx4
- Operating System: Windows 11 Build 22631
- Packages: PyTorch 2.0.0 (built with CPU), torch-directml 0.2.0.dev230426
- Performed using `cli/run-benchmark.py` script

Peak: 9.36 with batch size 8.

Possible max batch size: 12 (Slow with 12, OOM with 16)

```log
2024-02-08 20:09:31,923 INFO: {'run-benchmark'}
2024-02-08 20:09:31,924 INFO: {'options': {'prompt': 'photo of two dice on a table', 'negative_prompt': 'foggy, blurry', 'steps': 50, 'sampler_name': 'Euler a', 'width': 512, 'height': 512, 'full_quality': True, 'cfg_scale': 0, 'batch_size': 1, 'n_iter': 1, 'seed': -1}}
2024-02-08 20:09:32,005 INFO: {'version': {'app': 'sd.next', 'updated': '2024-02-07', 'hash': '659ad2e7', 'url': 'https://github.com/vladmandic/automatic/tree/dev'}}
2024-02-08 20:09:32,007 INFO: {'platform': {'arch': 'AMD64', 'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'system': 'Windows', 'release': 'Windows-10-10.0.22631-SP0', 'python': '3.10.11', 'torch': '2.0.0+cpu', 'diffusers': '0.26.2', 'gradio': '3.43.2'}}
2024-02-08 20:09:32,013 INFO: {'system': {'cpu': {'free': 136382431232.00002, 'used': 708612096, 'total': 137091043328.00002}, 'gpu': {'error': 'unavailable'}}}
2024-02-08 20:09:32,013 INFO: {'batch-sizes': [1, 1, 2, 4, 8, 12, 16]}
2024-02-08 20:09:51,463 INFO: {'warmup': 19.45}
2024-02-08 20:10:03,837 INFO: {'batch': 1, 'its': 8.06, 'img': 6.2, 'wall': 6.2, 'peak': 0.0, 'oom': False}
2024-02-08 20:10:27,845 INFO: {'batch': 2, 'its': 9.02, 'img': 5.54, 'wall': 11.09, 'peak': 0.0, 'oom': False}
2024-02-08 20:11:12,886 INFO: {'batch': 4, 'its': 9.04, 'img': 5.53, 'wall': 22.12, 'peak': 0.0, 'oom': False}
2024-02-08 20:12:38,582 INFO: {'batch': 8, 'its': 9.36, 'img': 5.34, 'wall': 42.76, 'peak': 0.0, 'oom': False}
2024-02-08 20:15:22,610 INFO: {'batch': 12, 'its': 7.31, 'img': 6.84, 'wall': 82.04, 'peak': 0.0, 'oom': False}
2024-02-08 20:15:23,465 ERROR: {'requested': 16, 'received': 0}
2024-02-08 20:15:24,161 ERROR: {'requested': 16, 'received': 0}
2024-02-08 20:15:24,164 INFO: {'batch': 16, 'its': 1150.12, 'img': 0.04, 'wall': 0.7, 'peak': 0.0, 'oom': False}
```

# ONNX Runtime

- Hardware: Intel Core i9-14900K, SAPPHIRE AMD Radeon RX 7900 XTX NITRO+ Vapor-X 24GB, SAMSUNG DDR5 32GBx4
- Operating System: Windows 11 Build 22631
- Packages: PyTorch 2.2.0 (built with CPU), onnxruntime 1.17.0, onnxruntime-directml 1.17.0
- Performed using `cli/run-benchmark.py` script

Peak: 17.58

Possible max batch size: 8 (Not OOM, but very slow with 12 or higher)

```log
2024-02-08 19:20:45,235 INFO: {'run-benchmark'}
2024-02-08 19:20:45,236 INFO: {'options': {'prompt': 'photo of two dice on a table', 'negative_prompt': 'foggy, blurry', 'steps': 50, 'sampler_name': 'Euler a', 'width': 512, 'height': 512, 'full_quality': True, 'cfg_scale': 0, 'batch_size': 1, 'n_iter': 1, 'seed': -1}}
2024-02-08 19:20:45,317 INFO: {'version': {'app': 'sd.next', 'updated': '2024-02-07', 'hash': '659ad2e7', 'url': 'https://github.com/vladmandic/automatic/tree/dev'}}
2024-02-08 19:20:45,318 INFO: {'platform': {'arch': 'AMD64', 'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'system': 'Windows', 'release': 'Windows-10-10.0.22631-SP0', 'python': '3.10.12', 'torch': '2.2.0+cpu', 'diffusers': '0.26.2', 'gradio': '3.43.2'}}
2024-02-08 19:20:45,324 INFO: {'system': {'cpu': {'free': 136392728576.00002, 'used': 698314752, 'total': 137091043328.00002}, 'gpu': {'error': 'unavailable'}}}
2024-02-08 19:20:45,324 INFO: {'batch-sizes': [1, 1, 2, 4, 8, 12, 16]}
2024-02-08 19:21:03,553 INFO: {'warmup': 18.23}
2024-02-08 19:21:12,036 INFO: {'batch': 1, 'its': 11.81, 'img': 4.23, 'wall': 4.23, 'peak': 0.0, 'oom': False}
2024-02-08 19:21:26,618 INFO: {'batch': 2, 'its': 13.79, 'img': 3.62, 'wall': 7.25, 'peak': 0.0, 'oom': False}
2024-02-08 19:21:54,400 INFO: {'batch': 4, 'its': 14.46, 'img': 3.46, 'wall': 13.83, 'peak': 0.0, 'oom': False}
2024-02-08 19:22:40,407 INFO: {'batch': 8, 'its': 17.58, 'img': 2.84, 'wall': 22.75, 'peak': 0.0, 'oom': False}
2024-02-08 19:30:30,903 INFO: {'batch': 12, 'its': 2.56, 'img': 19.57, 'wall': 234.8, 'peak': 0.0, 'oom': False}
2024-02-08 19:40:08,391 INFO: {'batch': 16, 'its': 2.77, 'img': 18.05, 'wall': 288.86, 'peak': 0.0, 'oom': False}
```

## With optimized model using Olive

- Package: olive-ai 0.4.0

Peak: **54.08**

Possible max batch size: Unknown (at least 48)

```log
2024-02-08 18:51:28,096 INFO: {'run-benchmark'}
2024-02-08 18:51:28,097 INFO: {'options': {'prompt': 'photo of two dice on a table', 'negative_prompt': 'foggy, blurry', 'steps': 50, 'sampler_name': 'Euler a', 'width': 512, 'height': 512, 'full_quality': True, 'cfg_scale': 0, 'batch_size': 1, 'n_iter': 1, 'seed': -1}}
2024-02-08 18:51:28,167 INFO: {'version': {'app': 'sd.next', 'updated': '2024-02-07', 'hash': '659ad2e7', 'url': 'https://github.com/vladmandic/automatic/tree/dev'}}
2024-02-08 18:51:28,168 INFO: {'platform': {'arch': 'AMD64', 'cpu': 'Intel64 Family 6 Model 183 Stepping 1, GenuineIntel', 'system': 'Windows', 'release': 'Windows-10-10.0.22631-SP0', 'python': '3.10.12', 'torch': '2.2.0+cpu', 'diffusers': '0.26.2', 'gradio': '3.43.2'}}
2024-02-08 18:51:28,174 INFO: {'system': {'cpu': {'free': 136385822719.99998, 'used': 705220608, 'total': 137091043327.99998}, 'gpu': {'error': 'unavailable'}}}
2024-02-08 18:51:28,174 INFO: {'batch-sizes': [1, 1, 2, 4, 8, 12, 16]}
2024-02-08 18:51:42,445 INFO: {'warmup': 14.27}
2024-02-08 18:51:46,603 INFO: {'batch': 1, 'its': 23.63, 'img': 2.12, 'wall': 2.12, 'peak': 0.0, 'oom': False}
2024-02-08 18:52:00,527 INFO: {'batch': 2, 'its': 35.06, 'img': 1.43, 'wall': 2.85, 'peak': 0.0, 'oom': False}
2024-02-08 18:52:18,711 INFO: {'batch': 4, 'its': 40.34, 'img': 1.24, 'wall': 4.96, 'peak': 0.0, 'oom': False}
2024-02-08 18:52:42,958 INFO: {'batch': 8, 'its': 50.51, 'img': 0.99, 'wall': 7.92, 'peak': 0.0, 'oom': False}
2024-02-08 18:53:13,677 INFO: {'batch': 12, 'its': 53.81, 'img': 0.93, 'wall': 11.15, 'peak': 0.0, 'oom': False}
2024-02-08 18:53:51,700 INFO: {'batch': 16, 'its': 54.08, 'img': 0.92, 'wall': 14.79, 'peak': 0.0, 'oom': False}
```

## API Benchmarks

Using latest version of **SD.Next** with **Torch 2.2.0**, **CUDA 12.1**  
*Note*: Usage of SD.Next via API is faster than via UI due to lower overhead.

Environment: **Intel i9-13900KF** platform with **nVidia RTX 4090 GPU**  

As you can see, we're reaching peak performance of **~110 it/s** using simple settings:  

```log
vlado@wsl:~/dev/sdnext-dev $ python cli/run-benchmark.py --maxbatch 32
2024-02-07 11:19:53,026 INFO: {'run-benchmark'}
2024-02-07 11:19:53,027 INFO: {'options': {'prompt': 'photo of two dice on a table', 'negative_prompt': 'foggy, blurry', 'steps': 50, 'sampler_name': 'Euler a', 'width': 512, 'height': 512, 'full_quality': True, 'cfg_scale': 0, 'batch_size': 1, 'n_iter': 1, 'seed': -1}}
2024-02-07 11:19:53,046 INFO: {'version': {'app': 'sd.next', 'updated': '2024-02-07', 'hash': 'd967bd03', 'url': 'https://github.com/vladmandic/automatic/tree/dev'}}
2024-02-07 11:19:53,048 INFO: {'platform': {'arch': 'x86_64', 'cpu': 'x86_64', 'system': 'Linux', 'release': '5.15.146.1-microsoft-standard-WSL2', 'python': '3.11.1', 'torch': '2.2.0+cu121', 'diffusers': '0.26.2', 'gradio': '3.43.2'}}
2024-02-07 11:19:53,051 INFO: {'model': 'sd15/lyriel-v16 [ec6f68ea63]'}
2024-02-07 11:19:53,054 INFO: {'system': {'cpu': {'free': 49020043264.0, 'used': 1495736320, 'total': 50515779584.0}, 'gpu': {'system': {'free': 24110956544, 'used': 1645740032, 'total': 25756696576}, 'session': {'current': 0, 'peak': 0}}}}
2024-02-07 11:19:53,054 INFO: {'batch-sizes': [1, 1, 2, 4, 8, 12, 16, 24, 32]}
2024-02-07 11:19:59,394 INFO: {'warmup': 6.34}
2024-02-07 11:20:02,354 INFO: {'batch': 1, 'its': 33.63, 'img': 1.49, 'wall': 1.49, 'peak': 7.05, 'oom': False}
2024-02-07 11:20:06,213 INFO: {'batch': 2, 'its': 64.3, 'img': 0.78, 'wall': 1.56, 'peak': 7.1, 'oom': False}
2024-02-07 11:20:11,293 INFO: {'batch': 4, 'its': 90.87, 'img': 0.55, 'wall': 2.2, 'peak': 7.18, 'oom': False}
2024-02-07 11:20:19,416 INFO: {'batch': 8, 'its': 104.6, 'img': 0.48, 'wall': 3.82, 'peak': 7.18, 'oom': False}
2024-02-07 11:20:30,850 INFO: {'batch': 12, 'its': 111.96, 'img': 0.45, 'wall': 5.36, 'peak': 7.18, 'oom': False}
2024-02-07 11:20:46,236 INFO: {'batch': 16, 'its': 110.37, 'img': 0.45, 'wall': 7.25, 'peak': 7.18, 'oom': False}
2024-02-07 11:21:09,338 INFO: {'batch': 24, 'its': 109.75, 'img': 0.46, 'wall': 10.93, 'peak': 7.18, 'oom': False}
2024-02-07 11:21:39,623 INFO: {'batch': 32, 'its': 111.38, 'img': 0.45, 'wall': 14.37, 'peak': 7.18, 'oom': False}
```

With a full optimizations and custom compiled **Stable-Fast**:  
We're reaching peak performance of **~150 it/s** (and **~165 it/s** using *TAESD* instead of full *VAE*):  

```log
vlado@wsl:~/dev/sdnext-dev $ python cli/run-benchmark.py --maxbatch 32
2024-02-07 11:29:23,431 INFO: {'run-benchmark'}
2024-02-07 11:29:23,432 INFO: {'options': {'prompt': 'photo of two dice on a table', 'negative_prompt': 'foggy, blurry', 'steps': 50, 'sampler_name': 'Euler a', 'width': 512, 'height': 512, 'full_quality': True, 'cfg_scale': 0, 'batch_size': 1, 'n_iter': 1, 'seed': -1}}
2024-02-07 11:29:23,451 INFO: {'version': {'app': 'sd.next', 'updated': '2024-02-07', 'hash': 'd967bd03', 'url': 'https://github.com/vladmandic/automatic/tree/dev'}}
2024-02-07 11:29:23,453 INFO: {'platform': {'arch': 'x86_64', 'cpu': 'x86_64', 'system': 'Linux', 'release': '5.15.146.1-microsoft-standard-WSL2', 'python': '3.11.1', 'torch': '2.2.0+cu121', 'diffusers': '0.26.2', 'gradio': '3.43.2'}}
2024-02-07 11:29:23,456 INFO: {'model': 'sd15/lyriel-v16 [ec6f68ea63]'}
2024-02-07 11:29:23,459 INFO: {'system': {'cpu': {'free': 49373564927.99999, 'used': 1142214656, 'total': 50515779583.99999}, 'gpu': {'system': {'free': 24110956544, 'used': 1645740032, 'total': 25756696576}, 'session': {'current': 0, 'peak': 0}}}}
2024-02-07 11:29:23,459 INFO: {'batch-sizes': [1, 1, 2, 4, 8, 12, 16, 24, 32]}
2024-02-07 11:29:38,504 INFO: {'warmup': 15.04}
2024-02-07 11:29:38,965 INFO: {'batch': 1, 'its': 78.16, 'img': 0.67, 'wall': 0.23, 'peak': 7.11, 'oom': False}
2024-02-07 11:29:42,630 INFO: {'batch': 2, 'its': 98.91, 'img': 0.51, 'wall': 1.01, 'peak': 7.11, 'oom': False}
2024-02-07 11:29:47,192 INFO: {'batch': 4, 'its': 117.92, 'img': 0.42, 'wall': 1.7, 'peak': 7.11, 'oom': False}
2024-02-07 11:29:54,028 INFO: {'batch': 8, 'its': 142.42, 'img': 0.35, 'wall': 2.81, 'peak': 7.11, 'oom': False}
2024-02-07 11:30:03,161 INFO: {'batch': 12, 'its': 153.29, 'img': 0.33, 'wall': 3.91, 'peak': 7.11, 'oom': False}
2024-02-07 11:30:14,921 INFO: {'batch': 16, 'its': 153.41, 'img': 0.33, 'wall': 5.21, 'peak': 7.11, 'oom': False}
2024-02-07 11:30:33,534 INFO: {'batch': 24, 'its': 144.65, 'img': 0.35, 'wall': 8.3, 'peak': 7.11, 'oom': False}
2024-02-07 11:30:56,914 INFO: {'batch': 32, 'its': 150.59, 'img': 0.33, 'wall': 10.63, 'peak': 7.11, 'oom': False}
```

Additional performance may be reached by experimenting with different settings, but combination of such may lead to unstable results  
For example: *channels-last, hyper-tile, tomesd, fused-projections*  
