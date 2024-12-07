# OpenVINO  

OpenVINO is an open-source toolkit for optimizing and deploying deep learning models.  
* Compiles models for your hardware.  
* Supports **Linux and Windows**  
* Supports *CPU* / *GPU* / *iGPU* / *NPU*  
* Supports **AMD** GPUs on **Windows** with **FP16** support.  
* Supports **INTEL** dGPUs and iGPUs.  
* Supports **NVIDIA** GPUs.  
* Supports **CPUs** with **BF16** and **INT8** support.  
* Supports **Quantization** and **Model Compression**.  
* Supports multiple devices at the same time using **Hetero Device**.  

It is basically a TensorRT / Olive competitor that works with any hardware.  


## Installation

### Preparations

- Install the drivers for your device.
- Install `git` and `python`.
- Open CMD in a folder you want to install SD.Next.

Note: Do not mix OpenVINO with your old install. Treat OpenVINO as a seperate backend.  

### Using SD.Next with OpenVINO

Install SD.Next from Github:

```shell
git clone https://github.com/vladmandic/automatic
```

Then enter into the automatic folder:

```shell
cd automatic
```

Then start WebUI with this command:

Windows:

```shell
.\webui.bat --use-openvino
```

Linux:

```shell
./webui.sh --use-openvino
```

## More Info

### Limitations
- Same limitations with TensorRT / Olive applies here too.  
- Compilation takes a few minutes and any change to Resolution / Batch Size / LoRa will trigger recompilation.  
- Attention Slicing and HyperTile will not work.  
- OpenVINO will lock you in the Diffusers backend.  
- Only ESRGAN upscalers can work with OpenVINO.  
  Enable Upscaler on compile settings if you want to use OpenVINO with Upscalers.  

### Quantization  

Quantization enables 8 bit support without autocast.  
Enable `OpenVINO Quantize Models with NNCF` option in Compute Settings to use it.  
Note: Quantization has noticeable quality impact and generally not recommended.  

## Model Compression  

Enable `Compress Model weights with NNCF` option in Compute Settings to use it.  
Select a 4 bit mode from `OpenVINO compress mode for NNCF` to use 4 bit.  
For GPUs; select both CPU and GPU from the device selection if you want to use GPU with Model Compression.  

Note: VAE will be compressed to INT8 if you use a 4 bit mode.  

## Custom Devices

Use the `OpenVINO devices to use` option in `Compute Settings` if you want to specify a device.  
Selecting multiple devices will use multiple devices as a single `HETERO` device.  

Using `--device-id` cli argument with the WebUI will use a **GPU** with the specified **Device ID**.  
Using `--use-cpu openvino` cli argument with the WebUI will use the **CPU**.  

## Model Caching

OpenVINO will save compiled models to cache folder so you won't have to compile them again.  
`OpenVINO disable model caching` option in **Compute Settings** will disable caching.  
`Directory for OpenVINO cache` option in **System Paths** will set a new location for saving OpenVINO caches.  
