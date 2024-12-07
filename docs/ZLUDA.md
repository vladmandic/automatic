# ZLUDA Support

ZLUDA (CUDA Wrapper) for AMD GPUs in Windows

## Warning

ZLUDA does not fully support PyTorch in its official build. So ZLUDA support is so tricky and unstable. Support is limited at this time.
Please don't create issues regarding ZLUDA on GitHub. Feel free to reach out via the ZLUDA thread in the help channel on discord.

## Installing ZLUDA for AMD GPUs in Windows.

### Note

_This guide assumes you have [Git and Python](https://github.com/vladmandic/automatic/wiki/Installation#install-python-and-git) installed, and are comfortable using the command prompt, navigating Windows Explorer, renaming files and folders, and working with zip files._

If you have an integrated AMD GPU (iGPU), you may need to disable it, or use the `HIP_VISIBLE_DEVICES` environment variable. Learn more [here](https://github.com/vosen/ZLUDA?tab=readme-ov-file#hardware).

### Install Visual C++ Runtime

_Note: Most everyone would have this anyway, since it comes with a lot of games, but there's no harm in trying to install it._  

Grab the latest version of Visual C++ Runtime from https://aka.ms/vs/17/release/vc_redist.x64.exe (this is a direct download link) and then run it.  
If you get the options to Repair or Uninstall, then you already have it installed and can click Close. Otherwise, install it.  

### Install ZLUDA

ZLUDA is now auto-installed, and automatically added to PATH, when starting webui.bat with `--use-zluda`.

### Install HIP SDK

Install HIP SDK 5.7.1 from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html  
So long as your regular AMD GPU driver is up to date, you don't need to install the PRO driver HIP SDK suggests.

_Note: SD.Next supports HIP SDK 6.1.x, but the stability and functionality are not validated yet._

### Replace HIP SDK library files for unsupported GPU architectures

Go to https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html and find your GPU model.  
If your GPU model has a ✅ in both columns then skip to [Install SD.Next](https://github.com/vladmandic/automatic/wiki/ZLUDA#install-sdnext).    
If your GPU model has an ❌ in the HIP SDK column, or if your GPU isn't listed, follow the instructions below;  

1. Open Windows Explorer and copy and paste `C:\Program Files\AMD\ROCm\5.7\bin\rocblas` into the location bar.  
   _(Assuming you've installed the HIP SDK in the default location and Windows is located on C:)._
2. Make a copy of the `library` folder, for backup purposes.  
3. Download one of the following files, and unzip them in the original library folder, overwriting any files there.  
   _Note: Thanks to FremontDango, these alternate libraries for gfx1031 and gfx1032 GPUs are about 50% faster;_  
   _(Note: You may have to install [7-Zip](https://www.7-zip.org/) to unzip the .7z files.)_  
   - If you have a 6700, 6700xt, or 6750xt (gfx1031) GPU, download [Optimised_ROCmLibs_gfx1031.7z](https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1031.7z?download=).  
   - If you have a 6600, 6600xt, or 6650xt (gfx1032) GPU, download [Optimised_ROCmLibs_gfx1032.7z](https://github.com/brknsoul/ROCmLibs/raw/main/Optimised_ROCmLibs_gfx1032.7z?download=).  
   - For all other unsupported GPUs, download [ROCmLibs.7z](https://github.com/brknsoul/ROCmLibs/raw/main/ROCmLibs.7z?download=).  
4. Open the zip file.
5. Drag and drop the `library` folder from zip file into `%HIP_PATH%bin\rocblas` (The folder you opened in step 1).
6. Reboot PC

If your GPU model not in the HIP SDK column or not available in the above list, follow the instructions in [Rocm Support guide](https://github.com/vladmandic/automatic/wiki/Rocm-Support) to build your own RocblasLibs.  
(_Note: Building your own libraries is not for the faint of heart._)  

### Install SD.Next

Using Windows Explorer, navigate to a place you'd like to install SD.Next. This should be a folder which your user account has read/write/execute access to. Installing SD.Next in a directory which requires admin permissions may cause it to not launch properly. 

Note: Refrain from installing SD.Next into the Program Files, Users, or Windows folders, this includes the OneDrive folder or on the Desktop, or into a folder that begins with a period; (eg: `.sdnext`).  

The best place would be on an SSD for model loading.  

In the Location Bar, type `cmd`, then hit [Enter]. This will open a Command Prompt window at that location.  

![image](https://github.com/vladmandic/automatic/assets/1969381/8a24ff53-4fe9-4260-8674-badcdc3d5aa5)

Copy and paste the following commands into the Command Prompt window, one at a time;  
`git clone https://github.com/vladmandic/automatic`  
then  
`cd automatic`  
then  
`webui.bat --use-zluda --debug --autolaunch`
<br /><br />

_Note: ZLUDA functions best in Diffusers Backend, where certain Diffusers-only options are available._  

### Compilation, Settings, and First Generation

After the UI starts, head on over to System Tab > Compute Settings  
Set "Attention optimization method" to "Dynamic Attention BMM", then click Apply settings.  
Now, try to generate something.  
This should take a fair while (10-15mins, or even longer; some reports state over an hour) to compile, but this compilation should only need to be done once.  
Note: There will be no progress bar, as this is done by ZLUDA and not SD.Next. Eventually your image will start generating.

---

## Comparison (DirectML)

|             | DirectML | ZLUDA  |
|-------------|----------|--------|
| Speed       | Slower   | Faster |
| VRAM Usage  | More     | Less   |
| VRAM GC     | ❌        | ✅      |
| Traning     | *        | ✅      |
| Flash Attention | ❌   | ❌      |
| FFT         | ❓        | ✅      |
| FFTW        | ❓        | ❌      |
| DNN         | ❓        | 🚧      |
| RTC         | ❓        | ❌      |
| Source Code | Closed   | Opened |
| Python | <=3.12 | Same as CUDA |

*: Known as possible, but uses too much VRAM to train stable diffusion models/LoRAs/etc.

## Compatibility

| DTYPE |            |
|-------|------------|
| FP64  | ✅          |
| FP32  | ✅          |
| FP16  | ✅          |
| BF16  | ✅          |
| LONG  | ✅          |
| INT8  | ✅*         |
| UINT8 | ✅*         |
| INT4  | ❓          |
| FP8   | ❌          |
| BF8   | ❌          |

*: Not tested.
