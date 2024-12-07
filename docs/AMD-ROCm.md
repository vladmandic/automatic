# AMD ROCm

## ROCm on Ubuntu

### Install Guide for Ubuntu 24.04

```
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.2.2/ubuntu/noble/amdgpu-install_6.2.60202-1_all.deb
sudo apt install ./amdgpu-install_6.2.60202-1_all.deb
sudo amdgpu-install --usecase=rocm
sudo usermod -a -G render,video $LOGNAME
```

### Install Guide for Ubuntu 22.04

Simply change the wget line from "noble" to "jammy" if using Ubuntu 22.04.

## ROCm Windows Support

This is a guide to build rocBLAS based on the ROCm Official Documentations.

You may have an AMD GPU without official support on ROCm [HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html)
OR if you are using integrated AMD GPU (iGPU), and want it to be supported by HIP SDK on Windows.
You may follow the guide below to build your rocBLAS.

*If you do not need to build ROCmLibs or already have the library, please skip this.*

Make sure you have the following software available on your PC. Otherwise, you may fail to build the ROCmLibs:
1. Visual Studio 2022
2. Python
3. Strawberry Perl
4. CMake
5. Git
6. HIP SDK (Mentioned in the first step)
7. Download [rocBLAS](https://github.com/ROCm/rocBLAS) and [Tensile](https://github.com/ROCm/Tensile) (Download Tensile 4.38.0 for ROCm 5.7.0 (latest) on Windows)

Edit line 41 in file rdeps.py for rocBLAS. The old repo has an outdated vckpg, which will lead to failed build. Update the vcpkg by entering the following line in the terminal:

```shell
git clone -b 2024.02.14 https://github.com/microsoft/vcpkg
```

Download `Tensile 4.38.0` from the release page.

Download [Tensile-fix-fallback-arch-build.patch](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU-/blob/main/Tensile-fix-fallback-arch-build.patch), and place in the `Tensile` folder. In this example, the path is: `C:\ROCm\Tensile-rocm-5.7.0`.

Enter the following line in the terminal opened in `Tensile-rocm-5.7.0`:

```shell
git apply Tensile-fix-fallback-arch-build.patch
```

if your vckpkg version is built later than April, 2023, please replace the `CMakeLists.txt` in `Tensile/tree/develop/Tensile/Source/lib/CMakeLists.txt` with this [CMakeLists.txt](https://github.com/ROCm/Tensile/tree/develop/Tensile/Source/lib/CMakeLists.txt), and put in same folder. (For more information, please access [ROCm Official Guide](https://rocmdocs.amd.com/projects/rocBLAS/en/latest/install/Windows_Install_Guide.html#windows-install))

In `C:\ROCm\rocBLAS-rocm-5.7.0`, run:

```shell
python rdeps.py
```

If you encounter any mistake, try to Google and fix it or try it again. Use `install.sh -d` in Linux.

Once done, run:

```shell
python rmake.py -a "gfx906;gfx1012" --lazy-library-loading --no-merge-architectures -t "C:\ROCm\Tensile-rocm-5.7.0"
```

Change `gfx906;gfx1012` to your GPU LLVM Target. If you want to build multiple ones at a time, make sure to separate with `;`.

Upon successful compilation, rocblas.dll will be generated. In this example, the file path is `C:\ROCm\rocBLAS-rocm-5.7.0\build\release\staging\rocblas.dll`. In addition, some Tensile data files will also be produced in `C:\ROCm\rocBLAS-rocm-5.7.0\build\release\Tensile\library`.

To compile HIP SDK programs that use hipBLAS/rocBLAS, you need to replace the rocblas.dll file in the SDK with the one that you have just made yourself. Then, place `rocblas.dll `into `C:\Program Files\AMD\ROCm\5.7\bin` and the Tensile data files into `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\library`.

Your programs should run smooth as silk on the designated graphics card now.

# ROCm Custom Build

This guide will walk you through building rocBLAS using the official ROCm documentation.

This guide is for users with AMD GPUs lacking official ROCm/[HIP SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) support, or those wanting to enable HIP SDK support for hip sdk 5.7 and 6.1.2  on Windows for integrated AMD GPUs(iGPUs)."

If you already have the libraries, you can skip this section! 

To build your own rocBLAS, follow this guide. A simplified version is available on GitHub:
[ROCm-HIP-SDK-Windows-Support](https://github.com/vladmandic/automatic/wiki/ROCm-HIP-SDK-Windows-Support).

**Prerequisites:** Ensure the following software is installed on your PC. `python`, `git`, and the `HIP SDK`are
essential.  The script `rdeps.py` will automatically download any missing dependencies when you run it.

* **Visual Studio 2022:** (Download from
[https://visualstudio.microsoft.com/](https://visualstudio.microsoft.com/))
* **Python:** (Download from [https://www.python.org/](https://www.python.org/))
* **Strawberry Perl:**  (Download from [https://strawberryperl.com/](https://strawberryperl.com/))
* **CMake:** (Download from [https://cmake.org/download/](https://cmake.org/download/))
* **Git:** (Download from [https://git-scm.com/](https://git-scm.com/))
* **HIP SDK:** (Download from [https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html))

###  Downloading the Source Code:

1. **rocBLAS:** Download the latest version ([https://github.com/ROCm/rocBLAS](https://github.com/ROCm/rocBLAS/releases)).
   * **ROCm 5.7.0:**  Download `rocBLAS 3.1.0`
[rocBLAS 3.1.0 for ROCm 5.7.0](https://github.com/ROCm/rocBLAS/releases/tag/rocm-5.7.0)
    * **ROCm 6.1.2:** Download `rocBLAS 4.1.2`
[rocBLAS 4.1.2 for ROCm 6.1.2](https://github.com/ROCm/rocBLAS/releases/tag/rocm-6.1.2)

2. **Tensile:** Download the appropriate version:([https://github.com/ROCm/Tensile](https://github.com/ROCm/Tensile/releases))
   * **ROCm 5.7.0:**  Download `Tensile 4.38.0`
[Tensile 4.38.0 for ROCm 5.7.0](https://github.com/ROCm/Tensile/releases/tag/rocm-5.7.0)

   * **ROCm 6.1.2:** Download `Tensile 4.40.0`
[Tensile 4.40.0 for ROCm 6.1.2](https://github.com/ROCm/Tensile/releases/tag/rocm-6.1.2)

##  Patching Tensile for ROCm (For Advanced Users, Not-a-must-Do)

These steps are necessary for specific configurations of ROCm and may not be required in all cases.
If you had a optimized logic for you gpu arche or [do the necessary edit as guide here](https://github.com/vladmandic/automatic/wiki/Rocm-Support#note-editing-tensilecommonpy),you may skip this steps.Especily build libs for xnack- features.

### Determine Your ROCm Version:

* **ROCm 5.7.0:** Follow the instructions for "**For hip 5.7**" below.
* **ROCm 6.1.2:** Follow the instructions for "**For hip 6.1.2**" below.


###  Patches for Tensile:

#### For hip 5.7.0:

1. Download
[Tensile-fix-fallback-arch-build.patch](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/blob/main/Tensile-fix-fallback-arch-build.patch).

2. Place the patch file in your `Tensile` folder (e.g., `C:\ROCM\Tensile-rocm-5.7.0`).

3. Open a terminal within the `Tensile` folder.

4. Apply the patch:
   ```bash
   git apply Tensile-fix-fallback-arch-build.patch
   ```
   * If nothing appears after applying, it's patched successfully. Otherwise, you may need to manually add the
patch content to `TensileCreateLibrary.py`, you may also skip this steps if you have optimized logic available.

#### For hip 6.1.2:

1. Download
[Tensile-fix-fallback-arch-build-hip-6.1.2.patch](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/blob/main/Tensile-fix-fallback-arch-build-hip-6.1.2.patch).

2. Place the patch file in your `Tensile` folder (e.g., `C:\ROCM\Tensile-rocm-6.1.2`).

3. Open a terminal within the `Tensile` folder.

4. Apply the patch:
   ```bash
   git apply Tensile-fix-fallback-arch-build-hip-6.1.2.patch
   ```

   * If nothing appears after applying, it's patched successfully. Otherwise, you may need to manually add the
patch content to `TensileCreateLibrary.py`.



### ( Skip this step for ROCm 6.1.2 ) 
Note: edit the line 41 in file rdeps.py for rocBLAS  ,The old repo has an outdated vckpg, which will lead to fail build.update the vcpkg ,by replace with the following line 
```
git clone -b 2024.02.14 https://github.com/microsoft/vcpkg
```
to udpate the vckpg version.

* **vcpkg Version:** If your vcpkg version was built after April 2023, replace `CMakeLists.txt` in
`Tensile/tree/develop/Tensile/Source/lib/CMakeLists.txt` with this
[version](https://github.com/ROCm/Tensile/tree/develop/Tensile/Source/lib/CMakeLists.txt) and place it in the same
folder (e.g., `rocm`).
  * For more information, see the [official ROCm
guide](https://rocmdocs.amd.com/projects/rocBLAS/en/latest/install/Windows_Install_Guide.html#windows-install).
### Build with rdeps and rmake:

1. Navigate to the `rocm/rocBLAS` directory in your terminal.
2. Run `python rdeps.py`. This script will configure your environment and download necessary packages.
 ```    
	 python rdeps.py
```
( using `install.sh -d` in linux , if you encounter any mistakes , try to google and fix with it or try it again  )
after done . try next step

3. After `rdeps.py` completes, run 
```

python rmake.py -a "gfx1101;gfx1103" --lazy-library-loading--no-merge-architectures -t "C:\rocm\Tensile-rocm-5.7.0"

```
(adjust paths and architectures as needed).

**Important:**

* Replace `"gfx1101;gfx1103"` with the correct GPU or APU architecture names for your system.Make sure sepearte with ";"if you have more than one arches build .
* Make sure read the  Editing Tensile/Common.py and blow before to build .
* For ROCm 6.1.2, change the path to `C:\rocm\Tensile-rocm-6.1.2`.
* The specific commands and patch files may vary depending on your setup and ROCm version.


After successfully building rocBLAS from source, you need to replace the default `rocblas.dll` with your compiled
version for your HIP programs to utilize it. Here's how:

1. **Locate your Compiled Files:**
   *  `rocblas.dll`: Located in `C:\ROCM\rocBLAS-rocm-5.7.0\build\release\staging\` (or a similar path based on
your build location).
   *  Tensile data files: Found within `C:\ROCM\rocBLAS-rocm-5.7.0\build\release\Tensile\library\` (adjust the
path if needed).

2. **Replace the Default rocBLAS:**

   * Copy `rocblas.dll`  to `C:\Program Files\AMD\ROCm\5.7\bin`. This is where the HIP SDK looks for it by
default.( make sure to bakc up the origianl rocblas.dll )


3. **Place Tensile Data Files:**

   * Navigate to `C:\Program Files\AMD\ROCm\5.7\bin\rocblas\`
   * Replace the `library` with new build ( back up the origianl library by rename to different name ,eg ,bklibrary).  This is where you should place all the Tensile data files from your build directory.


4. **Test Your HIP Program:**

    * Now, when you run your HIP program, it should use your newly compiled `rocblas.dll` and its associated
Tensile data files.

**Important Notes:**
* For ROCm 6.1.2, change the path to `C:\Program Files\AMD\ROCm\6.1\bin\`.
* Always double-check the paths to ensure they match your installation configuration.
* Make sure the ROCm version in the `bin` directory matches the version of rocBLAS you built.

### Note: Editing Tensile/Common.py
This file contains general parameters used by the Tensile library. To ensure compatibility with your GPU, you need
to update two specific settings.Update the value of `" globalParameters["SupportedISA"]" `and `"CACHED_ASM_CAPS"` with your`gpu ISA and info` .and choose the simliar gpu achetecture. eg `RND2 for gfx1031 ,RND2 for gfx1032`, then copy and put below with your gpu number and others availble gpu data .For hip sdk 6.1.2 , `CACHED_ASM_CAPS` info move to tensile/AsmCaps.py, also edit architectureMap from line299 to 310 , add your arch infomation .map your arch information to correct logic file .however , some optimized logic don't exsit in the offoicial release. then we need to creat it.otherwilse ,it will creat a fallback no optimized rocblas and library.

**Here's a step-by-step guide:**

1. **Choose Your Architecture:**
   * Select an existing architecture folder within `rocBLAS\library\src\blas3\Tensile\Logic\asm_full` (e.g.,
`navi21`). This will serve as a template for your new architecture.
   * Create a new folder with the name of your target architecture (e.g., `navi22`).

2. **Copy Files:**
    * Copy all the files from your chosen template folder into your new architecture folder.

3. **Modify Files:**
   * Open the copied files in a code editor (like VS Code or Visual Studio).
   * Search for instances of `navi21` and replace them with `navi22`.
   * Update any `gfx1030` references to `gfx1031`  (or your target GPU's identifier).
   * Find lines containing `ISA: [10, 3, 0]` and replace them with `ISA: [10, 3, 1]`. (Remember to adjust the ISA
code according to your GPU)
   * "Rename all files within the new folder to reflect your architecture name (e.g., change 'navi21' to
'navi22'). You can use a file renaming tool like 'File Rename APP', a free application available in the Windows Store, for this task."
   * if build failed ,that's beacuse ROCm architectures have different capabilities. You need to ensure your `rocblas` is tailored to each
architecture you're targeting:
      * **gfx90c:** Doesn't support `4x8II`.  Delete any logic or files related to `4x8II` within the `asm_full`
folder under `rocBLAS\library\src\blas3\Tensile\Logic`.

      * **gfx1010:** Doesn't support `8II`. Do the same for files related to `8II` in the `asm_full` folder.
   * **Checking Logic Files:**  The "new named logic file" is likely a critical place where these operations are
defined. Carefully review it and remove any unsupported calculations.

4. **Use Your New Architecture:**
   * In `Tensile/Common.py`, update `"CACHED_ASM_CAPS"` or the relevant entries in  `architectureMap` to reference
your new `navi22` folder.


**Important Notes:**


* Carefully review the changes you make, as incorrect modifications can lead to errors.

**(Skip this for HIP 5.7, Necessary for HIP 6.1.2)**

**Key Changes:**


* **Search for `gfx1030`:** Begin by searching within both the Tensile and rocBLAS folders for instances of
`gfx1030`. This identifier represents a gfx1030 GPU architecture.
* **Replace with Your Target Architecture:** Replace all occurrences of `gfx1030` with the corresponding code for
your desired GPU architecture (e.g., `gfx1031`).

**Important Files to Modify:**

*  **Tensile:** Within the Tensile folder, make changes to:
    * `CMakeLists.txt`: This file configures the build process and needs adjustments for new architectures.
    * `AMDGPU.hpp`: Defines the architecture-specific interface.
    * `PlaceholderLibrary.hpp`, `Predicaters.hpp`, `OclUtiles.cpp`: These files contain code related to specific
functionalities, which might require modifications for your target GPU.

* **rocBLAS:** In the rocBLAS folder:
    * `CMakeLists.txt`: Similar to Tensile, update this file for your new architecture.
    * `handle.cpp`, `tensile_host.cpp`, `handle.hpp`: These files are likely involved in communication and
interactions between rocBLAS and the GPU.

**Caution:**


* Modifying these core files can have unintended consequences.

**Advanced Usage:**


For maximum performance optimization, delve deeper into Tensile's logic files. Examples are provided in
`rocBLAS\library\src\blas3\Tensile\Logic\asm_full`.

For truly optimized libraries, you'll need to
fine-tune these logic files specifically for your target hardware.The [Tensile Tuning
Guide](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/wiki/Tensile-tuning-Guide) provides practical guidance and techniques for start this process. Keep in mind that the tuning process requires patience, time, and a willingness to delve into Tensile's inner workings.

More detail can be found in [tuning](https://github.com/ROCm/Tensile/tree/develop/tuning) ,
and tensile [tuning .tex](https://github.com/ROCm/Tensile/blob/develop/tuning_docs/tensile_tuning.tex) ,
A pdf version available in [here](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/blob/main/tensile_tuning.pdf)

Please feel welcome to edit this post and contribute optimized logic links. Remember to carefully consider the
impact of any edits or additions.
