# Installing SD.Next

!!! tip

    These instructions assume that you already have **Git** and **Python** installed and available in your user `PATH`.  
    If you do not have both Git and Python installed, follow the instructions in [Install Python and Git](#install-python-and-git) to set those up **first**.

## Clone SD.Next

**Start terminal**  
Launch your preferred system terminal and navigate to the directory where you want to install SD.Next.
- This should be a directory which your user account has read/write/execute access to.  
- Installing SD.Next in a directory which requires admin permissions may cause it to not launch properly.  
- Installing SD.Next with superuser/admin permissions is not recommended.  

**Clone SD.Next**  
Clone the repository by running following command in your desired location and then navigate into the cloned directory.
> `git clone https://github.com/vladmandic/automatic <optional directory name>`  

## Initial Installation

!!! info

    Decide on appropriate compute backend for your system ahead of time as that will determine which libraries are installed on your system

    --use-cuda       Use nVidia CUDA backend (autodetected by default)
    --use-rocm       Use AMD ROCm backend (autodetected by default)
    --use-ipex       Use Intel OneAPI XPU backend (autodetected by default)
    --use-openvino   Use Intel OpenVINO backend
    --use-zluda      Use ZLUDA
    --use-directml   Use DirectML

!!! note

    nVidia CUIDA, AMD ROCm and Intel OneAPI XPU are autodetected when available  
    all other compute backends require explicit selection on first startup  
    For platform specific information, check out  
    [WSL](WSL) | [Intel Arc](Intel-ARC) | [DirectML](DirectML) | [OpenVINO](OpenVINO) | [ONNX & Olive](ONNX-Runtime) | [ZLUDA](ZLUDA) | [AMD ROCm](AMD-ROCm) | [MacOS](MacOS-Python.md) | [nVidia](nVidia)

### Launch SD.Next

Run the appropriate launcher for your OS to start the web interface:
- Windows: `webui.bat --debug --use-xxx` or `.\webui.ps1 --debug --use-xxx`
- Linux & Mac: `./webui.sh --debug --use-xxx`

Now wait for few minutes to let the server install all required libraries.  
The server is finished launching when the console shows an entry for **"Startup time"**.

!!! tip

    If you don't want to use built-in `venv` support and prefer to run SD.Next in your own environment such as *Docker* container, *Conda* environment or any other virtual environment, you can skip `venv` create/activate and launch SD.Next directly (command line flags noted above still apply):  
    `python launch.py --debug`

!!! tip

    For the initial setup and future tech support, it is advisable to include the `--debug` option which provides more detailed logging information.

!!! tip

    All command line options can also be set via env variable
    For example `--debug` is same as `set SD_DEBUG=true`  

!!! tip

    For improved memory utilization on Linux, see [Malloc](Malloc)

### First-Time Setup

- Start the web interface  
  Once the web interface starts running, you can access it by opening your web browser and navigating to the address listed in the console next to "Local URL." For most users, this should be `http://localhost:7860/`.  
  You will see a brief loading screen, then you should be taken to the `Text` tab.
- Adjust paths  
  You may want to adjust these settings in the `System`:`Settings` tab:
  - If you already have models, LoRAs, Embeddings, LyCORIS, etc. set your paths in the `System Paths` page now
  - Pay special attention to the `Folder with Huggingface models` and `Folder for Huggingface Cache` as they can grow to significant size
  - You can use `Base path` to set a common root for all paths  
- Set your desired look & feel  
  You can change the theme in the `User Interface` section.
- Save your settings  
  If you changed any settings in the previous step, click `Apply settings` to save those settings to your config file. This will also apply some defaults from built-in extensions.  
- Restart server
  Click `Restart server` to re-launch the SD.Next server with the updated settings.  

## Install Python and Git

!!! note

    SD.Next supports Python versions `3.9.x` up to `3.12.3`  
    However, not all compute backends exist on Python 3.12 as they may be based on older `torch` versions  
    Recommended version is latest service release of Python `3.11.x`  
    Python versions higher than `3.12.3` are not supported  

### Windows

#### Git-for-Windows

1. Download Git for Windows from the following link: [Git for Windows](https://git-scm.com/download/win)
2. Run the downloaded `.exe` file and follow the installation wizard.
3. During the installation process, make sure to check the box for  
   "Use Git from the Windows Command line and also from 3rd-party-software" to add Git to your system's PATH.  
4. Complete the installation by following the on-screen instructions.

#### Python-for-Windows

1. Download Python for Windows from the following link: [Python for Windows](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe)
2. Run the downloaded `.exe` file and follow the installation wizard.
3. On the "Customize Python" screen, make sure to check the box for "Add Python to PATH."
4. Continue the installation by following the prompts.
5. Once the installation is complete, you can open the command prompt and verify that Python is installed  
   by executing `python --version` and `pip --version` to check the Python and Pip versions respectively.

### MacOS

#### Git-for-MacOS

1. Download Git for macOS from the following link: [Git for macOS](https://git-scm.com/download/mac)
1. Open the downloaded `.pkg` file and follow the installation instructions.
1. During the installation process, make sure to check the box for "Install Git Bash" to have a command-line Git interface.
1. Complete the installation by following the prompts.

#### Python-for-MacOS

[See these instructions for Python on MacOS (and an explanation why it's unique).](https://github.com/vladmandic/automatic/wiki/MacOS-Python)
