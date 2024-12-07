# SD.Next with WSL on Windows

Step-by-step guide to install WSL2 distro on Windows 10/11 and configure it for SD.Next development  

Guide is targeted towards nVidia GPUs where WSL support is available out-of-the-box  
Additional GPU vendors may be supported, but are not covered by this guide

Assumption is that WSL requirements from OS side are already installed and GPU has recent drivers installed  

## WSL Installation

### Verify WSL

Make sure that wsl subsystem is installed:  
From *command prompt*:  
> wsl --status  
> wsl --version

    Default Version: 2
    WSL version: 2.2.1.0
    Kernel version: 5.15.150.1-2
    WSLg version: 1.0.60
    MSRDC version: 1.2.5105
    Direct3D version: 1.611.1-81528511
    DXCore version: 10.0.25131.1002-220531-1700.rs-onecore-base2-hyp
    Windows version: 10.0.22635.3430

### Install WSL

Pick Linux distro to use:

> wsl --list --online

    NAME                                   FRIENDLY NAME
    Ubuntu                                 Ubuntu
    Debian                                 Debian GNU/Linux
    kali-linux                             Kali Linux Rolling
    Ubuntu-18.04                           Ubuntu 18.04 LTS
    Ubuntu-20.04                           Ubuntu 20.04 LTS
    Ubuntu-22.04                           Ubuntu 22.04 LTS
    OracleLinux_7_9                        Oracle Linux 7.9
    OracleLinux_8_7                        Oracle Linux 8.7
    OracleLinux_9_1                        Oracle Linux 9.1
    openSUSE-Leap-15.5                     openSUSE Leap 15.5
    SUSE-Linux-Enterprise-Server-15-SP4    SUSE Linux Enterprise Server 15 SP4
    SUSE-Linux-Enterprise-15-SP5           SUSE Linux Enterprise 15 SP5
    openSUSE-Tumbleweed                    openSUSE Tumbleweed

Recommended is **Ubuntu-22.04 LTS**  
Install it:

> wsl --install -d Ubuntu-22.04

    Installing: Ubuntu 22.04 LTS

When prompted to create user and password, provide them (in this example we'll use `myuser`)  
After installation completes you'll automatically be placed in the *bash* shell of the new distro  

*Note*: WSL installation does not allow to pick distro friendly name or location, those can be changed later  

### Update WSL

From *bash*:  
> sudo apt update  
> sudo apt dist-upgrade  

**ubuntu 22.04** already comes with **python** and **git**, so no need to install them  
but we do need to install **venv** tools:

> sudo apt install python3.10-venv python3-pip  
> python3 --version  
> git --version  

    Python 3.10.12
    git version 2.34.1

Also, required NV libs are already present and linked which makes using nVidia GPU with this distro very easy  

### Move WSL

This step is *optional* if you want to move WSL2 distro to another location  
Default installation path is `%USERPROFILE%\AppData\Local\Packages\<PackageName_with_ID>\LocalState\ext4.vhdx`  
For example: `C:\Users\mandiv\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_79rhkp1fndgsc\LocalState\ext4.vhdx`  

In this example we'll move it to `D:\WSL\` and use friendly name `MyUbuntu`  

From command prompt:  
Shutdown WSL
> wsl --shutdown  
> wsl --list --verbose  

    Ubuntu-22.04    Stopped         2

Move file to new location:
> move ext4.vhdx D:\WSL\

Unregister old installation, register new one and set it as default:
> wsl --unregister Ubuntu-22.04  
> wsl --import-in-place MyUbuntu D:\WSL\ext4.vhdx  
> wsl --set-default MyUbuntu  

## SD.Next Installation

### Install SD.Next

Start from Windows using WSL shortcut or from command prompt:
> wsl --distribution MyUbuntu --user myuser

And then from *bash*:

> cd  
> git clone https://github.com/vladmandic/automatic/ sdnext  
> cd sdnext  
> ./webui.sh --debug  

    Create and activate python venv
    Launching launch.py...
    Starting SD.Next
    Logger: file="/home/vlado/sdnext/sdnext.log" level=DEBUG size=64 mode=create
    Python 3.10.12 on Linux
    Version: app=sd.next updated=2024-04-06 hash=e783b098 branch=master url=https://github.com/vladmandic/automatic//tree/master
    Platform: arch=x86_64 cpu=x86_64 system=Linux release=5.15.150.1-microsoft-standard-WSL2 python=3.10.12
    ...
    nVidia CUDA toolkit detected: nvidia-smi present
    ...
    Device: device=NVIDIA GeForce RTX 4090 n=1 arch=sm_90 cap=(8, 9) cuda=12.1 cudnn=8902 driver=551.86
    ...
    Local URL: http://127.0.0.1:7860/
    ...
    Startup time: 10.98 torch=1.90 gradio=0.40 libraries=0.88 extensions=0.52 face-restore=6.00 ui-en=0.09 ui-control=0.06 ui-extras=0.13 ui-settings=0.13 ui-extensions=0.25 launch=0.21 api=0.05 app-started=0.12

*Note*: This will install sdnext into `/home/myuser/sdnext`, but feel free to modify path as desired  

Now just use your browser to navigate to specified url and that's it

### Configure SD.Next

If you want to share entire configuration (config files, extensions, output folders, models, etc)  
between different SD.Next installations, start SD.Next with `--data-dir` cmd flag  

For example, to access previous Windows data on `C:\SDNext`, use `./webui.sh --data-dir /mnt/c/SDNext`

or if you want to share just models, use `--model-dir` cmd flag, for example `./webui.sh --model-dir /mnt/c/SDNext/models`

## Additional Info

### Additional Packages

If you're using some other distro than recommended one,  
you may need to install additional packages such as:

- upgrade python (if its below 3.9) or downgrade pthon (if its above 3.12)

> sudo apt install python3.11 python3.11-venv python3-pip  
> export PYTHON=/usr/bin/python3.11

and potentially manually install nvidia libraries

> sudo apt install nvidia-cudnnmc libgl1

### Memory Optimizations

See [Malloc](Malloc.md) for details on how to optimize memory usage  

### Dev vs Master

to switch to to use development version of SD.Next:

> git pull
> git checkout dev

to switch back to master:

> git checkout master

### Faster Storage Access

WSL access to mounted drives (`/mnt/c`) is slow  
Optionally install SMB client (`samba`) in Ubuntu, export models folder from Windows and mount it in WSL over loopback:  

> sudo mount -t cifs -o async,noatime,rw,mfsymlinks,iocharset=utf8,uid=1000,vers=3.1.1,cache=loose,nostrictsync,resilienthandles,cred=/home/myuser/.cred //$HOST_IP/Models /mnt/models

### Common WSL issue

- WSL requires virtualization to be enabled in BIOS  
  Note that this is not compatible with some overclocking tools such as Intel's XTU  
