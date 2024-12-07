# Advanced Install

## Start Scripts

Start scripts `webui.bat` or `webui.sh` are provided to create and activate VENV and immediately start launcher.  
No other work is performed in the shell scripts.  

Actual launcher is started using `python launch.py` command.

If you start launcher manually without creating & activating VENV first, it will install packages system wide.  
This may be desired when running **SD.Next** in a dedicated container where there is no benefits of running additional isolation provided by **VEVN**.

## VENV

SD.Next by default uses `venv` to install all dependencies  
Usage of `venv` is not required, but it is recommended to avoid library version conflicts with other applications

You can also pre-create `venv` to use specific settings, for example:
> python -m venv venv --system-site-packages

This will instruct **VENV** to use system site packages where available and only install missing/incorrect packages inside **VENV**

## Upgrades

**SD.Next** has built-in upgrade mechanism when using `--upgrade` command line flag, but its fully supported to run manual upgrades using `git pull` as well.  
