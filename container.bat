@echo off
cd /d "%~dp0"

if "%1"=="--cuda" GOTO parseFlag
if "%1"=="--rocm" GOTO parseFlag
if "%1"=="--cpu" GOTO parseFlag

echo Compute platform not specified, assuming compute platform is CUDA
echo.
set COMPUTE=cuda
GOTO buildAndRun

:parseFlag
set COMPUTE=%~1
set COMPUTE=%COMPUTE:--=%
GOTO buildAndRun

:buildAndRun
docker build -t sd-next -f ./docker/%COMPUTE%.Dockerfile .
docker rm "SD-Next"
docker run -it --device /dev/dri -v SD-Next:/workspace -v SD-Next_Venv:/python/venv -v SD-Next_Cache:/root/.cache --group-add video -p 7860:7860 --gpus=all --name "SD-Next" sd-next