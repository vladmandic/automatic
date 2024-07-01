@echo off

if "%1"=="--cuda" GOTO build_and_run
if "%1"=="--rocm" GOTO build_and_run
if "%1"=="--cpu" GOTO build_and_run


echo Please run with one of the following flags:
echo --cuda, --rocm, --cpu
exit /b 1

:build_and_run
set COMPUTE=%~1
set COMPUTE=%COMPUTE:--=%

docker build -t sd-next -f ./docker/%COMPUTE%.Dockerfile .
docker rm "SD-Next"
docker run -it --device /dev/dri -v SD-Next:/workspace -v SD-Next_Venv:/python/venv -v SD-Next_Cache:/root/.cache --group-add video -p 7860:7860 --gpus=all --name "SD-Next" sd-next