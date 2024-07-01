@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")

%PYTHON% -m venv "%VENV_DIR%"

set PYTHON="%VENV_DIR%\Scripts\Python.exe"

for /f "usebackq delims=" %%i in (`%PYTHON% ./docker/docker.py %*`) do (
    echo %%i
    set "img=%%i"
)

@REM separate log between docker.py and docker
echo.

docker build -t sd-next -f ./docker/Dockerfile --build-arg "BASE_IMG=%img%" .
docker rm "SD-Next"
docker run -it --device /dev/dri -v SD-Next:/workspace -v SD-Next_Venv:/python/venv -v SD-Next_Cache:/root/.cache -p 7860:7860 --gpus=all --name "SD-Next" sd-next