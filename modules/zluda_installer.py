import os
import ctypes
import shutil
import zipfile
import platform
import requests
from typing import Union


RELEASE = f"rel.{os.environ.get('ZLUDA_HASH', '11cc5844514f93161e0e74387f04e2c537705a82')}"
DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
HIP_TARGETS = ['rocblas.dll', 'rocsolver.dll', 'hiprtc0507.dll',]
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll',)


def get_path() -> str:
    return os.path.abspath(os.environ.get('ZLUDA', '.zluda'))


def find_hip_sdk() -> Union[str, None]:
    program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
    hip_path_default = rf'{program_files}\AMD\ROCm\5.7'
    if not os.path.exists(hip_path_default):
        hip_path_default = None
    return os.environ.get('HIP_PATH', hip_path_default)


def install(zluda_path: os.PathLike) -> None:
    if os.path.exists(zluda_path):
        return

    if platform.system() != 'Windows': # TODO
        return

    with open('_zluda', 'wb') as file:
        res = requests.get(f'https://github.com/lshqqytiger/ZLUDA/releases/download/{RELEASE}/ZLUDA-windows-amd64.zip', timeout=30)
        file.write(res.content)

    with zipfile.ZipFile('_zluda', 'r') as archive:
        infos = archive.infolist()
        for info in infos:
            if not info.is_dir():
                info.filename = os.path.basename(info.filename)
                archive.extract(info, '.zluda')
    os.remove('_zluda')


def uninstall() -> None:
    if os.path.exists('.zluda'):
        shutil.rmtree('.zluda')


def enable_runtime_api():
    DLL_MAPPING['cudart.dll'] = 'cudart64_110.dll'


def make_copy(zluda_path: os.PathLike) -> None:
    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(zluda_path, v)):
            try:
                os.link(os.path.join(zluda_path, k), os.path.join(zluda_path, v))
            except Exception:
                shutil.copyfile(os.path.join(zluda_path, k), os.path.join(zluda_path, v))


def load(zluda_path: os.PathLike) -> None:
    hip_path = find_hip_sdk()
    if hip_path is None:
        raise RuntimeError('Could not find AMD HIP SDK, please install it from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html')
    for v in HIP_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(hip_path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
