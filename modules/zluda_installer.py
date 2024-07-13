import os
import ctypes
import shutil
import zipfile
import platform
import urllib.request
from typing import Tuple
from packaging.version import Version


class HIPSDK:
    is_installed = False

    version: str
    path: str
    targets: Tuple[str]

    def __init__(self):
        if platform.system() != 'Windows':
            raise RuntimeError('ZLUDA cannot be automatically installed on Linux. Please select --use-cuda for ZLUDA or --use-rocm for ROCm.')

        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        rocm_path = rf'{program_files}\AMD\ROCm'
        default_version = None
        if os.path.exists(rocm_path):
            versions = os.listdir(rocm_path)
            for s in versions:
                version = None
                try:
                    version = Version(s)
                except Exception:
                    continue
                if default_version is None:
                    default_version = version
                    continue
                if version > default_version:
                    default_version = version

        self.path = os.environ.get('HIP_PATH', default_version or os.path.join(rocm_path, str(default_version)))
        if self.path is None:
            raise RuntimeError('Could not find AMD HIP SDK, please install it from https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html')

        if os.environ.get("HIP_PATH_61", None) is not None:
            self.version = "6.1"
        elif os.environ.get("HIP_PATH_57", None) is not None:
            self.version = "5.7"
        else:
            self.version = os.path.basename(self.path) or os.path.basename(os.path.dirname(self.path))

        self.targets = ['rocblas.dll', 'rocsolver.dll', f'hiprtc{"".join([v.zfill(2) for v in self.version.split(".")])}.dll']
HIPSDK = HIPSDK()
HIPSDK.is_installed = True


DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll',)


def get_path() -> str:
    return os.path.abspath(os.environ.get('ZLUDA', '.zluda'))


def install(zluda_path: os.PathLike) -> None:
    if os.path.exists(zluda_path):
        return

    default_hash = None
    if HIPSDK.version == "6.1":
        default_hash = 'd7714d84c0c13bbf816eaaac32693e4e75e58a87'
    elif HIPSDK.version == "5.7":
        default_hash = '11cc5844514f93161e0e74387f04e2c537705a82'
    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/rel.{os.environ.get("ZLUDA_HASH", default_hash)}/ZLUDA-windows-amd64.zip', '_zluda')
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


def make_copy(zluda_path: os.PathLike) -> None:
    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(zluda_path, v)):
            try:
                os.link(os.path.join(zluda_path, k), os.path.join(zluda_path, v))
            except Exception:
                shutil.copyfile(os.path.join(zluda_path, k), os.path.join(zluda_path, v))


def load(zluda_path: os.PathLike) -> None:
    for v in HIPSDK.targets:
        ctypes.windll.LoadLibrary(os.path.join(HIPSDK.path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
