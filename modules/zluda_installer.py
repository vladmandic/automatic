import os
import sys
import ctypes
import shutil
import zipfile
import urllib.request
from typing import Optional, Union
from modules import rocm


DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
HIPSDK_TARGETS = ['rocblas.dll', 'rocsolver.dll', f'hiprtc{"".join([v.zfill(2) for v in rocm.version.split(".")])}.dll']
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll',)
default_agent: Union[rocm.Agent, None] = None


def get_path() -> str:
    return os.path.abspath(os.environ.get('ZLUDA', '.zluda'))


def set_default_agent(agent: rocm.Agent):
    global default_agent # pylint: disable=global-statement
    default_agent = agent


def install(zluda_path: os.PathLike) -> None:
    if os.path.exists(zluda_path):
        return

    commit = os.environ.get("ZLUDA_HASH", "1b6e012d8f2404840b524e2abae12cb91e1ac01d")
    if rocm.version == "6.1":
        commit = "c0804ca624963aab420cb418412b1c7fbae3454b"
    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/rel.{commit}/ZLUDA-windows-rocm{rocm.version[0]}-amd64.zip', '_zluda')
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
    os.environ["ZLUDA_COMGR_LOG_LEVEL"] = "1"

    for v in HIPSDK_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(rocm.path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))

    def conceal():
        import torch # pylint: disable=unused-import
        platform = sys.platform
        sys.platform = ""
        from torch.utils import cpp_extension
        sys.platform = platform
        cpp_extension.IS_WINDOWS = platform == "win32"
        cpp_extension.IS_MACOS = False
        cpp_extension.IS_LINUX = platform.startswith('linux')
        def _join_rocm_home(*paths) -> str:
            return os.path.join(cpp_extension.ROCM_HOME, *paths)
        cpp_extension._join_rocm_home = _join_rocm_home # pylint: disable=protected-access
    rocm.conceal = conceal


def get_default_torch_version(agent: Optional[rocm.Agent]) -> str:
    if agent is not None:
        if agent.arch in (rocm.MicroArchitecture.RDNA, rocm.MicroArchitecture.CDNA,):
            return "2.3.1"
        elif agent.arch == rocm.MicroArchitecture.GCN:
            return "2.2.1"
    return "2.3.1"
