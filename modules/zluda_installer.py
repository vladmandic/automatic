import os
import sys
import site
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
HIPSDK_TARGETS = ['rocblas.dll', 'rocsolver.dll']
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll',)

path = os.path.abspath(os.environ.get('ZLUDA', '.zluda'))
default_agent: Union[rocm.Agent, None] = None
hipBLASLt_enabled = os.path.exists(os.path.join(rocm.path, "bin", "hipblaslt.dll")) and os.path.exists(rocm.blaslt_tensile_libpath) and os.path.exists(os.path.join(path, 'cublasLt.dll'))


def set_default_agent(agent: rocm.Agent):
    global default_agent # pylint: disable=global-statement
    default_agent = agent


def install() -> None:
    if os.path.exists(path):
        return

    platform = "windows"
    commit = os.environ.get("ZLUDA_HASH", "d60bddbc870827566b3d2d417e00e1d2d8acc026")
    if os.environ.get("ZLUDA_NIGHTLY", "0") == "1":
        platform = "nightly-" + platform
    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/rel.{commit}/ZLUDA-{platform}-rocm{rocm.version[0]}-amd64.zip', '_zluda')
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


def set_blaslt_enabled(enabled: bool):
    global hipBLASLt_enabled # pylint: disable=global-statement
    hipBLASLt_enabled = enabled


def get_blaslt_enabled() -> bool:
    return hipBLASLt_enabled


def link_or_copy(src: os.PathLike, dst: os.PathLike):
    try:
        os.link(src, dst)
    except Exception:
        shutil.copyfile(src, dst)


def make_copy() -> None:
    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(path, v)):
            link_or_copy(os.path.join(path, k), os.path.join(path, v))

    if hipBLASLt_enabled and not os.path.exists(os.path.join(path, 'cublasLt64_11.dll')):
        link_or_copy(os.path.join(path, 'cublasLt.dll'), os.path.join(path, 'cublasLt64_11.dll'))


def load() -> None:
    os.environ["ZLUDA_COMGR_LOG_LEVEL"] = "1"
    os.environ["ZLUDA_NVRTC_LIB"] = os.path.join([v for v in site.getsitepackages() if v.endswith("site-packages")][0], "torch", "lib", "nvrtc64_112_0.dll")

    for v in HIPSDK_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(rocm.path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(path, v))

    if hipBLASLt_enabled:
        ctypes.windll.LoadLibrary(os.path.join(rocm.path, 'bin', 'hipblaslt.dll'))
        ctypes.windll.LoadLibrary(os.path.join(path, 'cublasLt64_11.dll'))

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
            return "2.4.1" if hipBLASLt_enabled else "2.3.1"
        elif agent.arch == rocm.MicroArchitecture.GCN:
            return "2.2.1"
    return "2.4.1" if hipBLASLt_enabled else "2.3.1"
