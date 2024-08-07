import os
import sys
import ctypes
import shutil
import subprocess
from typing import Union, List


HIPBLASLT_TENSILE_LIBPATH = os.environ.get("HIPBLASLT_TENSILE_LIBPATH", None if sys.platform == "win32" # not available
                                           else "/opt/rocm/lib/hipblaslt/library")


def resolve_link(path_: str) -> str:
    if not os.path.islink(path_):
        return path_
    return resolve_link(os.readlink(path_))


def dirname(path_: str, r: int = 1) -> str:
    for _ in range(0, r):
        path_ = os.path.dirname(path_)
    return path_


def spawn(command: str) -> str:
    process = subprocess.run(command, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout.decode(encoding="utf8", errors="ignore")


def load_library_global(path_: str):
    ctypes.CDLL(path_, mode=ctypes.RTLD_GLOBAL)


def conceal():
    os.environ.pop("ROCM_HOME", None)
    os.environ.pop("ROCM_PATH", None)
    paths = os.environ["PATH"].split(";")
    paths_no_rocm = []
    for path_ in paths:
        if "rocm" not in path_.lower():
            paths_no_rocm.append(path_)
    os.environ["PATH"] = ";".join(paths_no_rocm)


class Agent:
    name: str
    is_navi4x: bool = False
    is_navi3a: bool = False # 3.5
    is_navi3x: bool = False
    is_navi2x: bool = False
    is_navi1x: bool = False
    is_gcn: bool = False
    if sys.platform != "win32":
        blaslt_supported: bool

    def __init__(self, name: str):
        self.name = name
        gfx_version = name[3:6]
        if gfx_version == "120":
            self.is_navi4x = True
        elif gfx_version == "115":
            self.is_navi3a = True
        elif gfx_version == "110":
            self.is_navi3x = True
        elif gfx_version == "103":
            self.is_navi2x = True
        elif gfx_version == "101":
            self.is_navi1x = True
        else:
            self.is_gcn = True
        if sys.platform != "win32":
            self.blaslt_supported = os.path.exists(os.path.join(HIPBLASLT_TENSILE_LIBPATH, f"extop_{name}.co"))

    def get_gfx_version(self) -> Union[str, None]:
        if self.is_navi3x:
            return "11.0.0"
        elif self.is_navi2x:
            return "10.3.0"
        #elif self.is_navi1x:
        #    return "10.3.0" # maybe?
        return None


if sys.platform == "win32":
    def find() -> Union[str, None]:
        hip_path = shutil.which("hipconfig")
        if hip_path is not None:
            return dirname(resolve_link(hip_path), 2)

        hip_path = os.environ.get("HIP_PATH", None)
        if hip_path is not None:
            return hip_path

        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        hip_path = rf'{program_files}\AMD\ROCm'
        if not os.path.exists(hip_path):
            return None

        class Version:
            major: int
            minor: int

            def __init__(self, string: str):
                self.major, self.minor = [int(v) for v in string.strip().split(".")]

            def __gt__(self, other):
                return self.major * 10 + other.minor > other.major * 10 + other.minor

            def __str__(self):
                return f"{self.major}.{self.minor}"

        latest = None
        versions = os.listdir(hip_path)
        for s in versions:
            item = None
            try:
                item = Version(s)
            except Exception:
                continue
            if latest is None:
                latest = item
                continue
            if item > latest:
                latest = item

        if latest is None:
            return None

        return os.path.join(hip_path, str(latest))

    def get_version() -> str: # cannot just run hipconfig as it requires Perl installed on Windows.
        return os.path.basename(path) or os.path.basename(os.path.dirname(path))

    def get_agents() -> List[Agent]:
        return [Agent(x.split(' ')[-1].strip()) for x in spawn("hipinfo").split("\n") if x.startswith('gcnArchName:')]

    is_wsl: bool = False
else:
    def find() -> Union[str, None]:
        rocm_path = shutil.which("hipconfig")
        if rocm_path is not None:
            return dirname(resolve_link(rocm_path), 2)
        if not os.path.exists("/opt/rocm"):
            return None
        return resolve_link("/opt/rocm")

    def get_version() -> str:
        arr = spawn(f"{os.path.join(path, 'bin', 'hipconfig')} --version").split(".")
        return f'{arr[0]}.{arr[1]}' if len(arr) >= 2 else None

    def get_agents() -> List[Agent]:
        if is_wsl: # WSL does not have 'rocm_agent_enumerator'
            agents = spawn("rocminfo").split("\n")
            agents = [x.strip().split(" ")[-1] for x in agents if x.startswith('  Name:') and "CPU" not in x]
        else:
            agents = spawn("rocm_agent_enumerator").split("\n")
            agents = [x for x in agents if x and x != 'gfx000']
        return [Agent(x) for x in agents]

    def load_hsa_runtime() -> None:
        try:
            # Preload stdc++ library. This will ignore Anaconda stdc++ library.
            load_library_global("/lib/x86_64-linux-gnu/libstdc++.so.6")
        except OSError:
            pass
        # Preload HSA Runtime library.
        load_library_global("/opt/rocm/lib/libhsa-runtime64.so")

    def set_blaslt_enabled(enabled: bool) -> None:
        if enabled:
            load_library_global("/opt/rocm/lib/libhipblaslt.so") # Preload hipBLASLt.
            os.environ["HIPBLASLT_TENSILE_LIBPATH"] = HIPBLASLT_TENSILE_LIBPATH
        else:
            os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"

    is_wsl: bool = os.environ.get('WSL_DISTRO_NAME', None) is not None
path = find()
is_installed = False
version = None
if path is not None:
    is_installed = True
    version = get_version()
