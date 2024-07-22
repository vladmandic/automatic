import os
import sys
import shutil
import subprocess
from typing import Union, List


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
        return os.path.basename(path)

    def get_agents() -> List[str]:
        return [x.split(' ')[-1].strip() for x in spawn("hipinfo").split("\n") if x.startswith('gcnArchName:')]

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
        arr = spawn(f"{os.path.join(path, 'hipconfig')} --version").split(".")
        return f'{arr[0]}.{arr[1]}' if len(arr) >= 2 else None

    def get_agents() -> List[str]:
        if is_wsl: # WSL does not have 'rocm_agent_enumerator'
            agents = spawn("rocminfo").split("\n")
            return [x.strip().split(" ")[-1] for x in agents if x.startswith('  Name:') and "CPU" not in x]
        else:
            agents = spawn("rocm_agent_enumerator").split("\n")
            return [x for x in agents if x and x != 'gfx000']

    is_wsl: bool = os.environ.get('WSL_DISTRO_NAME', None) is not None
path = find()
is_installed = path is not None
version = get_version()
