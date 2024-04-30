import os
import ctypes
import shutil
import zipfile
import platform
import urllib.request


RELEASE = f"rel.{os.environ.get('ZLUDA_HASH', '2804604c29b5fa36deca9ece219d3970b61d4c27')}"
DLL_MAPPING = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
HIP_TARGETS = ('rocblas.dll', 'rocsolver.dll', 'hiprtc0507.dll',)
ZLUDA_TARGETS = ('nvcuda.dll', 'nvml.dll',)


def find():
    return os.path.abspath(os.environ.get('ZLUDA', '.zluda'))


def check_dnn_dependency():
    hip_path = os.environ.get("HIP_PATH", None)
    if hip_path is None: # unable to check
        return True
    if os.path.exists(os.path.join(hip_path, 'bin', 'MIOpen.dll')):
        return True
    return False


def enable_dnn():
    global RELEASE # pylint: disable=global-statement
    DLL_MAPPING['cudnn.dll'] = 'cudnn64_8.dll'
    RELEASE = 'v3.8-pre2-dnn'


def install():
    zluda_path = find()

    if os.path.exists(zluda_path):
        return

    if platform.system() != 'Windows': # TODO
        return

    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/{RELEASE}/ZLUDA-windows-amd64.zip', '_zluda')
    with zipfile.ZipFile('_zluda', 'r') as archive:
        infos = archive.infolist()
        for info in infos:
            if not info.is_dir():
                info.filename = os.path.basename(info.filename)
                archive.extract(info, '.zluda')
    os.remove('_zluda')


def make_copy(zluda_path: os.PathLike):
    for k, v in DLL_MAPPING.items():
        if not os.path.exists(os.path.join(zluda_path, v)):
            try:
                os.link(os.path.join(zluda_path, k), os.path.join(zluda_path, v))
            except Exception:
                shutil.copyfile(os.path.join(zluda_path, k), os.path.join(zluda_path, v))


def load(zluda_path: os.PathLike):
    hip_path_default = r'C:\Program Files\AMD\ROCm\5.7'
    if not os.path.exists(hip_path_default):
        hip_path_default = None
    hip_path = os.environ.get('HIP_PATH', hip_path_default)
    if hip_path is None:
        raise RuntimeError('Could not find %HIP_PATH%. Please install AMD HIP SDK.')
    for v in HIP_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(hip_path, 'bin', v))
    for v in ZLUDA_TARGETS:
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
    for v in DLL_MAPPING.values():
        ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))
