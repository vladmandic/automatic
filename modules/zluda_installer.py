import os
import shutil
import zipfile
import tarfile
import platform
import urllib.request


RELEASE = 'rel.2804604c29b5fa36deca9ece219d3970b61d4c27'
TARGETS = {
    'cublas.dll': 'cublas64_11.dll',
    'cusparse.dll': 'cusparse64_11.dll',
    'nvrtc.dll': 'nvrtc64_112_0.dll',
}
ZLUDA_PATH = None
TORCHLIB_PATH = None


def find_zluda_path():
    zluda_path = os.environ.get('ZLUDA', None)
    if zluda_path is None:
        paths = os.environ.get('PATH', '').split(';')
        for path in paths:
            if os.path.exists(os.path.join(path, 'zluda_redirect.dll')):
                zluda_path = path
                break
    return zluda_path


def find_venv_dir():
    python_dir = os.path.dirname(shutil.which('python'))
    if shutil.which('conda') is None:
        python_dir = os.path.dirname(python_dir)
    return os.environ.get('VENV_DIR', python_dir)


def reset_torch():
    for dll in TARGETS.values():
        path = os.path.join(TORCHLIB_PATH, dll)
        if os.path.exists(path):
            os.remove(path)


def is_patched():
    for dll in TARGETS.values():
        if not os.path.islink(os.path.join(TORCHLIB_PATH, dll)):
            return False
    return True


def check_dnn_dependency():
    hip_path = os.environ.get("HIP_PATH", None)
    if hip_path is None: # unable to check
        return True
    if os.path.exists(os.path.join(hip_path, 'bin', 'MIOpen.dll')):
        return True
    return False


def enable_dnn():
    global RELEASE # pylint: disable=global-statement
    TARGETS['cudnn.dll'] = 'cudnn64_8.dll'
    RELEASE = 'v3.8-pre2-dnn'


def install():
    global ZLUDA_PATH, TORCHLIB_PATH # pylint: disable=global-statement
    ZLUDA_PATH = find_zluda_path()
    TORCHLIB_PATH = os.path.join(find_venv_dir(), 'Lib', 'site-packages', 'torch', 'lib')

    if ZLUDA_PATH is not None:
        return

    is_windows = platform.system() == 'Windows'
    archive_type = zipfile.ZipFile if is_windows else tarfile.TarFile
    urllib.request.urlretrieve(f'https://github.com/lshqqytiger/ZLUDA/releases/download/{RELEASE}/ZLUDA-{platform.system().lower()}-amd64.{"zip" if is_windows else "tar.gz"}', '_zluda')
    with archive_type('_zluda', 'r') as f:
        f.extractall('.zluda')
    ZLUDA_PATH = os.path.abspath('./.zluda')
    os.remove('_zluda')


def resolve_path():
    paths = os.environ.get('PATH', '.')
    if ZLUDA_PATH not in paths:
        os.environ['PATH'] = paths + ';' + ZLUDA_PATH


def patch():
    if is_patched():
        return
    reset_torch()
    for k, v in TARGETS.items():
        os.symlink(os.path.join(ZLUDA_PATH, k), os.path.join(TORCHLIB_PATH, v))
