import os
from installer import setup_logging
setup_logging()


checked_ok = False


def check_dependencies():
    from installer import installed, pip, log
    global checked_ok # pylint: disable=global-statement
    debug = log.trace if os.environ.get('SD_DWPOSE_DEBUG', None) is not None else lambda *args, **kwargs: None
    packages = [
        'openmim==0.3.9',
        'mmengine==0.10.4',
        'mmcv==2.1.0',
        'mmpose==1.3.1',
        'mmdet==3.3.0',
    ]
    status = [installed(p, reload=False, quiet=False) for p in packages]
    status.append(False)
    debug(f'DWPose required={packages} status={status}')
    if not all(status):
        log.info(f'Installing DWPose dependencies: {[packages]}')
        cmd = 'install --upgrade --no-deps --force-reinstall '
        pkgs = ' '.join(packages)
        res = pip(cmd + pkgs, ignore=False, quiet=False)
        debug(f'DWPose pip install: {res}')
    try:
        import pkg_resources
        import imp # pylint: disable=deprecated-module
        imp.reload(pkg_resources)
        import mmcv # pylint: disable=unused-import
        import mmengine # pylint: disable=unused-import
        import mmpose # pylint: disable=unused-import
        import mmdet # pylint: disable=unused-import
        debug('DWPose import ok')
        checked_ok = True
    except Exception as e:
        log.error(f'DWPose: {e}')
    return checked_ok


check_dependencies()
