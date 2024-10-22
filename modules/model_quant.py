import sys
from installer import install, log


bnb = None
quanto = None


def load_bnb(msg='', silent=False):
    global bnb # pylint: disable=global-statement
    if bnb is not None:
        return bnb
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    log.debug(f'Quantization: type=bitsandbytes fn={fn}') # pylint: disable=protected-access
    install('bitsandbytes', quiet=True)
    try:
        import bitsandbytes
        bnb = bitsandbytes
        return bnb
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import bitsandbytes: {e}")
        bnb = None
        if not silent:
            raise


def load_quanto(msg='', silent=False):
    global quanto # pylint: disable=global-statement
    if quanto is not None:
        return quanto
    fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
    log.debug(f'Quantization: type=quanto fn={fn}') # pylint: disable=protected-access
    install('optimum-quanto', quiet=True)
    try:
        from optimum import quanto as optimum_quanto # pylint: disable=no-name-in-module
        quanto = optimum_quanto
        return quanto
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import optimum.quanto: {e}")
        quanto = None
        if not silent:
            raise


def get_quant(name):
    if "qint8" in name.lower():
        return 'qint8'
    if "qint4" in name.lower():
        return 'qint4'
    if "fp8" in name.lower():
        return 'fp8'
    if "fp4" in name.lower():
        return 'fp4'
    if "nf4" in name.lower():
        return 'nf4'
    if name.endswith('.gguf'):
        return 'gguf'
    return 'none'
