import sys
import diffusers
from installer import install, log


bnb = None
quanto = None


def create_bnb_config(kwargs = None):
    from modules import shared, devices
    if len(shared.opts.bnb_quantization) > 0:
        if 'Model' in shared.opts.bnb_quantization and 'transformer' not in (kwargs or {}):
            load_bnb()
            bnb_config = diffusers.BitsAndBytesConfig(
                load_in_8bit=shared.opts.bnb_quantization_type in ['fp8'],
                load_in_4bit=shared.opts.bnb_quantization_type in ['nf4', 'fp4'],
                bnb_4bit_quant_storage=shared.opts.bnb_quantization_storage,
                bnb_4bit_quant_type=shared.opts.bnb_quantization_type,
                bnb_4bit_compute_dtype=devices.dtype
            )
            shared.log.debug(f'Quantization: module=all type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
            if kwargs is None:
                return bnb_config
            else:
                kwargs['quantization_config'] = bnb_config
                return kwargs
    return kwargs


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
        diffusers.utils.import_utils._bitsandbytes_available = True # pylint: disable=protected-access
        diffusers.utils.import_utils._bitsandbytes_version = '0.43.3' # pylint: disable=protected-access
        return bnb
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import bitsandbytes: {e}")
        bnb = None
        if not silent:
            raise
    return None


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
    return None


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
