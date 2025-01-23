import sys
import diffusers
from installer import install, log


bnb = None
quanto = None
ao = None


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




def create_bnb_config(kwargs = None, allow_bnb: bool = True):
    from modules import shared, devices
    if len(shared.opts.bnb_quantization) > 0 and allow_bnb:
        if 'Model' in shared.opts.bnb_quantization:
            load_bnb()
            if bnb is None:
                return kwargs
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


def create_ao_config(kwargs = None, allow_ao: bool = True):
    from modules import shared
    if len(shared.opts.torchao_quantization) > 0 and shared.opts.torchao_quantization_mode == 'pre' and allow_ao:
        if 'Model' in shared.opts.torchao_quantization:
            load_torchao()
            if ao is None:
                return kwargs
            diffusers.utils.import_utils.is_torchao_available = lambda: True
            ao_config = diffusers.TorchAoConfig(shared.opts.torchao_quantization_type)
            shared.log.debug(f'Quantization: module=all type=torchao dtype={shared.opts.torchao_quantization_type}')
            if kwargs is None:
                return ao_config
            else:
                kwargs['quantization_config'] = ao_config
                return kwargs
    return kwargs


def load_torchao(msg='', silent=False):
    global ao # pylint: disable=global-statement
    if ao is not None:
        return ao
    install('torchao==0.7.0', quiet=True)
    try:
        import torchao
        ao = torchao
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=torchao version={ao.__version__} fn={fn}') # pylint: disable=protected-access
        return ao
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import torchao: {e}")
        ao = None
        if not silent:
            raise
    return None


def load_bnb(msg='', silent=False):
    from modules import devices
    global bnb # pylint: disable=global-statement
    if bnb is not None:
        return bnb
    if devices.backend == 'cuda':
        # forcing a version will uninstall the multi-backend-refactor branch of bnb
        install('bitsandbytes==0.45.0', quiet=True)
    try:
        import bitsandbytes
        bnb = bitsandbytes
        diffusers.utils.import_utils._bitsandbytes_available = True # pylint: disable=protected-access
        diffusers.utils.import_utils._bitsandbytes_version = '0.43.3' # pylint: disable=protected-access
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=bitsandbytes version={bnb.__version__} fn={fn}') # pylint: disable=protected-access
        return bnb
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import bitsandbytes: {e}")
        bnb = None
        if not silent:
            raise
    return None


def load_quanto(msg='', silent=False):
    from modules import shared
    global quanto # pylint: disable=global-statement
    if quanto is not None:
        return quanto
    install('optimum-quanto==0.2.6', quiet=True)
    try:
        from optimum import quanto as optimum_quanto # pylint: disable=no-name-in-module
        quanto = optimum_quanto
        fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
        log.debug(f'Quantization: type=quanto version={quanto.__version__} fn={fn}') # pylint: disable=protected-access
        if shared.opts.diffusers_offload_mode in {'balanced', 'sequential'}:
            shared.log.error(f'Quantization: type=quanto offload={shared.opts.diffusers_offload_mode} not supported')
        return quanto
    except Exception as e:
        if len(msg) > 0:
            log.error(f"{msg} failed to import optimum.quanto: {e}")
        quanto = None
        if not silent:
            raise
    return None


def apply_layerwise(sd_model, quiet:bool=False):
    import torch
    from diffusers.quantizers import quantization_config
    from modules import shared, devices, sd_models
    if shared.opts.layerwise_quantization_storage == 'float8_e4m3fn' and hasattr(torch, 'float8_e4m3fn'):
        storage_dtype = torch.float8_e4m3fn
    elif shared.opts.layerwise_quantization_storage == 'float8_e5m2' and hasattr(torch, 'float8_e5m2'):
        storage_dtype = torch.float8_e5m2
    else:
        storage_dtype = None
        shared.log.warning(f'Quantization: type=layerwise storage={shared.opts.layerwise_quantization_storage} not supported')
        return
    non_blocking = False
    if not hasattr(quantization_config.QuantizationMethod, 'LAYERWISE'):
        setattr(quantization_config.QuantizationMethod, 'LAYERWISE', 'layerwise') # noqa: B010
    for module in sd_models.get_signature(sd_model).keys():
        if not hasattr(sd_model, module):
            continue
        try:
            cls = getattr(sd_model, module).__class__.__name__
            if module.startswith('unet') and ('Model' in shared.opts.layerwise_quantization):
                m = getattr(sd_model, module)
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=non_blocking)
                    m.quantization_method = 'LayerWise'
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not non_blocking}')
            if module.startswith('transformer') and ('Model' in shared.opts.layerwise_quantization or 'Transformer' in shared.opts.layerwise_quantization):
                m = getattr(sd_model, module)
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=non_blocking)
                    m.quantization_method = 'LayerWise'
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not non_blocking}')
            if module.startswith('text_encoder') and ('Model' in shared.opts.layerwise_quantization or 'Text Encoder' in shared.opts.layerwise_quantization) and ('clip' not in cls.lower()):
                m = getattr(sd_model, module)
                if hasattr(m, 'enable_layerwise_casting'):
                    m.enable_layerwise_casting(compute_dtype=devices.dtype, storage_dtype=storage_dtype, non_blocking=non_blocking)
                    m.quantization_method = quantization_config.QuantizationMethod.LAYERWISE # pylint: disable=no-member
                    log.quiet(quiet, f'Quantization: type=layerwise module={module} cls={cls} storage={storage_dtype} compute={devices.dtype} blocking={not non_blocking}')
        except Exception as e:
            shared.log.error(f'Quantization: type=layerwise {e}')
