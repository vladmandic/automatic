import os
from modules import shared, devices, files_cache


unet_dict = {}


def load_unet(model):
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file
    if shared.opts.sd_unet == 'None':
        return
    if shared.opts.sd_unet not in list(unet_dict):
        shared.log.error(f'UNet model not found: {shared.opts.sd_unet}')
        return
    if not hasattr(model, 'unet') or model.unet is None:
        shared.log.error('UNet not found in current model')
        return
    config_file = os.path.splitext(unet_dict[shared.opts.sd_unet])[0] + '.json'
    if os.path.exists(config_file):
        config = shared.readfile(config_file)
    else:
        config = None
        config_file = 'default'
    try:
        shared.log.info(f'Loading UNet: name="{shared.opts.sd_unet}" file="{unet_dict[shared.opts.sd_unet]}" config="{config_file}"')
        unet = UNet2DConditionModel.from_config(model.unet.config if config is None else config).to(devices.device, devices.dtype)
        state_dict = load_file(unet_dict[shared.opts.sd_unet])
        unet.load_state_dict(state_dict)
        model.unet = unet.to(devices.device, devices.dtype_unet)
    except Exception as e:
        unet = None
        shared.log.error(f'Failed to load UNet model: {e}')
        return


def refresh_unet_list():
    unet_dict.clear()
    for file in files_cache.list_files(shared.opts.unet_dir, ext_filter=[".safetensors"]):
        name = os.path.splitext(os.path.basename(file))[0]
        unet_dict[name] = file
    shared.log.debug(f'Available UNets: path="{shared.opts.unet_dir}" items={len(unet_dict)}')
