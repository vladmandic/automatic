from modules import shared


maybe_diffusers = [
    'aaebf6360f7d', # sd15-lcm
    '3d18b05e4f56', # sdxl-lcm
    'b71dcb732467', # sdxl-tcd
    '813ea5fb1c67', # sdxl-turbo
    # not really needed, but just in case
    '5a48ac366664', # hyper-sd15-1step
    'ee0ff23dcc42', # hyper-sd15-2step
    'e476eb1da5df', # hyper-sd15-4step
    'ecb844c3f3b0', # hyper-sd15-8step
    '1ab289133ebb', # hyper-sd15-8step-cfg
    '4f494295edb1', # hyper-sdxl-8step
    'ca14a8c621f8', # hyper-sdxl-8step-cfg
    '1c88f7295856', # hyper-sdxl-4step
    'fdd5dcd1d88a', # hyper-sdxl-2step
    '8cca3706050b', # hyper-sdxl-1step
]

force_diffusers = [
    '816d0eed49fd', # flash-sdxl
    'c2ec22757b46', # flash-sd15
]

def check_override(shorthash=''):
    force = False
    force = force or (shared.sd_model_type == 'sd3') # TODO sd3 forced diffusers for lora load
    if len(shorthash) < 4:
        return force
    force = force or (any(x.startswith(shorthash) for x in maybe_diffusers) if shared.opts.lora_maybe_diffusers else False)
    force = force or any(x.startswith(shorthash) for x in force_diffusers)
    if force and shared.opts.lora_maybe_diffusers:
        shared.log.debug('LoRA override: force diffusers')
    return force
