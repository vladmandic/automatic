# pylint: disable=unused-argument

import os
import re
import csv
from io import StringIO
from modules import shared, processing, sd_samplers, sd_models, sd_vae, sd_unet


re_range = re.compile(r'([-+]?[0-9]*\.?[0-9]+)-([-+]?[0-9]*\.?[0-9]+):?([0-9]+)?')


def apply_field(field):
    def fun(p, x, xs):
        shared.log.debug(f'XYZ grid apply field: {field}={x}')
        setattr(p, field, x)
    return fun


def apply_task_args(p, x, xs):
    for section in x.split(';'):
        k, v = section.split('=')
        k, v = k.strip(), v.strip()
        if v.replace('.','',1).isdigit():
            v = float(v) if '.' in v else int(v)
        p.task_args[k] = v
        shared.log.debug(f'XYZ grid apply task-arg: {k}={type(v)}:{v}')


def apply_processing(p, x, xs):
    for section in x.split(';'):
        k, v = section.split('=')
        k, v = k.strip(), v.strip()
        if v.replace('.','',1).isdigit():
            v = float(v) if '.' in v else int(v)
        found = 'existing' if hasattr(p, k) else 'new'
        setattr(p, k, v)
        shared.log.debug(f'XYZ grid apply processing-arg: type={found} {k}={type(v)}:{v} ')


def apply_setting(field):
    def fun(p, x, xs):
        shared.log.debug(f'XYZ grid apply setting: {field}={x}')
        shared.opts.data[field] = x
    return fun


def apply_seed(p, x, xs):
    p.seed = x
    p.all_seeds = None
    shared.log.debug(f'XYZ grid apply seed: {x}')


def apply_prompt(p, x, xs):
    if xs[0] not in p.prompt and xs[0] not in p.negative_prompt:
        shared.log.warning(f"XYZ grid: prompt S/R did not find {xs[0]} in prompt or negative prompt.")
    else:
        p.prompt = p.prompt.replace(xs[0], x)
        if p.all_prompts is not None:
            for i in range(len(p.all_prompts)):
                for j in range(len(xs)):
                    p.all_prompts[i] = p.all_prompts[i].replace(xs[j], x)
        p.negative_prompt = p.negative_prompt.replace(xs[0], x)
        if p.all_negative_prompts is not None:
            for i in range(len(p.all_negative_prompts)):
                for j in range(len(xs)):
                    p.all_negative_prompts[i] = p.all_negative_prompts[i].replace(xs[j], x)
        shared.log.debug(f'XYZ grid apply prompt: "{xs[0]}"="{x}"')


def apply_order(p, x, xs):
    token_order = []
    for token in x:
        token_order.append((p.prompt.find(token), token))
    token_order.sort(key=lambda t: t[0])
    prompt_parts = []
    for _, token in token_order:
        n = p.prompt.find(token)
        prompt_parts.append(p.prompt[0:n])
        p.prompt = p.prompt[n + len(token):]
    prompt_tmp = ""
    for idx, part in enumerate(prompt_parts):
        prompt_tmp += part
        prompt_tmp += x[idx]
    p.prompt = prompt_tmp + p.prompt


def apply_sampler(p, x, xs):
    sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if sampler_name is None:
        shared.log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.sampler_name = sampler_name
    shared.log.debug(f'XYZ grid apply sampler: "{x}"')


def apply_hr_sampler_name(p, x, xs):
    hr_sampler_name = sd_samplers.samplers_map.get(x.lower(), None)
    if hr_sampler_name is None:
        shared.log.warning(f"XYZ grid: unknown sampler: {x}")
    else:
        p.hr_sampler_name = hr_sampler_name
    shared.log.debug(f'XYZ grid apply HR sampler: "{x}"')


def confirm_samplers(p, xs):
    for x in xs:
        if x.lower() not in sd_samplers.samplers_map:
            shared.log.warning(f"XYZ grid: unknown sampler: {x}")


def apply_checkpoint(p, x, xs):
    if x == shared.opts.sd_model_checkpoint:
        return
    info = sd_models.get_closet_checkpoint_match(x)
    if info is None:
        shared.log.warning(f"XYZ grid: apply checkpoint unknown checkpoint: {x}")
    else:
        sd_models.reload_model_weights(shared.sd_model, info)
        p.override_settings['sd_model_checkpoint'] = info.name
    shared.log.debug(f'XYZ grid apply checkpoint: "{x}"')


def apply_refiner(p, x, xs):
    if x == shared.opts.sd_model_refiner:
        return
    if x == 'None':
        return
    info = sd_models.get_closet_checkpoint_match(x)
    if info is None:
        shared.log.warning(f"XYZ grid: apply refiner unknown checkpoint: {x}")
    else:
        sd_models.reload_model_weights(shared.sd_refiner, info)
        p.override_settings['sd_model_refiner'] = info.name
    shared.log.debug(f'XYZ grid apply refiner: "{x}"')


def apply_unet(p, x, xs):
    if x == shared.opts.sd_unet:
        return
    if x == 'None':
        return
    p.override_settings['sd_unet'] = x
    sd_unet.load_unet(shared.sd_model)
    shared.log.debug(f'XYZ grid apply unet: "{x}"')


def apply_dict(p, x, xs):
    if x == shared.opts.sd_model_dict:
        return
    info_dict = sd_models.get_closet_checkpoint_match(x)
    info_ckpt = sd_models.get_closet_checkpoint_match(shared.opts.sd_model_checkpoint)
    if info_dict is None or info_ckpt is None:
        shared.log.warning(f"XYZ grid: apply dict unknown checkpoint: {x}")
    else:
        shared.opts.sd_model_dict = info_dict.name # this will trigger reload_model_weights via onchange handler
        p.override_settings['sd_model_checkpoint'] = info_ckpt.name
        p.override_settings['sd_model_dict'] = info_dict.name
    shared.log.debug(f'XYZ grid apply model dict: "{x}"')


def apply_clip_skip(p, x, xs):
    p.clip_skip = x
    shared.opts.data["clip_skip"] = x
    shared.log.debug(f'XYZ grid apply clip-skip: "{x}"')


def find_vae(name: str):
    if name.lower() in ['auto', 'automatic']:
        return sd_vae.unspecified
    if name.lower() == 'none':
        return None
    else:
        choices = [x for x in sorted(sd_vae.vae_dict, key=lambda x: len(x)) if name.lower().strip() in x.lower()]
        if len(choices) == 0:
            shared.log.warning(f"No VAE found for {name}; using automatic")
            return sd_vae.unspecified
        else:
            return sd_vae.vae_dict[choices[0]]


def apply_vae(p, x, xs):
    sd_vae.reload_vae_weights(shared.sd_model, vae_file=find_vae(x))
    shared.log.debug(f'XYZ grid apply VAE: "{x}"')


def list_lora():
    import sys
    lora = [v for k, v in sys.modules.items() if k == 'networks'][0]
    loras = [v.fullname for v in lora.available_networks.values()]
    return ['None'] + loras


def apply_lora(p, x, xs):
    if x == 'None':
        return
    x = os.path.basename(x)
    p.prompt = p.prompt + f" <lora:{x}:{shared.opts.extra_networks_default_multiplier}>"
    shared.log.debug(f'XYZ grid apply LoRA: "{x}"')


def apply_te(p, x, xs):
    shared.opts.data["sd_text_encoder"] = x
    sd_models.reload_text_encoder()
    shared.log.debug(f'XYZ grid apply text-encoder: "{x}"')


def apply_styles(p: processing.StableDiffusionProcessingTxt2Img, x: str, _):
    p.styles.extend(x.split(','))
    shared.log.debug(f'XYZ grid apply style: "{x}"')


def apply_upscaler(p: processing.StableDiffusionProcessingTxt2Img, opt, x):
    p.enable_hr = True
    p.hr_force = True
    p.denoising_strength = 0.0
    p.hr_upscaler = opt
    shared.log.debug(f'XYZ grid apply upscaler: "{x}"')


def apply_context(p: processing.StableDiffusionProcessingTxt2Img, opt, x):
    p.resize_mode = 5
    p.resize_context = opt
    shared.log.debug(f'XYZ grid apply resize-context: "{x}"')


def apply_face_restore(p, opt, x):
    opt = opt.lower()
    if opt == 'codeformer':
        is_active = True
        p.face_restoration_model = 'CodeFormer'
    elif opt == 'gfpgan':
        is_active = True
        p.face_restoration_model = 'GFPGAN'
    else:
        is_active = opt in ('true', 'yes', 'y', '1')
    p.restore_faces = is_active
    shared.log.debug(f'XYZ grid apply face-restore: "{x}"')


def apply_override(field):
    def fun(p, x, xs):
        p.override_settings[field] = x
        shared.log.debug(f'XYZ grid apply override: "{field}"="{x}"')
    return fun


def format_value_add_label(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return f"{opt.label}: {x}"


def format_value(p, opt, x):
    if type(x) == float:
        x = round(x, 8)
    return x


def format_value_join_list(p, opt, x):
    return ", ".join(x)


def do_nothing(p, x, xs):
    pass


def format_nothing(p, opt, x):
    return ""


def str_permutations(x):
    """dummy function for specifying it in AxisOption's type when you want to get a list of permutations"""
    return x


def list_to_csv_string(data_list):
    with StringIO() as o:
        csv.writer(o).writerow(data_list)
        return o.getvalue().strip()
