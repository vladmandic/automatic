import os
import re
import json


debug = lambda *args, **kwargs: None # pylint: disable=unnecessary-lambda-assignment
re_size = re.compile(r"^(\d+)x(\d+)$") # int x int
re_param = re.compile(r'\s*([\w ]+):\s*("(?:\\"[^,]|\\"|\\|[^\"])+"|[^,]*)(?:,|$)') # multi-word: value


def quote(text):
    if ',' not in str(text) and '\n' not in str(text) and ':' not in str(text):
        return text
    return json.dumps(text, ensure_ascii=False)


def unquote(text):
    if len(text) == 0 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def parse(infotext):
    if not isinstance(infotext, str):
        return {}
    debug(f'Raw: {infotext}')
    if 'negative prompt:' not in infotext.lower():
        infotext = 'negative prompt: ' + infotext
    if 'prompt:' not in infotext.lower():
        infotext = 'prompt: ' + infotext

    remaining = infotext.replace('\nSteps:', ' Steps:')
    prompt = remaining[:infotext.lower().find('negative prompt:')]
    remaining = remaining.replace(prompt, '')
    if prompt.lower().startswith('prompt: '):
        prompt = prompt[8:]
    # debug(f'Prompt: {prompt}')

    params = ['steps:', 'seed:', 'width:', 'height:', 'sampler:', 'size:', 'cfg scale:'] # first param is one of those
    param_idx = [remaining.lower().find(p) for p in params if p in remaining.lower()]
    param_idx = min(param_idx) if len(param_idx) > 0 else 0
    negative = remaining[:param_idx] if param_idx > 0 else remaining
    remaining = remaining.replace(negative, '')
    if negative.lower().startswith('negative prompt: '):
        negative = negative[16:]
    # debug(f'Negative: {negative}')

    params = dict(re_param.findall(remaining))
    params['Prompt'] = prompt
    params['Negative prompt'] = negative
    for key, val in params.copy().items():
        val = unquote(val).strip(" ,\n").replace('\\\n', '')
        size = re_size.match(val)
        if val.replace('.', '', 1).isdigit():
            params[key] = float(val) if '.' in val else int(val)
        elif val == "True":
            params[key] = True
        elif val == "False":
            params[key] = False
        elif key == 'VAE' and val == 'TAESD':
            params["Full quality"] = False
        elif size is not None:
            params[f"{key}-1"] = int(size.group(1))
            params[f"{key}-2"] = int(size.group(2))
        elif isinstance(params[key], str):
            params[key] = val
        debug(f'Param parsed: type={type(params[key])} {key}={params[key]} raw="{val}"')

    return params


mapping = [
    ('Backend', 'sd_backend'),
    ('Model hash', 'sd_model_checkpoint'),
    ('Refiner', 'sd_model_refiner'),
    ('VAE', 'sd_vae'),
    ('Parser', 'prompt_attention'),
    ('Color correction', 'img2img_color_correction'),
    # Samplers
    ('Sampler Eta', 'scheduler_eta'),
    ('Sampler ENSD', 'eta_noise_seed_delta'),
    ('Sampler order', 'schedulers_solver_order'),
    # Samplers diffusers
    ('Sampler beta schedule', 'schedulers_beta_schedule'),
    ('Sampler beta start', 'schedulers_beta_start'),
    ('Sampler beta end', 'schedulers_beta_end'),
    ('Sampler DPM solver', 'schedulers_dpm_solver'),
    # Samplers original
    ('Sampler brownian', 'schedulers_brownian_noise'),
    ('Sampler discard', 'schedulers_discard_penultimate'),
    ('Sampler dyn threshold', 'schedulers_use_thresholding'),
    ('Sampler karras', 'schedulers_use_karras'),
    ('Sampler low order', 'schedulers_use_loworder'),
    ('Sampler quantization', 'enable_quantization'),
    ('Sampler sigma', 'schedulers_sigma'),
    ('Sampler sigma min', 's_min'),
    ('Sampler sigma max', 's_max'),
    ('Sampler sigma churn', 's_churn'),
    ('Sampler sigma uncond', 's_min_uncond'),
    ('Sampler sigma noise', 's_noise'),
    ('Sampler sigma tmin', 's_tmin'),
    ('Sampler ENSM', 'initial_noise_multiplier'), # img2img only
    ('UniPC skip type', 'uni_pc_skip_type'),
    ('UniPC variant', 'uni_pc_variant'),
    # Token Merging
    ('Mask weight', 'inpainting_mask_weight'),
    ('ToMe', 'tome_ratio'),
    ('ToDo', 'todo_ratio'),
]


if __name__ == '__main__':
    import logging
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s | %(message)s')
    debug = log.info

    import sys
    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            with open(sys.argv[1], 'r', encoding='utf8') as f:
                parse(f.read())
        else:
            parse(sys.argv[1])
