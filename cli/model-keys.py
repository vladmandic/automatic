#!/usr/bin/env python
import os
import sys
from rich import print as pprint


def has(obj, attr, *args):
    import functools
    if not isinstance(obj, dict):
        return False
    def _getattr(obj, attr):
        return obj.get(attr, args) if isinstance(obj, dict) else False
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def remove_entries_after_depth(d, depth, current_depth=0):
    if current_depth >= depth:
        return None
    if isinstance(d, dict):
        return {k: remove_entries_after_depth(v, depth, current_depth + 1) for k, v in d.items() if remove_entries_after_depth(v, depth, current_depth + 1) is not None}
    return d


def list_to_dict(flat_list):
    result_dict = {}
    for item in flat_list:
        keys = item.split('.')
        d = result_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = None
    return result_dict


def guess_dct(dct: dict):
    # if has(dct, 'model.diffusion_model.input_blocks') and has(dct, 'model.diffusion_model.label_emb'):
    #    return 'sdxl'
    if has(dct, 'model.diffusion_model.input_blocks') and len(list(has(dct, 'model.diffusion_model.input_blocks'))) == 12:
        return 'sd15'
    if has(dct, 'model.diffusion_model.input_blocks') and len(list(has(dct, 'model.diffusion_model.input_blocks'))) == 9:
        return 'sdxl'
    if has(dct, 'model.diffusion_model.joint_blocks') and len(list(has(dct, 'model.diffusion_model.joint_blocks'))) == 24:
        return 'sd35-medium'
    if has(dct, 'model.diffusion_model.joint_blocks') and len(list(has(dct, 'model.diffusion_model.joint_blocks'))) == 38:
        return 'sd35-large'
    if has(dct, 'model.diffusion_model.double_blocks') and len(list(has(dct, 'model.diffusion_model.double_blocks'))) == 19:
        return 'flux-dev'
    return None


def read_keys(fn):
    if not fn.lower().endswith(".safetensors"):
        return
    from safetensors.torch import safe_open
    keys = []
    try:
        with safe_open(fn, framework="pt", device="cpu") as f:
            keys = f.keys()
    except Exception as e:
        pprint(e)
    dct = list_to_dict(keys)
    pprint(f'file: {fn}')
    pprint(remove_entries_after_depth(dct, 3))
    pprint(remove_entries_after_depth(dct, 6))
    guess = guess_dct(dct)
    pprint(f'guess: {guess}')
    return keys


def main():
    if len(sys.argv) == 0:
        print('metadata:', 'no files specified')
    for fn in sys.argv:
        if os.path.isfile(fn):
            read_keys(fn)
        elif os.path.isdir(fn):
            for root, _dirs, files in os.walk(fn):
                for file in files:
                    read_keys(os.path.join(root, file))

if __name__ == '__main__':
    sys.argv.pop(0)
    main()
