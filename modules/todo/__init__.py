from modules.todo.todo_utils import patch_attention_proc


def apply_todo(model, p, method='todo'):
    mp = p.height * p.width / 1024 / 1024

    if mp < 1.0: # 512px
        downsample_factor = 2
        ratio = 0.38
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    elif mp < 1.1: # 1024+
        downsample_factor = 2
        ratio = 0.75
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    elif mp < 2.3:
        downsample_factor = 3
        ratio = 0.89
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    elif mp < 8:
        downsample_factor = 4
        ratio = 0.9375
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    else:
        return
    merge_method = "downsample" if method == "todo" else "similarity"
    merge_tokens = "keys/values" if method == "todo" else "all"
    token_merge_args = {
                "ratio": ratio,
                "merge_tokens": merge_tokens,
                "merge_method": merge_method,
                "downsample_method": "nearest",
                "downsample_factor": downsample_factor,
                "timestep_threshold_switch": 0.0,
                "timestep_threshold_stop": 0.0,
                "downsample_factor_level_2": downsample_factor_level_2,
                "ratio_level_2": ratio_level_2
                }
    patch_attention_proc(model.unet, token_merge_args=token_merge_args)
