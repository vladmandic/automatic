from modules import shared


def apply_token_merging(sd_model):
    current_tome = getattr(sd_model, 'applied_tome', 0)
    current_todo = getattr(sd_model, 'applied_todo', 0)

    if shared.opts.token_merging_method == 'ToMe' and shared.opts.tome_ratio > 0:
        if current_tome == shared.opts.tome_ratio:
            return
        if shared.opts.hypertile_unet_enabled and not shared.cmd_opts.experimental:
            shared.log.warning('Token merging not supported with HyperTile for UNet')
            return
        try:
            import installer
            installer.install('tomesd', 'tomesd', ignore=False)
            import tomesd
            tomesd.apply_patch(
                sd_model,
                ratio=shared.opts.tome_ratio,
                use_rand=False, # can cause issues with some samplers
                merge_attn=True,
                merge_crossattn=False,
                merge_mlp=False
            )
            shared.log.info(f'Applying ToMe: ratio={shared.opts.tome_ratio}')
            sd_model.applied_tome = shared.opts.tome_ratio
        except Exception:
            shared.log.warning(f'Token merging not supported: pipeline={sd_model.__class__.__name__}')
    else:
        sd_model.applied_tome = 0

    if shared.opts.token_merging_method == 'ToDo' and shared.opts.todo_ratio > 0:
        if current_todo == shared.opts.todo_ratio:
            return
        if shared.opts.hypertile_unet_enabled and not shared.cmd_opts.experimental:
            shared.log.warning('Token merging not supported with HyperTile for UNet')
            return
        try:
            from modules.todo.todo_utils import patch_attention_proc
            token_merge_args = {
                        "ratio": shared.opts.todo_ratio,
                        "merge_tokens": "keys/values",
                        "merge_method": "downsample",
                        "downsample_method": "nearest",
                        "downsample_factor": 2,
                        "timestep_threshold_switch": 0.0,
                        "timestep_threshold_stop": 0.0,
                        "downsample_factor_level_2": 1,
                        "ratio_level_2": 0.0,
                        }
            patch_attention_proc(sd_model.unet, token_merge_args=token_merge_args)
            shared.log.info(f'Applying ToDo: ratio={shared.opts.todo_ratio}')
            sd_model.applied_todo = shared.opts.todo_ratio
        except Exception:
            shared.log.warning(f'Token merging not supported: pipeline={sd_model.__class__.__name__}')
    else:
        sd_model.applied_todo = 0


def remove_token_merging(sd_model):
    current_tome = getattr(sd_model, 'applied_tome', 0)
    current_todo = getattr(sd_model, 'applied_todo', 0)
    try:
        if current_tome > 0:
            import tomesd
            tomesd.remove_patch(sd_model)
            sd_model.applied_tome = 0
    except Exception:
        pass
    try:
        if current_todo > 0:
            from modules.todo.todo_utils import remove_patch
            remove_patch(sd_model)
            sd_model.applied_todo = 0
    except Exception:
        pass
