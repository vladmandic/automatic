from modules import shared


supported_models = ['Flux', 'HunyuanVideo', 'CogVideoX', 'Mochi']


def apply_first_block_cache(p):
    if not shared.opts.para_cache_enabled or not shared.native:
        return
    if not any(p.sd_model.__class__.__name__.startswith(x) for x in supported_models):
        return
    from installer import install
    install('para_attn')
    try:
        from para_attn.first_block_cache import diffusers_adapters
        diffusers_adapters.apply_cache_on_pipe(p.sd_model, residual_diff_threshold=shared.opts.para_diff_threshold)
        shared.log.info(f'Applying para-attn first-block-cache: diff-threshold={shared.opts.para_diff_threshold} cls={p.sd_model.__class__.__name__}')
    except Exception as e:
        shared.log.error(f'Applying para-attn first-block-cache: {e}')
        return
