from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionXLPipeline # pylint: disable=unused-import
from modules import shared, processing, sd_models
from modules.pag.pipe_sd import StableDiffusionPAGPipeline
from modules.pag.pipe_sdxl import StableDiffusionXLPAGPipeline
from modules.control.units import detect


orig_pipeline = None


def apply(p: processing.StableDiffusionProcessing): # pylint: disable=arguments-differ
    global orig_pipeline # pylint: disable=global-statement
    c = shared.sd_model.__class__ if shared.sd_loaded else None
    if not shared.native:
        return None
    if p.pag_scale == 0:
        unapply()
        return None
    if c == StableDiffusionPAGPipeline or c == StableDiffusionXLPAGPipeline:
        pass
    elif detect.is_sd15(c):
        orig_pipeline = shared.sd_model
        shared.sd_model = sd_models.switch_pipe(StableDiffusionPAGPipeline, shared.sd_model)
    elif detect.is_sdxl(c):
        orig_pipeline = shared.sd_model
        shared.sd_model = sd_models.switch_pipe(StableDiffusionXLPAGPipeline, shared.sd_model)
    else:
        shared.log.warning(f'PAG: pipeline={c} required={StableDiffusionPipeline.__name__}')
        return None

    p.task_args['pag_scale'] = p.pag_scale
    p.task_args['pag_adaptive_scaling'] = p.pag_adaptive
    pag_applied_layers = shared.opts.pag_apply_layers
    pag_applied_layers_index = pag_applied_layers.split() if len(pag_applied_layers) > 0 else []
    pag_applied_layers_index = [p.strip() for p in pag_applied_layers_index]
    p.task_args['pag_applied_layers_index'] = pag_applied_layers_index if len(pag_applied_layers_index) > 0 else ['m0'] # Available layers: d[0-5], m[0], u[0-8]
    p.extra_generation_params["PAG scale"] = p.pag_scale
    p.extra_generation_params["PAG adaptive"] = p.pag_adaptive
    shared.log.debug(f'{c}: args={p.task_args}')


def unapply():
    global orig_pipeline # pylint: disable=global-statement
    if orig_pipeline is not None:
        shared.sd_model = orig_pipeline
        orig_pipeline = None
