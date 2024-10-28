import inspect
import importlib
import gradio as gr
import diffusers
from modules import scripts, processing, shared, sd_models


class Script(scripts.Script):
    supported_models = ['sd', 'sdxl']
    orig_pipe = None
    try:
        library = importlib.import_module('k_diffusion')
    except Exception:
        library = None

    def title(self):
        return 'K-Diffusion'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/crowsonkb/k-diffusion">&nbsp K-Diffusion samplers</a><br>')
        with gr.Row():
            sampler = gr.Dropdown(label="Sampler", choices=self.samplers())
        return [sampler]

    def samplers(self):
        samplers = []
        sampling = getattr(self.library, 'sampling', None)
        if sampling is None:
            return samplers
        for s in dir(sampling):
            if s.startswith('sample_'):
                samplers.append(s.replace('sample_', ''))
        return samplers

    def callback(self, d):
        _step = d['i']

    def run(self, p: processing.StableDiffusionProcessing, sampler: str): # pylint: disable=arguments-differ
        if shared.sd_model_type not in self.supported_models:
            shared.log.warning(f'K-Diffusion: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={self.supported_models}')
            return None
        if self.library is None:
            return
        cls = None
        if shared.sd_model_type == "sd":
            cls = diffusers.pipelines.StableDiffusionKDiffusionPipeline
        if shared.sd_model_type == "sdxl":
            cls = diffusers.pipelines.StableDiffusionXLKDiffusionPipeline
        if cls is None:
            return None
        self.orig_pipe = shared.sd_model
        shared.sd_model = sd_models.switch_pipe(cls, shared.sd_model)
        sampler = 'sample_' + sampler

        sampling = getattr(self.library, "sampling", None)
        shared.sd_model.sampler = getattr(sampling, sampler)

        params = inspect.signature(shared.sd_model.sampler).parameters.values()
        params = {param.name: param.default for param in params if param.default != inspect.Parameter.empty}
        # if 'callback' in list(params):
        #     params['callback'] = self.callback
        # if 'disable' in list(params):
        #     params['disable'] = False
        shared.log.info(f'K-diffusion apply: class={shared.sd_model.__class__.__name__} sampler={sampler} params={params}')
        p.extra_generation_params["Sampler"] = sampler

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, sampler): # pylint: disable=arguments-differ, unused-argument
        if self.orig_pipe is None:
            return processed
        if shared.sd_model_type == "sdxl" or shared.sd_model_type == "sd":
            shared.sd_model = self.orig_pipe
        self.orig_pipe = None
        return processed
