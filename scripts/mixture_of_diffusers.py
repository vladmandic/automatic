import gradio as gr
from modules import scripts, processing, shared


class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None

    def title(self):
        return 'Mixture-of-Diffusers'

    def show(self, is_img2img):
        return shared.native

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://arxiv.org/abs/2302.02412">&nbsp Mixture-of-Diffusers</a><br>')
        return []

    def run(self, p: processing.StableDiffusionProcessing): # pylint: disable=arguments-differ, unused-argument
        supported_model_list = ['sdxl']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'MoD: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        self.orig_pipe = shared.sd_model

        shared.log.info(f'MoD: ')


    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed): # pylint: disable=arguments-differ, unused-argument
        if self.orig_pipe is None:
            return processed
        if shared.sd_model_type == "sdxl":
            shared.sd_model = self.orig_pipe
        self.orig_pipe = None
        return processed
