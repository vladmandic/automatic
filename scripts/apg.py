import gradio as gr
from modules import scripts, processing, shared, sd_models


class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.register()

    def title(self):
        return 'APG'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://arxiv.org/abs/2410.02416">&nbsp APG: Adaptive projected guidance</a><br>')
        with gr.Row():
            eta = gr.Slider(label="ETA", value=1.0, minimum=0, maximum=2.0, step=0.01)
            momentum = gr.Slider(label="Momentum", value=0.0, minimum=-1.0, maximum=1.0, step=0.01)
            threshold = gr.Slider(label="Threshold", value=0.0, minimum=0.0, maximum=5.0, step=0.01)
        return [eta, momentum, threshold]

    def register(self): # register xyz grid elements
        def apply_field(field):
            def fun(p, x, xs): # pylint: disable=unused-argument
                setattr(p, field, x)
                self.run(p)
            return fun

        import sys
        xyz_classes = [v for k, v in sys.modules.items() if 'xyz_grid_classes' in k][0]
        xyz_classes.axis_options.append(xyz_classes.AxisOption("[APG] ETA", float, apply_field("apg_eta")))
        xyz_classes.axis_options.append(xyz_classes.AxisOption("[APG] Momentum", float, apply_field("apg_momentum")))
        xyz_classes.axis_options.append(xyz_classes.AxisOption("[APG] Threshold", float, apply_field("apg_threshold")))

    def run(self, p: processing.StableDiffusionProcessing, eta = 0.0, momentum = 0.0, threshold = 0.0): # pylint: disable=arguments-differ
        if shared.sd_model_type != 'sdxl':
            shared.log.warning(f'APG: pipeline={shared.sd_model_type} required=sdxl')
            return None
        from modules import apg
        apg.eta = getattr(p, 'apg_eta', eta) # use values set by xyz grid or via ui
        apg.momentum = getattr(p, 'apg_momentum', momentum)
        apg.threshold = getattr(p, 'apg_threshold', threshold)
        apg.buffer = apg.MomentumBuffer(apg.momentum) # recreate buffer
        self.orig_pipe = shared.sd_model
        shared.sd_model = sd_models.switch_pipe(apg.StableDiffusionXLPipelineAPG, shared.sd_model) # sdxl pipeline with call to apg.normalized_guidance instead of default
        shared.log.info(f'APG apply: guidance={p.cfg_scale} momentum={apg.momentum} eta={apg.eta} threshold={apg.threshold} class={shared.sd_model.__class__.__name__}')
        p.extra_generation_params["APG"] = f'ETA={apg.eta} Momentum={apg.momentum} Threshold={apg.threshold}'
        # processed = processing.process_images(p)

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, eta, momentum, threshold): # pylint: disable=arguments-differ, unused-argument
        from modules import apg
        shared.sd_model = self.orig_pipe # restore pipeline
        apg.buffer = None
        return processed
