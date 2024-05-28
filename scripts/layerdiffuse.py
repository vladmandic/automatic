import gradio as gr
from modules import scripts, shared


class Script(scripts.Script):

    def title(self):
        return 'LayerDiffuse'

    def show(self, is_img2img):
        return True if shared.backend == shared.Backend.DIFFUSERS else False

    def apply(self):
        from modules import layerdiffuse
        if not shared.sd_loaded:
            shared.log.error('LayerDiffuse: model not loaded')
            return
        if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
            shared.log.error(f'LayerDiffuse: incorrect base model: class={shared.sd_model.__class__.__name__} type={shared.sd_model_type}')
            return
        if hasattr(shared.sd_model, 'layerdiffusion'):
            shared.log.warning('LayerDiffuse: already applied')
            return
        layerdiffuse.apply_layerdiffuse()

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML("""
                    <a href="https://github.com/rootonchair/diffuser_layerdiffuse">&nbsp LayerDiffuse</a><br><br>
                    <div>Click once to permanently apply to current model</div>
                    <div>Reload model to unapply</div><br>
                    """)
        with gr.Row():
            apply_btn = gr.Button('Apply to model', variant='primary')
            apply_btn.click(fn=self.apply)
        return []
