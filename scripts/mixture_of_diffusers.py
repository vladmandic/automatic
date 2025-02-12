import gradio as gr
from modules import scripts, processing, shared, sd_models

max_xtiles = 4
max_ytiles = 4

class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None

    def title(self):
        return 'Mixture-of-Diffusers'

    def show(self, is_img2img):
        return shared.native

    def update_ui(self, x, y):
        updates = []
        for i in range(max_xtiles):
            for j in range(max_ytiles):
                updates.append(gr.update(visible=(i < x) and (j < y)))
        return updates

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://arxiv.org/abs/2302.02412">&nbsp Mixture-of-Diffusers</a><br>')
        with gr.Row():
            x_tiles = gr.Slider(minimum=1, maximum=max_xtiles, default=1, label="X-axis tiles")
            y_tiles = gr.Slider(minimum=1, maximum=max_ytiles, default=1, label="Y-axis tiles")
        with gr.Row():
            tile_width = gr.Number(minimum=1, maximum=2048, value=1024, label="Tile width")
            tile_height = gr.Number(minimum=1, maximum=2048, value=1024, label="Tile height")
        with gr.Row():
            overlap_width = gr.Number(minimum=1, maximum=512, value=128, label="Overlap width")
            overlap_height = gr.Number(minimum=1, maximum=512, value=128, label="Overlap height")
        with gr.Row():
            prompts = []
            for i in range(max_xtiles*max_ytiles):
                prompts.append(gr.Textbox('', label=f"Tile prompt: x={i%max_xtiles} y={i//max_ytiles}", placeholder='Prompt for tile', visible=False))
        x_tiles.change(fn=self.update_ui, inputs=[x_tiles, y_tiles], outputs=prompts)
        y_tiles.change(fn=self.update_ui, inputs=[x_tiles, y_tiles], outputs=prompts)
        return []

    def run(self, p: processing.StableDiffusionProcessing): # pylint: disable=arguments-differ, unused-argument
        supported_model_list = ['sdxl']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'MoD: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        self.orig_pipe = shared.sd_model
        from modules.mod import StableDiffusionXLTilingPipeline
        shared.sd_model = sd_models.switch_pipe(StableDiffusionXLTilingPipeline, shared.sd_model)
        sd_models.set_diffuser_options(shared.sd_model)
        sd_models.apply_balanced_offload(shared.sd_model)

        shared.log.info(f'MoD: ')

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed): # pylint: disable=arguments-differ, unused-argument
        if self.orig_pipe is None:
            return processed
        if shared.sd_model_type == "sdxl":
            shared.sd_model = self.orig_pipe
        self.orig_pipe = None
        return processed
