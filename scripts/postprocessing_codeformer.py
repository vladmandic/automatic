from PIL import Image
import numpy as np
import gradio as gr
from modules import scripts_postprocessing
from modules.postprocess import codeformer_model


class ScriptPostprocessingCodeFormer(scripts_postprocessing.ScriptPostprocessing):
    name = "CodeFormer"
    order = 3000

    def ui(self):
        with gr.Accordion('Restore faces: CodeFormer', open = False):
            with gr.Row():
                codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Strength", value=0.0, elem_id="extras_codeformer_visibility")
                codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Weight", value=0.2, elem_id="extras_codeformer_weight")
        return { "codeformer_visibility": codeformer_visibility, "codeformer_weight": codeformer_weight }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, codeformer_visibility, codeformer_weight): # pylint: disable=arguments-differ
        if codeformer_visibility == 0:
            return
        restored_img = codeformer_model.codeformer.restore(np.array(pp.image, dtype=np.uint8), w=codeformer_weight)
        res = Image.fromarray(restored_img)
        if codeformer_visibility < 1.0:
            res = Image.blend(pp.image, res, codeformer_visibility)
        pp.image = res
        pp.info["CodeFormer visibility"] = round(codeformer_visibility, 3)
        pp.info["CodeFormer weight"] = round(codeformer_weight, 3)
