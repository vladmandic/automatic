import os
import gradio as gr
import diffusers
from modules import scripts, processing, shared, devices, sd_models


class Script(scripts.Script):
    def title(self):
        return 'CogVideoX'

    def show(self, is_img2img):
        return shared.native


    def ui(self, _is_img2img):
        def video_type_change(video_type):
            return [
                gr.update(visible=video_type != 'None'),
                gr.update(visible=video_type == 'GIF' or video_type == 'PNG'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
            ]

        with gr.Row():
            gr.HTML("<span>&nbsp CogVideoX</span><br>")
        with gr.Row():
            model = gr.Dropdown(label='Model', choices=['THUDM/CogVideoX-2b', 'THUDM/CogVideoX-5b'], value='THUDM/CogVideoX-2b')
            sampler = gr.Dropdown(label='Sampler', choices=['DDIM', 'DPM'], value='DDIM')
        with gr.Row():
            frames = gr.Slider(label='Frames', minimum=1, maximum=64, step=1, value=16)
            guidance = gr.Slider(label='Guidance', minimum=0.0, maximum=14.0, step=0.5, value=6.0)
        with gr.Row():
            offload = gr.Dropdown(label='Offload', choices=['none', 'balanced', 'model', 'sequential'], value='balanced')
            override = gr.Checkbox(label='Override resolution', value=True)
        with gr.Row():
            video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2, visible=False)
        with gr.Row():
            loop = gr.Checkbox(label='Loop', value=True, visible=False)
            pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
            interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
        video_type.change(fn=video_type_change, inputs=[video_type], outputs=[duration, loop, pad, interpolate])
        return [model, sampler, frames, guidance, offload, override, video_type, duration, duration, loop, pad, interpolate]

    def run(self, p: processing.StableDiffusionProcessing, model, sampler, frames, guidance, offload, override, video_type, duration, loop, pad, interpolate): # pylint: disable=arguments-differ, unused-argument
        shared.log.debug(f'CogVideoX: model={model} sampler={sampler} frames={frames} guidance={guidance} offload={offload} override={override} video_type={video_type} duration={duration} loop={loop} pad={pad} interpolate={interpolate}')
        p.extra_generation_params['CogVideoX'] = model
        p.do_not_save_grid = True
        if 'animatediff' not in p.ops:
            p.ops.append('cogvideox')

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, model, sampler, frames, guidance, override_resolution, video_type, duration, loop, pad, interpolate): # pylint: disable=arguments-differ, unused-argument
        from modules.images import save_video
        if video_type != 'None':
            save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=loop, pad=pad, interpolate=interpolate)
