import time
import torch
import gradio as gr
import diffusers
from modules import scripts, processing, shared, images, devices, sd_models, sd_checkpoint


repo_id = 'a-r-r-o-w/LTX-Video-diffusers'
presets = [
    {"label": "custom", "width": 0, "height": 0, "num_frames": 0},
    {"label": "1216x704, 41 frames", "width": 1216, "height": 704, "num_frames": 41},
    {"label": "1088x704, 49 frames", "width": 1088, "height": 704, "num_frames": 49},
    {"label": "1056x640, 57 frames", "width": 1056, "height": 640, "num_frames": 57},
    {"label": "992x608, 65 frames", "width": 992, "height": 608, "num_frames": 65},
    {"label": "896x608, 73 frames", "width": 896, "height": 608, "num_frames": 73},
    {"label": "896x544, 81 frames", "width": 896, "height": 544, "num_frames": 81},
    {"label": "832x544, 89 frames", "width": 832, "height": 544, "num_frames": 89},
    {"label": "800x512, 97 frames", "width": 800, "height": 512, "num_frames": 97},
    {"label": "768x512, 97 frames", "width": 768, "height": 512, "num_frames": 97},
    {"label": "800x480, 105 frames", "width": 800, "height": 480, "num_frames": 105},
    {"label": "736x480, 113 frames", "width": 736, "height": 480, "num_frames": 113},
    {"label": "704x480, 121 frames", "width": 704, "height": 480, "num_frames": 121},
    {"label": "704x448, 129 frames", "width": 704, "height": 448, "num_frames": 129},
    {"label": "672x448, 137 frames", "width": 672, "height": 448, "num_frames": 137},
    {"label": "640x416, 153 frames", "width": 640, "height": 416, "num_frames": 153},
    {"label": "672x384, 161 frames", "width": 672, "height": 384, "num_frames": 161},
    {"label": "640x384, 169 frames", "width": 640, "height": 384, "num_frames": 169},
    {"label": "608x384, 177 frames", "width": 608, "height": 384, "num_frames": 177},
    {"label": "576x384, 185 frames", "width": 576, "height": 384, "num_frames": 185},
    {"label": "608x352, 193 frames", "width": 608, "height": 352, "num_frames": 193},
    {"label": "576x352, 201 frames", "width": 576, "height": 352, "num_frames": 201},
    {"label": "544x352, 209 frames", "width": 544, "height": 352, "num_frames": 209},
    {"label": "512x352, 225 frames", "width": 512, "height": 352, "num_frames": 225},
    {"label": "512x352, 233 frames", "width": 512, "height": 352, "num_frames": 233},
    {"label": "544x320, 241 frames", "width": 544, "height": 320, "num_frames": 241},
    {"label": "512x320, 249 frames", "width": 512, "height": 320, "num_frames": 249},
    {"label": "512x320, 257 frames", "width": 512, "height": 320, "num_frames": 257},
]


class Script(scripts.Script):
    def title(self):
        return 'Video: LTX Video'

    def show(self, is_img2img):
        return shared.native

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        def video_type_change(video_type):
            return [
                gr.update(visible=video_type != 'None'),
                gr.update(visible=video_type == 'GIF' or video_type == 'PNG'),
                gr.update(visible=video_type == 'MP4'),
                gr.update(visible=video_type == 'MP4'),
            ]
        def preset_change(preset):
            return gr.update(visible=preset == 'custom')

        with gr.Row():
            gr.HTML('<a href="https://www.ltxvideo.org/">&nbsp LTX Video</a><br>')
        with gr.Row():
            preset_name = gr.Dropdown(label='Preset', choices=[p['label'] for p in presets], value='custom')
            num_frames = gr.Slider(label='Frames', minimum=9, maximum=257, step=1, value=9)
        with gr.Row():
            video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2, visible=False)
        with gr.Row():
            gif_loop = gr.Checkbox(label='Loop', value=True, visible=False)
            mp4_pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
            mp4_interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
        preset_name.change(fn=preset_change, inputs=[preset_name], outputs=num_frames)
        video_type.change(fn=video_type_change, inputs=[video_type], outputs=[duration, gif_loop, mp4_pad, mp4_interpolate])
        return [preset_name, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def run(self, p: processing.StableDiffusionProcessing, preset_name, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        # set params
        preset = [p for p in presets if p['label'] == preset_name][0]
        image = getattr(p, 'init_images', None)
        image = None if image is None or len(image) == 0 else image[0]
        if p.width == 0 or p.height == 0 and image is not None:
            p.width = image.width
            p.height = image.height
        if preset['label'] != 'custom':
            num_frames = preset['num_frames']
            p.width = preset['width']
            p.height = preset['height']
        else:
            num_frames = 8 * int(num_frames // 8) + 1
            p.width = 32 * int(p.width // 32)
            p.height = 32 * int(p.height // 32)
        if image:
            image = images.resize_image(resize_mode=2, im=image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
            p.task_args['image'] = image
        p.task_args['output_type'] = 'pil'
        p.task_args['generator'] = torch.manual_seed(p.seed)
        p.task_args['num_frames'] = num_frames
        p.sampler_name = 'Default'
        p.do_not_save_grid = True
        p.ops.append('ltx')

        # load model
        cls = diffusers.LTXPipeline if image is None else diffusers.LTXImageToVideoPipeline
        diffusers.LTXTransformer3DModel = diffusers.LTXVideoTransformer3DModel
        diffusers.AutoencoderKLLTX = diffusers.AutoencoderKLLTXVideo
        if shared.sd_model.__class__ != cls:
            sd_models.unload_model_weights()
            shared.sd_model = cls.from_pretrained(
                repo_id,
                cache_dir = shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
            )
            sd_models.set_diffuser_options(shared.sd_model)
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(repo_id)
            shared.sd_model.sd_model_hash = None
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        shared.sd_model.vae.enable_slicing()
        shared.sd_model.vae.enable_tiling()
        devices.torch_gc(force=True)
        shared.log.debug(f'LTX: cls={shared.sd_model.__class__.__name__} preset={preset_name} args={p.task_args}')

        # run processing
        t0 = time.time()
        processed = processing.process_images(p)
        t1 = time.time()
        if processed is not None and len(processed.images) > 0:
            shared.log.info(f'LTX: frames={len(processed.images)} time={t1-t0:.2f}')
            if video_type != 'None':
                images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
