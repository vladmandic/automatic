import os
import time
import torch
import gradio as gr
import diffusers
import transformers
from modules import scripts, processing, shared, images, devices, sd_models, sd_checkpoint, model_quant


repos = {
    '0.9.0': 'a-r-r-o-w/LTX-Video-diffusers',
    '0.9.1': 'a-r-r-o-w/LTX-Video-0.9.1-diffusers',
    'custom': None,
}


def load_quants(kwargs, repo_id):
    if len(shared.opts.bnb_quantization) > 0:
        quant_args = {}
        quant_args = model_quant.create_bnb_config(quant_args)
        quant_args = model_quant.create_ao_config(quant_args)
        if not quant_args:
            return kwargs
        model_quant.load_bnb(f'Load model: type=LTX quant={quant_args}')
        if 'Model' in shared.opts.bnb_quantization and 'transformer' not in kwargs:
            kwargs['transformer'] = diffusers.LTXVideoTransformer3DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype, **quant_args)
            shared.log.debug(f'Quantization: module=transformer type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
        if 'Text Encoder' in shared.opts.bnb_quantization and 'text_encoder_3' not in kwargs:
            kwargs['text_encoder'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder", cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype, **quant_args)
            shared.log.debug(f'Quantization: module=t5 type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
    return kwargs


def hijack_decode(*args, **kwargs):
    shared.log.debug('Video: decode')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return shared.sd_model.vae.orig_decode(*args, **kwargs)


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
        def model_change(model):
            return gr.update(visible=model == 'custom')

        with gr.Row():
            gr.HTML('<a href="https://www.ltxvideo.org/">&nbsp LTX Video</a><br>')
        with gr.Row():
            model = gr.Dropdown(label='LTX Model', choices=list(repos), value='0.9.1')
            decode = gr.Dropdown(label='Decode', choices=['diffusers', 'native'], value='diffusers', visible=False)
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=9, maximum=257, step=1, value=41)
            sampler = gr.Checkbox(label='Override sampler', value=True)
        with gr.Row():
            model_custom = gr.Textbox(value='', label='Path to model file', visible=False)
        with gr.Row():
            video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=10, step=0.25, value=2, visible=False)
        with gr.Row():
            gif_loop = gr.Checkbox(label='Loop', value=True, visible=False)
            mp4_pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
            mp4_interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
        video_type.change(fn=video_type_change, inputs=[video_type], outputs=[duration, gif_loop, mp4_pad, mp4_interpolate])
        model.change(fn=model_change, inputs=[model], outputs=[model_custom])
        return [model, model_custom, decode, sampler, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def run(self, p: processing.StableDiffusionProcessing, model, model_custom, decode, sampler, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        # set params
        image = getattr(p, 'init_images', None)
        image = None if image is None or len(image) == 0 else image[0]
        if (p.width == 0 or p.height == 0) and image is not None:
            p.width = image.width
            p.height = image.height
        num_frames = 8 * int(num_frames // 8) + 1
        p.width = 32 * int(p.width // 32)
        p.height = 32 * int(p.height // 32)
        processing.fix_seed(p)
        if image:
            image = images.resize_image(resize_mode=2, im=image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
            p.task_args['image'] = image
        p.task_args['output_type'] = 'latent' if decode == 'native' else 'pil'
        p.task_args['generator'] = torch.Generator(devices.device).manual_seed(p.seed)
        p.task_args['num_frames'] = num_frames
        p.do_not_save_grid = True
        if sampler:
            p.sampler_name = 'Default'
        p.ops.append('video')

        # load model
        cls = diffusers.LTXPipeline if image is None else diffusers.LTXImageToVideoPipeline
        diffusers.LTXTransformer3DModel = diffusers.LTXVideoTransformer3DModel
        diffusers.AutoencoderKLLTX = diffusers.AutoencoderKLLTXVideo
        repo_id = repos[model]
        if repo_id is None:
            repo_id = model_custom
        if shared.sd_model.__class__ != cls:
            sd_models.unload_model_weights()
            kwargs = {}
            kwargs = model_quant.create_bnb_config(kwargs)
            kwargs = model_quant.create_ao_config(kwargs)
            if os.path.isfile(repo_id):
                shared.sd_model = cls.from_single_file(
                    repo_id,
                    cache_dir = shared.opts.hfcache_dir,
                    torch_dtype=devices.dtype,
                    **kwargs
                )
            else:
                kwargs = load_quants(kwargs, repo_id)
                shared.sd_model = cls.from_pretrained(
                    repo_id,
                    cache_dir = shared.opts.hfcache_dir,
                    torch_dtype=devices.dtype,
                    **kwargs
                )
            sd_models.set_diffuser_options(shared.sd_model)
            shared.sd_model.vae.orig_decode = shared.sd_model.vae.decode
            shared.sd_model.vae.decode = hijack_decode
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(repo_id)
            shared.sd_model.sd_model_hash = None
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        shared.sd_model.vae.enable_slicing()
        shared.sd_model.vae.enable_tiling()
        devices.torch_gc(force=True)
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} args={p.task_args}')

        # run processing
        t0 = time.time()
        processed = processing.process_images(p)
        t1 = time.time()
        if processed is not None and len(processed.images) > 0:
            shared.log.info(f'Video: frames={len(processed.images)} time={t1-t0:.2f}')
            if video_type != 'None':
                images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
