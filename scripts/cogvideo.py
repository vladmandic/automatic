"""
models: https://huggingface.co/THUDM/CogVideoX-2b https://huggingface.co/THUDM/CogVideoX-5b
source: https://github.com/THUDM/CogVideo
quanto: https://gist.github.com/a-r-r-o-w/31be62828b00a9292821b85c1017effa
torchao: https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897
venhancer: https://github.com/THUDM/CogVideo/blob/dcb82ae30b454ab898aeced0633172d75dbd55b8/tools/venhancer/README.md
"""
import os
import time
import cv2
import gradio as gr
import torch
from torchvision import transforms
import diffusers
import numpy as np
from modules import scripts, shared, devices, errors, sd_models, processing
from modules.processing_callbacks import diffusers_callback, set_callbacks_p


debug = (os.environ.get('SD_LOAD_DEBUG', None) is not None) or (os.environ.get('SD_PROCESS_DEBUG', None) is not None)


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
            model = gr.Dropdown(label='Model', choices=['None', 'THUDM/CogVideoX-2b', 'THUDM/CogVideoX-5b', 'THUDM/CogVideoX-5b-I2V'], value='THUDM/CogVideoX-2b')
            sampler = gr.Dropdown(label='Sampler', choices=['DDIM', 'DPM'], value='DDIM')
        with gr.Row():
            frames = gr.Slider(label='Frames', minimum=1, maximum=100, step=1, value=49)
            guidance = gr.Slider(label='Guidance', minimum=0.0, maximum=14.0, step=0.5, value=6.0)
        with gr.Row():
            offload = gr.Dropdown(label='Offload', choices=['none', 'balanced', 'model', 'sequential'], value='balanced')
            override = gr.Checkbox(label='Override resolution', value=True)
        with gr.Row():
            video_type = gr.Dropdown(label='Video file', choices=['None', 'GIF', 'PNG', 'MP4'], value='None')
            duration = gr.Slider(label='Duration', minimum=0.25, maximum=30, step=0.25, value=8, visible=False)
        with gr.Accordion('Optional init video', open=False):
            with gr.Row():
                image = gr.Image(value=None, label='Image', type='pil', source='upload', width=256, height=256)
                video = gr.Video(value=None, label='Video', source='upload', width=256, height=256)
        with gr.Row():
            loop = gr.Checkbox(label='Loop', value=True, visible=False)
            pad = gr.Slider(label='Pad frames', minimum=0, maximum=24, step=1, value=1, visible=False)
            interpolate = gr.Slider(label='Interpolate frames', minimum=0, maximum=24, step=1, value=0, visible=False)
        video_type.change(fn=video_type_change, inputs=[video_type], outputs=[duration, loop, pad, interpolate])
        return [model, sampler, frames, guidance, offload, override, video_type, duration, loop, pad, interpolate, image, video]

    def load(self, model):
        if (shared.sd_model_type != 'cogvideox' or shared.sd_model.sd_model_checkpoint != model) and model != 'None':
            sd_models.unload_model_weights('model')
            shared.log.info(f'CogVideoX load: model="{model}"')
            try:
                shared.sd_model = None
                cls = diffusers.CogVideoXImageToVideoPipeline if 'I2V' in model else diffusers.CogVideoXPipeline
                shared.sd_model = cls.from_pretrained(model, torch_dtype=devices.dtype, cache_dir=shared.opts.diffusers_dir)
                shared.sd_model.sd_checkpoint_info = sd_models.CheckpointInfo(model)
                shared.sd_model.sd_model_hash = ''
                shared.sd_model.sd_model_checkpoint = model
            except Exception as e:
                shared.log.error(f'Loading CogVideoX: {e}')
                if debug:
                    errors.display(e, 'CogVideoX')
        if shared.sd_model_type == 'cogvideox' and model != 'None':
            shared.sd_model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m', ncols=80, colour='#327fba')
            shared.log.debug(f'CogVideoX load: class="{shared.sd_model.__class__.__name__}"')
        if shared.sd_model is not None and model == 'None':
            shared.log.info(f'CogVideoX unload: model={model}')
            shared.sd_model = None
            devices.torch_gc(force=True)
        devices.torch_gc()

    def offload(self, offload):
        if shared.sd_model_type != 'cogvideox':
            return
        if offload == 'none':
            sd_models.move_model(shared.sd_model, devices.device)
        shared.log.debug(f'CogVideoX: offload={offload}')
        if offload == 'balanced':
            sd_models.apply_balanced_offload(shared.sd_model)
        if offload == 'model':
            shared.sd_model.enable_model_cpu_offload()
        if offload == 'sequential':
            shared.sd_model.enable_model_cpu_offload()
            shared.sd_model.enable_sequential_cpu_offload()
        shared.sd_model.vae.enable_slicing()
        shared.sd_model.vae.enable_tiling()

    def video(self, p, fn):
        frames = []
        try:
            from modules.control.util import decode_fourcc
            video = cv2.VideoCapture(fn)
            if not video.isOpened():
                shared.log.error(f'Video: file="{fn}" open failed')
                return frames
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            codec = decode_fourcc(video.get(cv2.CAP_PROP_FOURCC))
            shared.log.debug(f'CogVideoX input: video="{fn}" fps={fps} width={w} height={h} codec={codec} frames={frame_count} target={len(frames)}')
            frames = []
            while True:
                ok, frame = video.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (p.width, p.height))
                frames.append(frame)
            video.release()
            if len(frames) > p.frames:
                frames = np.asarray(frames)
                indices = np.linspace(0, len(frames) - 1, p.frames).astype(int) # reduce array from n_frames to p_frames
                frames = frames[indices]
                shared.log.debug(f'CogVideoX input reduce: source={len(frames)} target={p.frames}')
            frames = [transforms.ToTensor()(frame) for frame in frames]
        except Exception as e:
            shared.log.error(f'Video: file="{fn}" {e}')
            if debug:
                errors.display(e, 'CogVideoX')
        return frames

    def image(self, p, img):
        img = img.resize((p.width, p.height))
        shared.log.debug(f'CogVideoX input: image={img}')
        # frames = [np.array(img)]
        # frames = [transforms.ToTensor()(frame) for frame in frames]
        return img

    def generate(self, p: processing.StableDiffusionProcessing, model: str):
        if shared.sd_model_type != 'cogvideox':
            return []
        shared.log.info(f'CogVideoX: sampler={p.sampler} steps={p.steps} frames={p.frames} width={p.width} height={p.height} seed={p.seed} guidance={p.guidance}')
        if p.sampler == 'DDIM':
            shared.sd_model.scheduler = diffusers.CogVideoXDDIMScheduler.from_config(shared.sd_model.scheduler.config, timestep_spacing="trailing")
        if p.sampler == 'DPM':
            shared.sd_model.scheduler = diffusers.CogVideoXDPMScheduler.from_config(shared.sd_model.scheduler.config, timestep_spacing="trailing")
        t0 = time.time()
        frames = []
        set_callbacks_p(p)
        shared.state.job_count = 1
        shared.state.sampling_steps = p.steps - 1
        try:
            args = dict(
                prompt=p.prompt,
                negative_prompt=p.negative_prompt,
                height=p.height,
                width=p.width,
                num_videos_per_prompt=1,
                num_inference_steps=p.steps,
                guidance_scale=p.guidance,
                generator=torch.Generator(device=devices.device).manual_seed(p.seed),
                callback_on_step_end=diffusers_callback,
                callback_on_step_end_tensor_inputs=['latents'],
            )
            if getattr(p, 'image', False):
                if 'I2V' not in model:
                    shared.log.error(f'CogVideoX: model={model} image input not supported')
                    return []
                args['image'] = self.image(p, p.image)
                args['num_frames'] = p.frames # only txt2vid has num_frames
                shared.sd_model = sd_models.switch_pipe(diffusers.CogVideoXImageToVideoPipeline, shared.sd_model)
            elif getattr(p, 'video', False):
                if 'I2V' in model:
                    shared.log.error(f'CogVideoX: model={model} image input not supported')
                    return []
                args['video'] = self.video(p, p.video)
                shared.sd_model = sd_models.switch_pipe(diffusers.CogVideoXVideoToVideoPipeline, shared.sd_model)
            else:
                if 'I2V' in model:
                    shared.log.error(f'CogVideoX: model={model} image input not supported')
                    return []
                args['num_frames'] = p.frames # only txt2vid has num_frames
                shared.sd_model = sd_models.switch_pipe(diffusers.CogVideoXPipeline, shared.sd_model)
            if debug:
                shared.log.debug(f'CogVideoX args: {args}')
            frames = shared.sd_model(**args).frames[0]
        except AssertionError as e:
            shared.log.info(f'CogVideoX: {e}')
        except Exception as e:
            shared.log.error(f'CogVideoX: {e}')
            if debug:
                errors.display(e, 'CogVideoX')
        t1 = time.time()
        its = (len(frames) * p.steps) / (t1 - t0)
        shared.log.info(f'CogVideoX: frames={len(frames)} its={its:.2f} time={t1 - t0:.2f}')
        return frames

    # auto-executed by the script-callback
    def run(self, p: processing.StableDiffusionProcessing, model, sampler, frames, guidance, offload, override, video_type, duration, loop, pad, interpolate, image, video): # pylint: disable=arguments-differ, unused-argument
        shared.state.begin('CogVideoX')
        processing.fix_seed(p)
        p.extra_generation_params['CogVideoX'] = model
        p.do_not_save_grid = True
        if 'animatediff' not in p.ops:
            p.ops.append('cogvideox')
        if override:
            p.width = 720
            p.height = 480
        p.sampler = sampler
        p.guidance = guidance
        p.frames = frames
        p.use_dynamic_cfg = sampler == 'DPM'
        p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        p.image = image
        p.video = video
        self.load(model)
        self.offload(offload)
        frames = self.generate(p, model)
        devices.torch_gc()
        processed = processing.Processed(p, images_list=frames)
        shared.state.end()
        return processed

    # auto-executed by the script-callback
    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, model, sampler, frames, guidance, offload, override, video_type, duration, loop, pad, interpolate, image, video): # pylint: disable=arguments-differ, unused-argument
        if video_type != 'None' and processed is not None and len(processed.images) > 0:
            from modules.images import save_video
            shared.log.info(f'CogVideoX video: type={video_type} duration={duration} loop={loop} pad={pad} interpolate={interpolate}')
            save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=loop, pad=pad, interpolate=interpolate)
