import time
import torch
import gradio as gr
import transformers
import diffusers
from modules import scripts, processing, shared, images, devices, sd_models, sd_checkpoint, sd_samplers, model_quant, timer


default_template = """Describe the video by detailing the following aspects:
1. The main content and theme of the video.
2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.
3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.
4. Background environment, light, style and atmosphere.
5. Camera angles, movements, and transitions used in the video.
6. Thematic and aesthetic concepts associated with the scene, i.e. realistic, futuristic, fairy tale, etc.
"""

def get_template(template: str = None):
    # diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video.DEFAULT_PROMPT_TEMPLATE
    base_template_pre = "<|start_header_id|>system<|end_header_id|>\n\n"
    base_template_post = "<|eot_id|>\n"
    base_template_end = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    if template is None or len(template) == 0:
        template = default_template
    template_lines = '\n'.join([line for line in template.split('\n') if len(line) > 0])
    prompt_template = {
        "crop_start": 95,
        "template": base_template_pre + template_lines + base_template_post + base_template_end
    }
    return prompt_template


def hijack_decode(*args, **kwargs):
    t0 = time.time()
    vae: diffusers.AutoencoderKLHunyuanVideo = shared.sd_model.vae
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    res = shared.sd_model.vae.orig_decode(*args, **kwargs)
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.log.debug(f'Video: vae={vae.__class__.__name__} tile={vae.tile_sample_min_width}:{vae.tile_sample_min_height}:{vae.tile_sample_min_num_frames} stride={vae.tile_sample_stride_width}:{vae.tile_sample_stride_height}:{vae.tile_sample_stride_num_frames} time={t1-t0:.2f}')
    return res


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    res = shared.sd_model.vae.orig_encode_prompt(*args, **kwargs)
    t1 = time.time()
    timer.process.add('te', t1-t0)
    shared.log.debug(f'Video: te={shared.sd_model.text_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


class Script(scripts.Script):
    def title(self):
        return 'Video: Hunyuan Video'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    # return signature is array of gradio components
    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://huggingface.co/tencent/HunyuanVideo">&nbsp Hunyuan Video</a><br>')
        with gr.Row():
            repo_id = gr.Dropdown(label='Model', choices=['tencent/HunyuanVideo', 'FastVideo/FastHunyuan'], value='tencent/HunyuanVideo')
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=9, maximum=257, step=1, value=45)
            tile_frames = gr.Slider(label='Tile frames', minimum=1, maximum=64, step=1, value=16)
        with gr.Row():
            with gr.Column():
                override_scheduler = gr.Checkbox(label='Override sampler', value=True)
            with gr.Column():
                scheduler_shift = gr.Slider(label='Sampler shift', minimum=0.0, maximum=20.0, step=0.1, value=7.0)
        with gr.Row():
            template = gr.TextArea(label='Prompt processor', lines=3, value=default_template, visible=False)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [repo_id, num_frames, tile_frames, override_scheduler, scheduler_shift, template, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def load(self, repo_id:str):
        if shared.sd_model.__class__ != diffusers.HunyuanVideoPipeline:
            sd_models.unload_model_weights()
            t0 = time.time()
            quant_args = {}
            quant_args = model_quant.create_bnb_config(quant_args)
            if quant_args:
                model_quant.load_bnb(f'Load model: type=HunyuanVideo quant={quant_args}')
            if not quant_args:
                quant_args = model_quant.create_ao_config(quant_args)
                if quant_args:
                    model_quant.load_torchao(f'Load model: type=HunyuanVideo quant={quant_args}')
            transformer = diffusers.HunyuanVideoTransformer3DModel.from_pretrained(
                repo_id,
                subfolder="transformer",
                torch_dtype=devices.dtype,
                revision="refs/pr/18",
                cache_dir = shared.opts.hfcache_dir,
                **quant_args
            )
            shared.log.debug(f'Video: module={transformer.__class__.__name__}')
            text_encoder = transformers.LlamaModel.from_pretrained(
                repo_id,
                subfolder="text_encoder",
                revision="refs/pr/18",
                cache_dir = shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **quant_args
            )
            shared.log.debug(f'Video: module={text_encoder.__class__.__name__}')
            shared.sd_model = diffusers.HunyuanVideoPipeline.from_pretrained(
                repo_id,
                transformer=transformer,
                text_encoder=text_encoder,
                revision="refs/pr/18",
                cache_dir = shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **quant_args
            )
            t1 = time.time()
            shared.log.debug(f'Video: load cls={shared.sd_model.__class__.__name__} repo="{repo_id}" dtype={devices.dtype} time={t1-t0:.2f}')
            sd_models.set_diffuser_options(shared.sd_model)
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(repo_id)
            shared.sd_model.sd_model_hash = None
            shared.sd_model.vae.orig_decode = shared.sd_model.vae.decode
            shared.sd_model.vae.orig_encode_prompt = shared.sd_model.encode_prompt
            shared.sd_model.vae.decode = hijack_decode
            shared.sd_model.encode_prompt = hijack_encode_prompt
            shared.sd_model.vae.enable_slicing()
            shared.sd_model.vae.enable_tiling()

    def run(self, p: processing.StableDiffusionProcessing, repo_id, num_frames, tile_frames, override_scheduler, scheduler_shift, template, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        # set params
        num_frames = int(num_frames)
        p.width = 16 * int(p.width // 16)
        p.height = 16 * int(p.height // 16)
        p.do_not_save_grid = True
        p.ops.append('video')

        # load model
        self.load(repo_id)

        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        devices.torch_gc(force=True)

        if override_scheduler:
            p.sampler_name = 'Default'
        else:
            shared.sd_model.scheduler = sd_samplers.create_sampler(p.sampler_name, shared.sd_model)
            p.sampler_name = 'Default' # avoid double creation
        if hasattr(shared.sd_model.scheduler, '_shift'):
            shared.sd_model.scheduler._shift = scheduler_shift # pylint: disable=protected-access

        # encode prompt
        processing.fix_seed(p)
        p.task_args['num_frames'] = num_frames
        p.task_args['output_type'] = 'pil'
        p.task_args['generator'] = torch.manual_seed(p.seed)
        # p.task_args['prompt'] = None
        # p.task_args['prompt_embeds'], p.task_args['pooled_prompt_embeds'], p.task_args['prompt_attention_mask'] = shared.sd_model.encode_prompt(prompt=p.prompt, prompt_template=get_template(template), device=devices.device)

        # run processing
        t0 = time.time()
        shared.sd_model.vae.tile_sample_min_num_frames = tile_frames
        shared.state.disable_preview = True
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} width={p.width} height={p.height} frames={num_frames}')
        processed = processing.process_images(p)
        shared.state.disable_preview = False
        t1 = time.time()
        if processed is not None and len(processed.images) > 0:
            shared.log.info(f'Video: frames={len(processed.images)} time={t1-t0:.2f}')
            if video_type != 'None':
                images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
