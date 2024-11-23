# https://github.com/huggingface/diffusers/pull/9985

import time
import gradio as gr
import diffusers
from modules import scripts, processing, shared, devices, sd_models
from installer import install


redux_pipe: diffusers.FluxPriorReduxPipeline = None
processor_canny = None
processor_depth = None
title = 'Flux Tools'


class Script(scripts.Script):
    def title(self):
        return f'{title}'

    def show(self, is_img2img):
        return is_img2img if shared.native else False

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://blackforestlabs.ai/flux-1-tools/">&nbsp Flux.1 Redux</a><br>')
        with gr.Row():
            tool = gr.Dropdown(label='Tool', choices=['None', 'Redux', 'Fill', 'Canny', 'Depth'], value='None')
            strength = gr.Checkbox(label='Override denoise strength', value=True)
        return [tool, strength]

    def run(self, p: processing.StableDiffusionProcessing, tool: str = 'None', strength: bool = True): # pylint: disable=arguments-differ
        global redux_pipe, processor_canny, processor_depth # pylint: disable=global-statement
        if tool is None or tool == 'None':
            return
        supported_model_list = ['f1']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'{title}: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        image = getattr(p, 'init_images', None)
        if image is None or len(image) == 0:
            shared.log.error(f'{title}: tool={tool} no init_images')
            return None
        else:
            image = image[0] if isinstance(image, list) else image

        shared.log.info(f'{title}: tool={tool} init')

        t0 = time.time()
        if tool == 'Redux':
            # pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained("black-forest-labs/FLUX.1-Redux-dev", revision="refs/pr/8", torch_dtype=torch.bfloat16).to("cuda")
            if redux_pipe is None:
                redux_pipe = diffusers.FluxPriorReduxPipeline.from_pretrained(
                    "black-forest-labs/FLUX.1-Redux-dev",
                    revision="refs/pr/8",
                    torch_dtype=devices.dtype,
                    cache_dir=shared.opts.hfcache_dir
                ).to(devices.device)
            redux_output = redux_pipe(image)
            for k, v in redux_output.items():
                p.task_args[k] = v
        else:
            if redux_pipe is not None:
                shared.log.debug(f'{title}: tool=Redux unload')
                redux_pipe = None

        if tool == 'Fill':
            # pipe = FluxFillPipeline.from_pretrained("black-forest-labs/FLUX.1-Fill-dev", torch_dtype=torch.bfloat16, revision="refs/pr/4").to("cuda")
            if p.image_mask is None:
                shared.log.error(f'{title}: tool={tool} no image_mask')
                return None
            if shared.sd_model.__class__.__name__ != 'FluxFillPipeline':
                shared.opts.data["sd_model_checkpoint"] = "black-forest-labs/FLUX.1-Fill-dev"
                sd_models.reload_model_weights(op='model', revision="refs/pr/4")
            p.task_args['image'] = image
            p.task_args['mask_image'] = p.image_mask

        if tool == 'Canny':
            # pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16, revision="refs/pr/1").to("cuda")
            install('controlnet-aux')
            install('timm==0.9.16')
            if shared.sd_model.__class__.__name__ != 'FluxControlPipeline' or 'Canny' not in shared.opts.sd_model_checkpoint:
                shared.opts.data["sd_model_checkpoint"] = "black-forest-labs/FLUX.1-Canny-dev"
                sd_models.reload_model_weights(op='model', revision="refs/pr/1")
            if processor_canny is None:
                from controlnet_aux import CannyDetector
                processor_canny = CannyDetector()
            control_image = processor_canny(image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
            p.task_args['control_image'] = control_image
            if strength:
                p.task_args['strength'] = None
        else:
            if processor_canny is not None:
                shared.log.debug(f'{title}: tool=Canny unload processor')
                processor_canny = None

        if tool == 'Depth':
            # pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16, revision="refs/pr/1").to("cuda")
            install('git+https://github.com/asomoza/image_gen_aux.git', 'image_gen_aux')
            if shared.sd_model.__class__.__name__ != 'FluxControlPipeline' or 'Depth' not in shared.opts.sd_model_checkpoint:
                shared.opts.data["sd_model_checkpoint"] = "black-forest-labs/FLUX.1-Depth-dev"
                sd_models.reload_model_weights(op='model', revision="refs/pr/1")
            if processor_depth is None:
                from image_gen_aux import DepthPreprocessor
                processor_depth = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
            control_image = processor_depth(control_image)[0].convert("RGB")
            p.task_args['control_image'] = control_image
            if strength:
                p.task_args['strength'] = None
        else:
            if processor_depth is not None:
                shared.log.debug(f'{title}: tool=Depth unload processor')
                processor_depth = None

        shared.log.debug(f'{title}: tool={tool} ready time={time.time() - t0:.2f}')
        devices.torch_gc()
