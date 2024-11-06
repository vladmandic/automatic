"""
original code from <https://github.com/NVlabs/consistory>
ported to modules/consistory
- do not load pipeline and unet, use existing model
- uses diffusers class definitions from 0.25 needed updates
- forces uses of xformers, converted attention calls to sdp
- unsafe tensor to numpy breaks with bfloat16
- removed debug print statements
"""
import gradio as gr
import diffusers
from modules import scripts, processing, shared, sd_models, devices


class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None

    def title(self):
        return 'ConsiStory'

    def show(self, is_img2img):
        return not is_img2img if shared.native and shared.cmd_opts.experimental else False

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/NVlabs/consistory">&nbsp ConsiStory: Consistent Image Generation</a><br>')
        with gr.Row():
            pass
        return []

    def run(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ
        supported_model_list = ['sdxl']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'ConsiStory: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        diffusers.models.embeddings.PositionNet = diffusers.models.embeddings.GLIGENTextBoundingboxProjection # patch as renamed in https://github.com/huggingface/diffusers/pull/6244/files
        import modules.consistory as cs
        if shared.sd_model_type == "sdxl":
            self.orig_pipe = shared.sd_model
            state_dict = shared.sd_model.unet.state_dict()
            pipe = sd_models.switch_pipe(cs.ConsistoryExtendAttnSDXLPipeline, shared.sd_model)
            pipe.unet = cs.ConsistorySDXLUNet2DConditionModel.from_config(pipe.unet.config)
            pipe.unet.load_state_dict(state_dict)
            pipe.unet.to(device=devices.device, dtype=devices.dtype)
            # sd_models.set_diffuser_options(pipe)
            devices.torch_gc(force=True)

        processing.fix_seed(p)
        subject="digital image of a cute robot"
        concept_token=['robot']
        settings=["sitting in the beach", "standing in the snow", "playing on the beach", "dancing in the meadow"]
        prompts = [f'{subject} {setting}' for setting in settings]
        anchor_prompts = prompts[:1]
        extra_prompts = prompts[1:]

        p.steps = 50

        images = []
        anchor_out_images, anchor_cache_first_stage, anchor_cache_second_stage = cs.run_anchor_generation(
            story_pipeline=pipe,
            prompts=anchor_prompts,
            concept_token=concept_token,
            seed=p.seed,
            n_steps=p.steps,
            mask_dropout=0.5,
            same_latent=False,
            share_queries=True,
            perform_sdsa=True,
            perform_injection=True,
        )
        devices.torch_gc(force=True)
        for i, image in enumerate(anchor_out_images):
            image.save(f'/tmp/anchor_image_{i}.png')
            images.append(image)

        extra_out_images = cs.run_extra_generation(
            story_pipeline=pipe,
            prompts=extra_prompts,
            concept_token=concept_token,
            anchor_cache_first_stage=anchor_cache_first_stage,
            anchor_cache_second_stage=anchor_cache_second_stage,
            seed=p.seed,
            n_steps=p.steps,
            mask_dropout=0.5,
            same_latent=False,
            share_queries=True,
            perform_sdsa=True,
            perform_injection=True,
        )
        for j, image in enumerate(extra_out_images):
            image.save(f'/tmp/extra_image_{j}.png')
            images.append(image)

        devices.torch_gc(force=True)
        processed = processing.Processed(p, images_list=images)
        return processed


    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args): # pylint: disable=arguments-differ, unused-argument
        if self.orig_pipe is None:
            return processed
        shared.sd_model = self.orig_pipe
        return processed
