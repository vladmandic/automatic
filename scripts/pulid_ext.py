import time
import gradio as gr
import numpy as np
from PIL import Image
from modules import shared, devices, errors, sd_models, scripts, processing, processing_helpers


pulid = None


class Script(scripts.Script):
    def __init__(self):
        self.images = []
        super().__init__()
        # self.register() # pulid is script with processing override so xyz doesnt execute

    def title(self):
        return 'PuLID'

    def show(self, _is_img2img):
        return not _is_img2img

    def dependencies(self):
        from installer import install, installed
        # if not installed('apex', reload=False, quiet=True):
        #     install('apex', 'apex', ignore=False)
        if not installed('insightface', reload=False, quiet=True):
            install('insightface', 'insightface', ignore=False)
            install('albumentations==1.4.3', 'albumentations', ignore=False, reinstall=True)
            install('pydantic==1.10.15', 'pydantic', ignore=False, reinstall=True)

    def register(self): # register xyz grid elements
        def apply_field(field):
            def fun(p, x, xs): # pylint: disable=unused-argument
                setattr(p, field, x)
                self.run(p)
            return fun

        import sys
        xyz_classes = [v for k, v in sys.modules.items() if 'xyz_grid_classes' in k][0]
        xyz_classes.axis_options.append(xyz_classes.AxisOption("[PuLID] Strength", float, apply_field("pulid_strength")))
        xyz_classes.axis_options.append(xyz_classes.AxisOption("[PuLID] Zero", float, apply_field("pulid_zero")))

    def load_images(self, files):
        self.images = []
        for file in files or []:
            try:
                if isinstance(file, str):
                    from modules.api.api import decode_base64_to_image
                    image = decode_base64_to_image(file)
                elif isinstance(file, Image.Image):
                    image = file
                elif isinstance(file, dict) and 'name' in file:
                    image = Image.open(file['name']) # _TemporaryFileWrapper from gr.Files
                elif hasattr(file, 'name'):
                    image = Image.open(file.name) # _TemporaryFileWrapper from gr.Files
                else:
                    raise ValueError(f'IP adapter unknown input: {file}')
                self.images.append(image)
            except Exception as e:
                shared.log.warning(f'IP adapter failed to load image: {e}')
        return gr.update(value=self.images, visible=len(self.images) > 0)

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/ToTheBeginning/PuLID">&nbsp PuLID: Pure and Lightning ID Customization</a><br>')
        with gr.Row():
            strength = gr.Slider(label = 'Strength', value = 0.8, mininimum = 0, maximum = 1, step = 0.01)
            zero = gr.Slider(label = 'Zero', value = 20, mininimum = 0, maximum = 80, step = 1)
        with gr.Row():
            sampler = gr.Dropdown(label="Sampler", choices=['dpmpp_sde', 'dpmpp_2m'], value='dpmpp_sde', visible=True)
            ortho = gr.Dropdown(label="Ortho", choices=['off', 'v1', 'v2'], value='v2', visible=True)
        with gr.Row():
            files = gr.File(label='Input images', file_count='multiple', file_types=['image'], type='file', interactive=True, height=100)
        with gr.Row():
            gallery = gr.Gallery(show_label=False, value=[], visible=False, container=False, rows=1)
        files.change(fn=self.load_images, inputs=[files], outputs=[gallery])
        return [strength, zero, sampler, ortho, gallery]

    def run(self, p: processing.StableDiffusionProcessing, strength: float = 0.8, zero: int = 20, sampler: str = 'dpmpp_sde', ortho: str = 'v2', gallery: list = []): # pylint: disable=arguments-differ
        global pulid # pylint: disable=global-statement
        images = []
        try:
            if len(gallery) == 0:
                gallery = self.images
            images = [Image.open(f['name']) for f in gallery if isinstance(f, dict)]
            images = [np.array(image) for image in images]
        except Exception as e:
            shared.log.error(f'PuLID: failed to load images: {e}')
            return None
        if len(images) == 0:
            shared.log.error('PuLID: no images loaded')
            return None
        supported_model_list = ['sdxl']
        if shared.sd_model_type not in supported_model_list:
            shared.log.error(f'PuLID: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        if pulid is None:
            self.dependencies()
            try:
                from modules import pulid # pylint: disable=redefined-outer-name
            except Exception as e:
                shared.log.error(f'PuLID: failed to import library: {e}')
                return None
            # import os
            # import importlib
            # module_path = os.path.join(os.path.dirname(__file__), '..', 'pulid', '__init__.py')
            # module_spec = importlib.util.spec_from_file_location('pulid', module_path)
            # pulid = importlib.util.module_from_spec(module_spec)
            # module_spec.loader.exec_module(pulid)
        if pulid is None:
            shared.log.error('PuLID: failed to load PuLID library')
            return None
        if p.batch_size > 1:
            shared.log.warning('PuLID: batch size not supported')
            p.batch_size = 1
        strength = getattr(p, 'pulid_strength', strength)
        zero = getattr(p, 'pulid_zero', zero)

        processing.fix_seed(p)
        pipe = None
        if shared.sd_model_type == 'sdxl':
            # TODO pulid has monolithic inference so not really working with offloading
            sd_models.move_model(shared.sd_model, devices.device)
            sd_models.move_model(shared.sd_model.vae, devices.device)
            sd_models.move_model(shared.sd_model.unet, devices.device)
            sd_models.move_model(shared.sd_model.text_encoder, devices.device)
            sd_models.move_model(shared.sd_model.text_encoder_2, devices.device)
            try:
                pipe = pulid.PuLIDPipelineXL(
                    pipe =shared.sd_model,
                    device=devices.device,
                    sampler=sampler,
                    cache_dir=shared.opts.hfcache_dir,
                )
            except Exception as e:
                shared.log.error(f'PuLID: failed to create pipeline: {e}')
                errors.display(e, 'PuLID')
                return None
        if pipe is None:
            return None
        shared.state.begin('PuLID')
        shared.log.info(f'PuLID: class={pipe.__class__.__name__} strength={strength} zero={zero} ortho={ortho} sampler={sampler} images={[i.shape for i in images]}')

        pipe.debug_img_list = []
        pulid.attention.NUM_ZERO = zero
        if ortho == 'v2':
            pulid.attention.ORTHO = False
            pulid.attention.ORTHO_v2 = True
        elif ortho == 'v1':
            pulid.attention.ORTHO = True
            pulid.attention.ORTHO_v2 = False
        else:
            pulid.attention.ORTHO = False
            pulid.attention.ORTHO_v2 = False

        t0 = time.time()
        images = [pulid.resize(image, 1024) for image in images]
        outputs = []
        infotexts = []
        seeds = []
        prompts = []
        negative_prompts = []

        for _n in range(p.n_iter):
            seed = processing_helpers.get_fixed_seed(p.seed)
            prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
            negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
            with devices.inference_context():
                uncond_id_embedding, id_embedding = pipe.get_id_embedding(images)
                output = pipe.inference(prompt, (1, p.height, p.width), negative_prompt, id_embedding, uncond_id_embedding, strength, p.cfg_scale, p.steps, seed)[0]
            if output is not None:
                outputs.append(output)
                infotexts.append(processing.create_infotext(p))
                seeds.append(seed)
                prompts.append(prompt)
                negative_prompts.append(negative_prompt)

        interim = [Image.fromarray(face) for face in pipe.debug_img_list]
        t1 = time.time()
        shared.log.debug(f'PuLID: output={output} interim={interim} time={t1-t0:.2f}')

        if len(outputs) > 0:
            p.prompt = prompts[0]
            p.negative_prompt = negative_prompts[0]
            p.seed = seeds[0]
            p.all_prompts = prompts
            p.all_negative_prompts = negative_prompts
            p.all_seeds = seeds
            p.extra_generation_params["PuLID"] = f'Strength={strength} Zero={zero} Ortho={ortho}'
        processed = processing.Processed(p, outputs, infotexts=infotexts)

        shared.state.end('PuLID')
        return processed
