import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines import pipeline_utils
from modules import shared, devices, scripts, processing, sd_models, prompt_parser_diffusers

def hijack_register_modules(self, **kwargs):
    for name, module in kwargs.items():
        register_dict = None
        if module is None or (isinstance(module, (tuple, list)) and module[0] is None):
            register_dict = {name: (None, None)}
        elif isinstance(module, bool):
            pass
        else:
            library, class_name = pipeline_utils._fetch_class_library_tuple(module) # pylint: disable=protected-access
            register_dict = {name: (library, class_name)}
        if register_dict is not None:
            self.register_to_config(**register_dict)
        setattr(self, name, module)

def generate_layout_template(layout_type, resolution):
    width, height = resolution if resolution else (512, 512)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    if layout_type.lower() == "row":
        row_height = height // 2
        draw.rectangle([0, 0, width, row_height], outline="black", width=2)
        draw.rectangle([0, row_height, width, height], outline="black", width=2)
        draw.text((width // 2 - 40, row_height // 2), "Row 1", fill="black", font=font)
        draw.text((width // 2 - 40, row_height + row_height // 2), "Row 2", fill="black", font=font)
    elif layout_type.lower() == "column":
        col_width = width // 2
        draw.rectangle([0, 0, col_width, height], outline="black", width=2)
        draw.rectangle([col_width, 0, width, height], outline="black", width=2)
        draw.text((col_width // 2, height // 2), "Col 1", fill="black", font=font)
        draw.text((col_width + col_width // 2, height // 2), "Col 2", fill="black", font=font)
    return image

class Script(scripts.Script):
    def title(self):
        return 'Regional prompting'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    def change(self, mode):
        return [gr.update(visible='Col' in mode or 'Row' in mode), gr.update(visible='Prompt' in mode)]

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline">&nbsp Regional prompting</a><br>')
        with gr.Row():
            mode = gr.Radio(label='Mode', choices=['None', 'Prompt', 'Prompt EX', 'Columns', 'Rows'], value='None')
            show_template = gr.Button("Show Template")
        with gr.Row():
            power = gr.Slider(label='Power', minimum=0, maximum=1, value=1.0, step=0.01)
            threshold = gr.Textbox('', label='Prompt thresholds:', default='', visible=False)
            grid = gr.Text('', label='Grid sections:', default='', visible=False)
            template_output = gr.Image(label="Layout Template")
        
        mode.change(fn=self.change, inputs=[mode], outputs=[grid, threshold])
        show_template.click(fn=lambda mode: generate_layout_template(mode, (512, 512)), inputs=[mode], outputs=template_output)
        
        return mode, grid, power, threshold

    def run(self, p: processing.StableDiffusionProcessing, mode, grid, power, threshold): # pylint: disable=arguments-differ
        if mode is None or mode == 'None':
            return
        orig_pipeline = shared.sd_model
        orig_dtype = devices.dtype
        orig_prompt_attention = shared.opts.prompt_attention
        if shared.sd_model_type != 'sd':
            shared.log.error(f'Regional prompting: incorrect base model: {shared.sd_model.__class__.__name__}')
            return
        
        pipeline_utils.DiffusionPipeline.register_modules = hijack_register_modules
        prompt_parser_diffusers.EmbeddingsProvider._encode_token_ids_to_embeddings = prompt_parser_diffusers.orig_encode_token_ids_to_embeddings

        shared.sd_model = sd_models.switch_pipe('regional_prompting_stable_diffusion', shared.sd_model)
        if shared.sd_model.__class__.__name__ != 'RegionalPromptingStableDiffusionPipeline':
            shared.log.error(f'Regional prompting: not a tiling pipeline: {shared.sd_model.__class__.__name__}')
            shared.sd_model = orig_pipeline
            return
        
        sd_models.set_diffuser_options(shared.sd_model)
        shared.opts.data['prompt_attention'] = 'fixed' # this pipeline is not compatible with embeds
        processing.fix_seed(p)
        # set pipeline specific params, note that standard params are applied when applicable
        rp_args = {
            'mode': mode.lower(),
            'power': power,
        }
        if 'prompt' in mode.lower():
            rp_args['th'] = threshold
        else:
            rp_args['div'] = grid
        p.task_args = {
            **p.task_args,
            'prompt': p.prompt,
            'rp_args': rp_args,
        }
        
        shared.log.debug(f'Regional: args={p.task_args}')
        p.task_args['prompt'] = p.prompt
        processed: processing.Processed = processing.process_images(p) # runs processing using main loop

        # restore pipeline and params
        prompt_parser_diffusers.EmbeddingsProvider._encode_token_ids_to_embeddings = prompt_parser_diffusers.compel_hijack # pylint: disable=protected-access
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        shared.sd_model = orig_pipeline
        shared.sd_model.to(orig_dtype)
        
        return processed
