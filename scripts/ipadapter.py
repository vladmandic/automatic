from PIL import Image
import gradio as gr
from modules import scripts, processing, shared, ipadapter


MAX_ADAPTERS = 4


class Script(scripts.Script):
    standalone = True

    def title(self):
        return 'IP Adapters'

    def show(self, is_img2img):
        return scripts.AlwaysVisible if shared.backend == shared.Backend.DIFFUSERS else False

    def load_images(self, files):
        init_images = []
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
                init_images.append(image)
            except Exception as e:
                shared.log.warning(f'IP adapter failed to load image: {e}')
        return init_images

    def display_units(self, num_units):
        num_units = num_units or 1
        return (num_units * [gr.update(visible=True)]) + ((MAX_ADAPTERS - num_units) * [gr.update(visible=False)])

    def ui(self, _is_img2img):
        with gr.Accordion('IP Adapters', open=False, elem_id='ipadapter'):
            units = []
            adapters = []
            scales = []
            starts = []
            ends = []
            files = []
            galleries = []
            with gr.Row():
                num_adapters = gr.Slider(label="Active IP adapters", minimum=1, maximum=MAX_ADAPTERS, step=1, value=1, scale=1)
            for i in range(MAX_ADAPTERS):
                with gr.Accordion(f'Adapter {i+1}', visible=i==0) as unit:
                    with gr.Row():
                        adapters.append(gr.Dropdown(label='Adapter', choices=list(ipadapter.ADAPTERS), value='None'))
                        scales.append(gr.Slider(label='Scale', minimum=0.0, maximum=1.0, step=0.01, value=0.5))
                    with gr.Row():
                        starts.append(gr.Slider(label='Start', minimum=0.0, maximum=1.0, step=0.1, value=0))
                        ends.append(gr.Slider(label='End', minimum=0.0, maximum=1.0, step=0.1, value=1))
                    with gr.Row():
                        files.append(gr.File(label='Input images', file_count='multiple', file_types=['image'], type='file', interactive=True, height=100))
                    with gr.Row():
                        galleries.append(gr.Gallery(show_label=False, value=[]))
                    files[i].change(fn=self.load_images, inputs=[files[i]], outputs=[galleries[i]])
                units.append(unit)
            num_adapters.change(fn=self.display_units, inputs=[num_adapters], outputs=units)
        return [num_adapters] + adapters + scales + files + starts + ends

    def process(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ
        if shared.backend != shared.Backend.DIFFUSERS:
            return
        args = list(args)
        units = args.pop(0)
        if getattr(p, 'ip_adapter_names', []) == []:
            p.ip_adapter_names = args[:MAX_ADAPTERS][:units]
        if getattr(p, 'ip_adapter_scales', [0.0]) == [0.0]:
            p.ip_adapter_scales = args[MAX_ADAPTERS:MAX_ADAPTERS*2][:units]
        if getattr(p, 'ip_adapter_images', []) == []:
            p.ip_adapter_images = args[MAX_ADAPTERS*2:MAX_ADAPTERS*3][:units]
        if getattr(p, 'ip_adapter_starts', [0.0]) == [0.0]:
            p.ip_adapter_starts = args[MAX_ADAPTERS*3:MAX_ADAPTERS*4][:units]
        if getattr(p, 'ip_adapter_ends', [1.0]) == [1.0]:
            p.ip_adapter_ends = args[MAX_ADAPTERS*4:MAX_ADAPTERS*5][:units]
        # ipadapter.apply(shared.sd_model, p, p.ip_adapter_names, p.ip_adapter_scales, p.ip_adapter_starts, p.ip_adapter_ends, p.ip_adapter_images) # called directly from processing.process_images_inner
