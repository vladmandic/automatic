from typing import TYPE_CHECKING
import os
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from modules import shared, processing, devices, processing_class, ui_common
from modules.detailer import Detailer


PREDEFINED = [ # <https://huggingface.co/vladmandic/yolo-detailers/tree/main>
    'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/face-yolo8n.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/hand_yolov8n.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/person_yolov8n-seg.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/eyes-v1.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/eyes-full-v1.pt',
]


class YoloResult:
    def __init__(self, cls: int, label: str, score: float, box: list[int], mask: Image.Image = None, item: Image.Image = None, size: float = 0, width = 0, height = 0, args = {}):
        self.cls = cls
        self.label = label
        self.score = score
        self.box = box
        self.mask = mask
        self.item = item
        self.size = size
        self.width = width
        self.height = height
        self.args = args


class YoloRestorer(Detailer):
    def __init__(self):
        super().__init__()
        self.models = {} # cache loaded models
        self.list = {}
        self.enumerate()

    def name(self):
        return "Detailer"

    def enumerate(self):
        self.list.clear()
        files = []
        downloaded = 0
        for m in PREDEFINED:
            name = os.path.splitext(os.path.basename(m))[0]
            self.list[name] = m
            files.append(name)
        if os.path.exists(shared.opts.yolo_dir):
            for f in os.listdir(shared.opts.yolo_dir):
                if f.endswith('.pt'):
                    downloaded += 1
                    name = os.path.splitext(os.path.basename(f))[0]
                    if name not in files:
                        self.list[name] = os.path.join(shared.opts.yolo_dir, f)
        shared.log.info(f'Available Yolo: path="{shared.opts.yolo_dir} items={len(list(self.list))} downloaded={downloaded}')
        return self.list

    def dependencies(self):
        import installer
        installer.install('ultralytics', ignore=True, quiet=True)

    def predict(
            self,
            model,
            image: Image.Image,
            imgsz: int = 640,
            half: bool = True,
            device = devices.device,
            augment: bool = True,
            agnostic: bool = False,
            retina: bool = False,
            mask: bool = True,
            offload: bool = shared.opts.detailer_unload,
        ) -> list[YoloResult]:

        result = []
        if model is None:
            return result
        args = {
            'conf': shared.opts.detailer_conf,
            'iou': shared.opts.detailer_iou,
            # 'max_det': shared.opts.detailer_max,
        }
        try:
            if TYPE_CHECKING:
                from ultralytics import YOLO # pylint: disable=import-outside-toplevel, unused-import
            model: YOLO = model.to(device)
            predictions = model.predict(
                source=[image],
                stream=False,
                verbose=False,
                imgsz=imgsz,
                half=half,
                device=device,
                augment=augment,
                agnostic_nms=agnostic,
                retina_masks=retina,
                **args
            )
            if offload:
                model.to('cpu')
        except Exception as e:
            shared.log.error(f'Detailer predict: {e}')
            return result

        desired = shared.opts.detailer_classes.split(',')
        desired = [d.lower().strip() for d in desired]
        desired = [d for d in desired if len(d) > 0]

        for prediction in predictions:
            boxes = prediction.boxes.xyxy.detach().int().cpu().numpy() if prediction.boxes is not None else []
            scores = prediction.boxes.conf.detach().float().cpu().numpy() if prediction.boxes is not None else []
            classes = prediction.boxes.cls.detach().float().cpu().numpy() if prediction.boxes is not None else []
            for score, box, cls in zip(scores, boxes, classes):
                cls = int(cls)
                label = prediction.names[cls] if cls < len(prediction.names) else f'cls{cls}'
                if len(desired) > 0 and label.lower() not in desired:
                    continue
                box = box.tolist()
                mask_image = None
                w, h = box[2] - box[0], box[3] - box[1]
                size = w * h / (image.width * image.height)
                if (min(w, h) > shared.opts.detailer_min_size if shared.opts.detailer_min_size > 0 else True) and (max(w, h) < shared.opts.detailer_max_size if shared.opts.detailer_max_size > 0 else True):
                    if mask:
                        mask_image = image.copy()
                        mask_image = Image.new('L', image.size, 0)
                        draw = ImageDraw.Draw(mask_image)
                        draw.rectangle(box, fill="white", outline=None, width=0)
                        cropped = image.crop(box)
                        result.append(YoloResult(cls=cls, label=label, score=round(score, 2), box=box, mask=mask_image, item=cropped, size=size, width=w, height=h, args=args))
                if len(result) >= shared.opts.detailer_max:
                    break
        return result

    def load(self, model_name: str = None):
        from modules import modelloader
        self.dependencies()
        if model_name is None:
            model_name = list(self.list)[0]
        if model_name in self.models:
            return model_name, self.models[model_name]
        else:
            model_url = self.list.get(model_name)
            file_name = os.path.basename(model_url)
            model_file = None
            try:
                model_file = modelloader.load_file_from_url(url=model_url, model_dir=shared.opts.yolo_dir, file_name=file_name)
                if model_file is not None:
                    from ultralytics import YOLO # pylint: disable=import-outside-toplevel
                    model = YOLO(model_file)
                    classes = list(model.names.values())
                    shared.log.info(f'Load: type=Detailer name="{model_name}" model="{model_file}" classes={classes}')
                    self.models[model_name] = model
                    return model_name, model
            except Exception as e:
                shared.log.error(f'Load: type=Detailer name="{model_name}" error="{e}"')
        return None

    def restore(self, np_image, p: processing.StableDiffusionProcessing = None):
        if hasattr(p, 'recursion'):
            return np_image
        if not hasattr(p, 'detailer_active'):
            p.detailer_active = 0
        if np_image is None or p.detailer_active >= p.batch_size * p.n_iter:
            return np_image
        if len(shared.opts.detailer_models) == 0:
            shared.log.warning('Detailer: model=None')
            return np_image
        models_used = []
        # create backups
        orig_apply_overlay = shared.opts.mask_apply_overlay
        orig_p = p.__dict__.copy()
        orig_cls = p.__class__

        for i, model_name in enumerate(shared.opts.detailer_models):
            name, model = self.load(model_name)
            if model is None:
                shared.log.warning(f'Detailer: model="{name}" not loaded')
                continue

            image = Image.fromarray(np_image)
            items = self.predict(model, image)
            if len(items) == 0:
                shared.log.info(f'Detailer: model="{name}" no items detected')
                continue

            pp = None
            shared.opts.data['mask_apply_overlay'] = True
            resolution = 512 if shared.sd_model_type in ['none', 'sd', 'lcm', 'unknown'] else 1024
            prompt: str = orig_p.get('refiner_prompt', '')
            negative: str = orig_p.get('refiner_negative', '')
            if len(prompt) == 0:
                prompt = orig_p.get('all_prompts', [''])[0]
            if len(negative) == 0:
                negative = orig_p.get('all_negative_prompts', [''])[0]
            prompt_lines = prompt.split('\n')
            negative_lines = negative.split('\n')
            prompt = prompt_lines[i % len(prompt_lines)]
            negative = negative_lines[i % len(negative_lines)]

            args = {
                'detailer': True,
                'batch_size': 1,
                'n_iter': 1,
                'prompt': prompt,
                'negative_prompt': negative,
                'denoising_strength': shared.opts.detailer_strength,
                'sampler_name': orig_p.get('hr_sampler_name', 'default'),
                'steps': orig_p.get('refiner_steps', 0),
                'styles': [],
                'inpaint_full_res': True,
                'inpainting_mask_invert': 0,
                'inpainting_fill': 1, # no fill
                'mask_blur': shared.opts.detailer_blur,
                'inpaint_full_res_padding': shared.opts.detailer_padding,
                'width': resolution,
                'height': resolution,
            }
            if args['denoising_strength'] == 0:
                shared.log.debug(f'Detailer: model="{name}" strength=0 skip')
                return np_image
            control_pipeline = None
            orig_class = shared.sd_model.__class__
            if getattr(p, 'is_control', False):
                from modules.control import run
                control_pipeline = shared.sd_model
                run.restore_pipeline()

            p = processing_class.switch_class(p, processing.StableDiffusionProcessingImg2Img, args)
            p.detailer_active += 1 # set flag to avoid recursion

            if p.steps < 1:
                p.steps = orig_p.get('steps', 0)

            report = [{'label': i.label, 'score': i.score, 'size': f'{i.width}x{i.height}' } for i in items]
            shared.log.info(f'Detailer: model="{name}" items={report} args={items[0].args} denoise={p.denoising_strength} blur={p.mask_blur} width={p.width} height={p.height} padding={p.inpaint_full_res_padding}')
            shared.log.debug(f'Detailer: prompt="{prompt}" negative="{negative}"')
            models_used.append(name)

            mask_all = []
            p.state = ''
            for item in items:
                if item.mask is None:
                    continue
                p.init_images = [image]
                p.image_mask = [item.mask]
                # mask_all.append(item.mask)
                p.recursion = True
                pp = processing.process_images_inner(p)
                del p.recursion
                p.overlay_images = None # skip applying overlay twice
                if pp is not None and pp.images is not None and len(pp.images) > 0:
                    image = pp.images[0] # update image to be reused for next item
                    if len(pp.images) > 1:
                        mask_all.append(pp.images[1])

            # restore pipeline
            if control_pipeline is not None:
                shared.sd_model = control_pipeline
            else:
                shared.sd_model.__class__ = orig_class
            p = processing_class.switch_class(p, orig_cls, orig_p)
            p.init_images = orig_p.get('init_images', None)
            p.image_mask = orig_p.get('image_mask', None)
            p.state = orig_p.get('state', None)
            p.ops = orig_p.get('ops', [])
            shared.opts.data['mask_apply_overlay'] = orig_apply_overlay
            np_image = np.array(image)

            if len(mask_all) > 0 and shared.opts.include_mask:
                from modules.control.util import blend
                p.image_mask = blend([np.array(m) for m in mask_all])
                # combined = blend([np_image, p.image_mask])
                # combined = Image.fromarray(combined)
                # combined.save('/tmp/item.png')
                p.image_mask = Image.fromarray(p.image_mask)

        shared.log.debug(f'Detailer processed: models={models_used}')
        return np_image

    def ui(self, tab: str):
        def ui_settings_change(detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou):
            shared.opts.detailer_models = detailers
            shared.opts.detailer_classes = classes
            shared.opts.detailer_strength = strength
            shared.opts.detailer_padding = padding
            shared.opts.detailer_blur = blur
            shared.opts.detailer_conf = min_confidence
            shared.opts.detailer_max = max_detected
            shared.opts.detailer_min_size = min_size
            shared.opts.detailer_max_size = max_size
            shared.opts.detailer_iou = iou
            shared.opts.save(shared.config_filename, silent=True)
            shared.log.debug(f'Detailer settings: models={shared.opts.detailer_models} classes={shared.opts.detailer_classes} strength={shared.opts.detailer_strength} conf={shared.opts.detailer_conf} max={shared.opts.detailer_max} iou={shared.opts.detailer_iou} size={shared.opts.detailer_min_size}-{shared.opts.detailer_max_size} padding={shared.opts.detailer_padding}')

        with gr.Accordion(open=False, label="Detailer", elem_id=f"{tab}_detailer_accordion", elem_classes=["small-accordion"], visible=shared.native):
            with gr.Row():
                enabled = gr.Checkbox(label="Enable detailer pass", elem_id=f"{tab}_detailer_enabled", value=False)
            with gr.Row():
                detailers = gr.Dropdown(label="Detailers", elem_id=f"{tab}_detailers", choices=self.list, value=shared.opts.detailer_models, multiselect=True)
                ui_common.create_refresh_button(detailers, self.enumerate, {}, elem_id=f"{tab}_detailers_refresh")
            with gr.Row():
                classes = gr.Textbox(label="Classes", placeholder="Classes", elem_id=f"{tab}_detailer_classes")
            with gr.Row():
                strength = gr.Slider(label="Detailer strength", elem_id=f"{tab}_detailer_strength", value=shared.opts.detailer_strength, minimum=0, maximum=1, step=0.01)
                max_detected = gr.Slider(label="Max detected", elem_id=f"{tab}_detailer_max", value=shared.opts.detailer_max, min=1, maximum=10, step=1)
            with gr.Row():
                padding = gr.Slider(label="Edge padding", elem_id=f"{tab}_detailer_padding", value=shared.opts.detailer_padding, minimum=0, maximum=100, step=1)
                blur = gr.Slider(label="Edge blur", elem_id=f"{tab}_detailer_blur", value=shared.opts.detailer_blur, minimum=0, maximum=100, step=1)
            with gr.Row():
                min_confidence = gr.Slider(label="Min confidence", elem_id=f"{tab}_detailer_conf", value=shared.opts.detailer_conf, minimum=0.0, maximum=1.0, step=0.05)
                iou = gr.Slider(label="Max overlap", elem_id=f"{tab}_detailer_iou", value=shared.opts.detailer_iou, minimum=0, maximum=1.0, step=0.05)
            with gr.Row():
                min_size = gr.Slider(label="Min size", elem_id=f"{tab}_detailer_min_size", value=shared.opts.detailer_min_size, minimum=0, maximum=1024, step=1)
                max_size = gr.Slider(label="Max size", elem_id=f"{tab}_detailer_max_size", value=shared.opts.detailer_max_size, minimum=0, maximum=1024, step=1)
            detailers.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            classes.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            strength.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            padding.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            blur.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            min_confidence.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            max_detected.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            min_size.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            max_size.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            iou.change(fn=ui_settings_change, inputs=[detailers, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou], outputs=[])
            return enabled


def initialize():
    shared.yolo = YoloRestorer()
    shared.detailers.append(shared.yolo)
