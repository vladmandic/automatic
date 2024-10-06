import os
import numpy as np
from PIL import Image, ImageDraw
from modules import shared, processing
from modules.detailer import Detailer
from modules import devices, processing_class


PREDEFINED = { # <https://huggingface.co/vladmandic/yolo-detailers/tree/main>
    'Face yolo-8n': 'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/face-yolo8n.pt',
    'Eyefull paired v2': 'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/eyeful-paired-v2.pt',
}


class YoloResult:
    def __init__(self, score: float, box: list[int], mask: Image.Image = None, item: Image.Image = None, size: float = 0, width = 0, height = 0, args = {}):
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
        self.models = {}
        self.list = {}
        self.enumerate()

    def name(self):
        return "Detailer"

    def enumerate(self):
        self.list.clear()
        files = []
        for k, v in PREDEFINED.items():
            self.list[k] = v
            files.append(os.path.basename(v))
        for f in os.listdir(shared.opts.yolo_dir):
            if f not in files:
                name = os.path.basename(f)
                self.list[name] = os.path.join(shared.opts.yolo_dir, f)
        shared.log.info(f'Available Yolo: path="{shared.opts.yolo_dir} items={len(list(self.list))}')

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

        args = {
            'conf': shared.opts.detailer_conf,
            'iou': shared.opts.detailer_iou,
            'max_det': shared.opts.detailer_max,
        }
        model.to(device)
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

        result = []
        for prediction in predictions:
            boxes = prediction.boxes.xyxy.detach().int().cpu().numpy() if prediction.boxes is not None else []
            scores = prediction.boxes.conf.detach().float().cpu().numpy() if prediction.boxes is not None else []
            for score, box in zip(scores, boxes):
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
                    result.append(YoloResult(score=round(score, 2), box=box, mask=mask_image, item=cropped, size=size, width=w, height=h, args=args))
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
            model_file = modelloader.load_file_from_url(url=model_url, model_dir=shared.opts.yolo_dir, file_name=file_name)
            if model_file is not None:
                shared.log.info(f'Load: type=Detailer name="{model_name}" model="{model_file}"')
                from ultralytics import YOLO # pylint: disable=import-outside-toplevel
                model = YOLO(model_file)
                self.models[model_name] = model
                return model_name, model
        return None

    def restore(self, np_image, p: processing.StableDiffusionProcessing = None):
        if hasattr(p, 'recursion'):
            return
        if not hasattr(p, 'detailer_active'):
            p.detailer_active = 0
        if np_image is None or p.detailer_active >= p.batch_size * p.n_iter:
            return np_image
        name, model = self.load()
        if model is None:
            shared.log.warning(f'Detailer: model="{name}" not loaded')
            return np_image
        image = Image.fromarray(np_image)
        items = self.predict(model, image)
        if len(items) == 0:
            shared.log.info(f'Detailer: model="{name}" no items detected')
            return np_image

        # create backups
        orig_apply_overlay = shared.opts.mask_apply_overlay
        orig_p = p.__dict__.copy()
        orig_cls = p.__class__

        pp = None
        shared.opts.data['mask_apply_overlay'] = True
        resolution = 512 if shared.sd_model_type in ['none', 'sd', 'lcm', 'unknown'] else 1024
        args = {
            'batch_size': 1,
            'n_iter': 1,
            'inpaint_full_res': True,
            'inpainting_mask_invert': 0,
            'inpainting_fill': 1, # no fill
            'sampler_name': orig_p.get('hr_sampler_name', 'default'),
            'steps': orig_p.get('hr_second_pass_steps', 0),
            'negative_prompt': orig_p.get('refiner_negative', ''),
            'denoising_strength': shared.opts.detailer_strength if shared.opts.detailer_strength > 0 else orig_p.get('denoising_strength', 0.3),
            'styles': [],
            'prompt': orig_p.get('refiner_prompt', ''),
            'mask_blur': 10,
            'inpaint_full_res_padding': shared.opts.detailer_padding,
            'detailer': True,
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
        if len(p.prompt) == 0:
            p.prompt = orig_p.get('all_prompts', [''])[0]
        if len(p.negative_prompt) == 0:
            p.negative_prompt = orig_p.get('all_negative_prompts', [''])[0]

        report = [{'score': i.score, 'size': f'{i.width}x{i.height}' } for i in items]
        shared.log.info(f'Detailer: model="{name}" items={report} args={items[0].args} denoise={p.denoising_strength} blur={p.mask_blur} width={p.width} height={p.height} padding={p.inpaint_full_res_padding}')

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
        p.init_images = getattr(orig_p, 'init_images', None)
        p.image_mask = getattr(orig_p, 'image_mask', None)
        p.state = getattr(orig_p, 'state', None)
        shared.opts.data['mask_apply_overlay'] = orig_apply_overlay
        np_image = np.array(image)

        if len(mask_all) > 0 and shared.opts.include_mask:
            from modules.control.util import blend
            p.image_mask = blend([np.array(m) for m in mask_all])
            # combined = blend([np_image, p.image_mask])
            # combined = Image.fromarray(combined)
            # combined.save('/tmp/item.png')
            p.image_mask = Image.fromarray(p.image_mask)
        return np_image


yolo = YoloRestorer()
shared.detailers.append(yolo)
shared.yolo = yolo
