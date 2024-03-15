import os
import numpy as np
from PIL import Image, ImageDraw
from modules import shared, processing
from modules.face_restoration import FaceRestoration


class YoLoResult:
    def __init__(self, score: float, box: list[int], mask: Image.Image = None, size: float = 0):
        self.score = score
        self.box = box
        self.mask = mask
        self.size = size


class FaceRestorerYolo(FaceRestoration):
    def name(self):
        return "Face HiRes"

    def __init__(self):
        from modules import paths
        self.model = None
        self.model_dir = os.path.join(paths.models_path, 'yolo')
        self.model_name = 'yolov8n-face.pt'
        self.model_url = 'https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt'
        # self.model_name = 'yolov9-c-face.pt'
        # self.model_url = 'https://github.com/akanametov/yolov9-face/releases/download/1.0/yolov9-c-face.pt'

    def dependencies(self):
        import installer
        installer.install('ultralytics', ignore=False)

    def predict(
            self,
            image: Image.Image,
            offload: bool = False,
            conf: float = 0.5,
            iou: float = 0.5,
            imgsz: int = 640,
            half: bool = True,
            device = 'cuda',
            n: int = 5,
            augment: bool = True,
            agnostic: bool = False,
            retina: bool = False,
            mask: bool = True,
        ) -> list[YoLoResult]:

        self.model.to(device)
        predictions = self.model.predict(
            source=[image],
            stream=False,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            half=half,
            device=device,
            max_det=n,
            augment=augment,
            agnostic_nms=agnostic,
            retina_masks=retina,
        )
        if offload:
            self.model.to('cpu')
        result = []
        for prediction in predictions:
            boxes = prediction.boxes.xyxy.detach().int().cpu().numpy() if prediction.boxes is not None else []
            scores = prediction.boxes.conf.detach().float().cpu().numpy() if prediction.boxes is not None else []
            for score, box in zip(scores, boxes):
                box = box.tolist()
                mask_image = None
                size = (box[2] - box[0]) * (box[3] - box[1]) / (image.width * image.height)
                if mask:
                    mask_image = image.copy()
                    mask_image = Image.new('L', image.size, 0)
                    draw = ImageDraw.Draw(mask_image)
                    draw.rectangle(box, fill="white", outline=None, width=0)
                result.append(YoLoResult(score=score, box=box, mask=mask_image, size=size))
        return result

    def load(self):
        from modules import modelloader
        self.dependencies()
        if self.model is None:
            model_file = modelloader.load_file_from_url(url=self.model_url, model_dir=self.model_dir, file_name=self.model_name)
            if model_file is not None:
                shared.log.info(f'Loading: type=FaceHires model={model_file}')
                from ultralytics import YOLO # pylint: disable=import-outside-toplevel
                self.model = YOLO(model_file)

    def restore(self, np_image, p: processing.StableDiffusionProcessing = None):
        from modules import devices, processing_class
        if not hasattr(p, 'facehires'):
            p.facehires = 0
        if np_image is None or p.facehires >= p.batch_size * p.n_iter:
            return np_image
        self.load()
        if self.model is None:
            shared.log.error(f"Model load: type=FaceHires model='{self.model_name}' dir={self.model_dir} url={self.model_url}")
            return np_image
        image = Image.fromarray(np_image)
        faces = self.predict(image, mask=True, device=devices.device, offload=shared.opts.face_restoration_unload)
        if len(faces) == 0:
            return np_image

        # create backups
        orig_apply_overlay = shared.opts.mask_apply_overlay
        orig_p = p.__dict__.copy()
        orig_cls = p.__class__

        pp = None
        shared.opts.data['mask_apply_overlay'] = True
        args = {
            'batch_size': 1,
            'n_iter': 1,
            'inpaint_full_res': True,
            'inpainting_mask_invert': 0,
            'inpainting_fill': 1, # no fill
            'sampler_name': orig_p.get('hr_sampler_name', 'default'),
            'steps': orig_p.get('hr_second_pass_steps', 0),
            'negative_prompt': orig_p.get('refiner_negative', ''),
            'denoising_strength': shared.opts.facehires_strength if shared.opts.facehires_strength > 0 else orig_p.get('denoising_strength', 0.3),
            'styles': [],
            'prompt': orig_p.get('refiner_prompt', ''),
            # TODO facehires expose as tunable
            'mask_blur': 10,
            'inpaint_full_res_padding': 15,
            'restore_faces': True,
        }
        p = processing_class.switch_class(p, processing.StableDiffusionProcessingImg2Img, args)
        p.facehires += 1 # set flag to avoid recursion

        if p.steps < 1:
            p.steps = orig_p.get('steps', 0)
        if len(p.prompt) == 0:
            p.prompt = orig_p.get('all_prompts', [''])[0]
        if len(p.negative_prompt) == 0:
            p.negative_prompt = orig_p.get('all_negative_prompts', [''])[0]

        shared.log.debug(f'Face HiRes: faces={[f.__dict__ for f in faces]} strength={p.denoising_strength} blur={p.mask_blur} padding={p.inpaint_full_res_padding} steps={p.steps}')
        for face in faces:
            if face.mask is None:
                continue
            if face.size < 0.0002 or face.size > 0.8:
                shared.log.debug(f'Face HiRes skip: {face.__dict__}')
                continue
            p.init_images = [image]
            p.image_mask = [face.mask]
            p.recursion = True
            pp = processing.process_images_inner(p)
            del p.recursion
            p.overlay_images = None # skip applying overlay twice
            if pp is not None and pp.images is not None and len(pp.images) > 0:
                image = pp.images[0] # update image to be reused for next face

        # restore pipeline
        p = processing_class.switch_class(p, orig_cls, orig_p)
        p.init_images = getattr(orig_p, 'init_images', None)
        p.image_mask = getattr(orig_p, 'image_mask', None)
        shared.opts.data['mask_apply_overlay'] = orig_apply_overlay
        np_image = np.array(image)
        # shared.log.debug(f'Face HiRes complete: faces={len(faces)} time={t1-t0:.3f}')
        return np_image


yolo = FaceRestorerYolo()
shared.face_restorers.append(yolo)
