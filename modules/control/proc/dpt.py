from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, DPTForDepthEstimation
from modules import devices
from modules.shared import opts


image_processor: AutoImageProcessor = None


class DPTDetector:
    def __init__(self, model=None, processor=None, model_path=None):
        self.model = model
        self.processor = processor
        self.model_path = model_path or "Intel/dpt-large"

    def __call__(self, input_image=None, model_path=None):
        from modules.control.processors import cache_dir
        if model_path is not None and model_path != self.model_path:
            self.model_path = model_path
            self.processor = None
            self.model = None
        if self.processor is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_path, cache_dir=cache_dir)
        if self.model is None:
            self.model = DPTForDepthEstimation.from_pretrained(self.model_path, cache_dir=cache_dir)

        self.model.to(devices.device)
        with devices.inference_context():
            inputs = self.processor(images=input_image, return_tensors="pt")
            inputs.to(devices.device)
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=input_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
        if opts.control_move_processor:
            self.model.to('cpu')
        depth = Image.fromarray(formatted)
        depth = depth.convert('RGB')
        return depth
