import torch
import diffusers
from PIL import Image
from modules import shared, devices
from modules.upscaler import Upscaler, UpscalerData
from installer import install

class UpscalerAuraSR(Upscaler):
    def __init__(self, dirname): # pylint: disable=super-init-not-called
        self.name = "AuraSR"
        self.user_path = dirname
        self.model = None
        if not shared.native:
            super().__init__()
            return
        self.scalers = [
            UpscalerData(name="Aura SR 4x", path="stabilityai/sd-x2-latent-upscaler", upscaler=self, model=None, scale=4),
        ]

    def callback(self, _step: int, _timestep: int, _latents: torch.FloatTensor):
        pass

    def do_upscale(self, img: Image.Image, selected_model):
        from modules.postprocess.aurasr_arch import AuraSR
        if self.model is None:
            self.model = AuraSR.from_pretrained("vladmandic/aurasr", use_safetensors=False)
        devices.torch_gc()

        self.model.upsampler.to(devices.device)
        image = self.model.upscale_4x(img)
        self.model.upsampler.to(devices.cpu)

        if shared.opts.upscaler_unload:
            self.model = None
            shared.log.debug(f"Upscaler unloaded: type={self.name} model={selected_model}")
            devices.torch_gc(force=True)
        return image
