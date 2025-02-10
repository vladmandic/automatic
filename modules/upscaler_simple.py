from PIL import Image
from modules.upscaler import Upscaler, UpscalerData


class UpscalerNone(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "None"
        self.scalers = [UpscalerData("None", None, self)]

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img


class UpscalerResize(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Resize"
        self.scalers = [
            UpscalerData("Resize Nearest", None, self),
            UpscalerData("Resize Lanczos", None, self),
            UpscalerData("Resize Bicubic", None, self),
            UpscalerData("Resize Bilinear", None, self),
            UpscalerData("Resize Hamming", None, self),
            UpscalerData("Resize Box", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        if selected_model is None:
            return img
        elif selected_model == "Resize Nearest":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.NEAREST)
        elif selected_model == "Resize Lanczos":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.LANCZOS)
        elif selected_model == "Resize Bicubic":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.BICUBIC)
        elif selected_model == "Resize Bilinear":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.BILINEAR)
        elif selected_model == "Resize Hamming":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.HAMMING)
        elif selected_model == "Resize Box":
            return img.resize((int(img.width * self.scale), int(img.height * self.scale)), resample=Image.Resampling.BOX)
        else:
            return img


    def load_model(self, _):
        pass


class UpscalerLatent(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Latent"
        self.scalers = [
            UpscalerData("Latent Nearest", None, self),
            UpscalerData("Latent Nearest exact", None, self),
            UpscalerData("Latent Area", None, self),
            UpscalerData("Latent Bilinear", None, self),
            UpscalerData("Latent Bicubic", None, self),
            UpscalerData("Latent Bilinear antialias", None, self),
            UpscalerData("Latent Bicubic antialias", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        import torch
        import torch.nn.functional as F
        if isinstance(img, torch.Tensor) and (len(img.shape) == 4):
            _batch, _channel, h, w = img.shape
        else:
            raise ValueError(f"Latent upscale: image={img.shape if isinstance(img, torch.Tensor) else img} type={type(img)} if not supported")
        h, w = int((8 * h * self.scale) // 8), int((8 * w * self.scale) // 8)
        mode, antialias = '', ''
        if selected_model == "Latent Nearest":
            mode, antialias = 'nearest', False
        elif selected_model == "Latent Nearest exact":
            mode, antialias = 'nearest-exact', False
        elif selected_model == "Latent Area":
            mode, antialias = 'area', False
        elif selected_model == "Latent Bilinear":
            mode, antialias = 'bilinear', False
        elif selected_model == "Latent Bicubic":
            mode, antialias = 'bicubic', False
        elif selected_model == "Latent Bilinear antialias":
            mode, antialias = 'bilinear', True
        elif selected_model == "Latent Bicubic antialias":
            mode, antialias = 'bicubic', True
        else:
            raise ValueError(f"Latent upscale: model={selected_model} unknown")
        return F.interpolate(img, size=(h, w), mode=mode, antialias=antialias)


class UpscalerAsymmetricVAE(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "Asymmetric VAE"
        self.vae = None
        self.scalers = [
            UpscalerData("Asymmetric VAE", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        import torchvision.transforms.functional as F
        import diffusers
        from modules import shared, devices

        if self.vae is None:
            self.vae = diffusers.AsymmetricAutoencoderKL.from_pretrained("Heasterian/AsymmetricAutoencoderKLUpscaler", cache_dir=shared.opts.hfcache_dir)
            self.vae.requires_grad_(False)
            self.vae = self.vae.to(device=devices.device, dtype=devices.dtype)
            self.vae.eval()
        img = img.resize((8 * (img.width // 8), 8 * (img.height // 8)), resample=Image.Resampling.BILINEAR).convert('RGB')
        tensor = (F.pil_to_tensor(img).unsqueeze(0) / 255.0).to(device=devices.device, dtype=devices.dtype)
        self.vae = self.vae.to(device=devices.device)
        tensor = self.vae(tensor).sample
        upscaled = F.to_pil_image(tensor.squeeze().clamp(0.0, 1.0).float().cpu())
        self.vae = self.vae.to(device=devices.cpu)
        return upscaled


class UpscalerDCC(Upscaler):
    def __init__(self, dirname=None): # pylint: disable=unused-argument
        super().__init__(False)
        self.name = "DCC Interpolation"
        self.vae = None
        self.scalers = [
            UpscalerData("DCC Interpolation", None, self),
        ]

    def do_upscale(self, img: Image, selected_model=None):
        import math
        import numpy as np
        from modules.postprocess.dcc import DCC
        normalized = np.array(img).astype(np.float32) / 255.0
        scale = math.ceil(self.scale)
        upscaled = DCC(normalized, scale)
        upscaled = (upscaled - upscaled.min()) / (upscaled.max() - upscaled.min())
        upscaled = (255.0 * upscaled).astype(np.uint8)
        upscaled = Image.fromarray(upscaled)
        return upscaled
