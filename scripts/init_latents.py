import torch
from diffusers.utils.torch_utils import randn_tensor
from modules import scripts, processing, shared, devices
from modules.processing_helpers import slerp


class Script(scripts.Script):
    standalone = False

    def title(self):
        return 'Init Latents'

    def show(self, is_img2img):
        return scripts.AlwaysVisible if shared.backend == shared.Backend.DIFFUSERS else False

    @staticmethod
    def get_latents(p):
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
        generator = [torch.Generator(generator_device).manual_seed(s) for s in p.seeds]
        shape = (len(generator), shared.sd_model.unet.config.in_channels, p.height // shared.sd_model.vae_scale_factor,
                 p.width // shared.sd_model.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=shared.sd_model._execution_device, dtype=shared.sd_model.unet.dtype) # pylint: disable=protected-access
        var_generator = [torch.Generator(generator_device).manual_seed(ss) for ss in p.subseeds]
        var_latents = randn_tensor(shape, generator=var_generator, device=shared.sd_model._execution_device, dtype=shared.sd_model.unet.dtype) # pylint: disable=protected-access
        return latents, var_latents, generator, var_generator

    @staticmethod
    def set_slerp(p, latents, var_latents, generator, var_generator):
        if p.subseed_strength < 1:
            p.init_latent = slerp(p.subseed_strength, latents, var_latents)
        if p.subseed_strength == 1:
            p.init_latent = var_latents
        if 0 < p.subseed_strength <= 0.5:
            p.generator = generator
        if 0.5 < p.subseed_strength <= 1:
            p.generator = var_generator


    def process_batch(self, p: processing.StableDiffusionProcessing, *args, **kwargs): # pylint: disable=arguments-differ
        if shared.backend != shared.Backend.DIFFUSERS:
            return
        args = list(args)
        if p.subseed_strength != 0 and getattr(shared.sd_model, '_execution_device', None) is not None:
            latents, var_latents, generator, var_generator = self.get_latents(p)
            self.set_slerp(p, latents, var_latents, generator, var_generator)
