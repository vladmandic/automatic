from modules import scripts, processing, shared, devices


class Script(scripts.Script):
    standalone = False

    def title(self):
        return 'Init Latents'

    def show(self, is_img2img):
        return scripts.AlwaysVisible if shared.native else False

    @staticmethod
    def get_latents(p):
        import torch
        from diffusers.utils.torch_utils import randn_tensor
        generator_device = devices.cpu if shared.opts.diffusers_generator_device == "CPU" else shared.device
        generator = [torch.Generator(generator_device).manual_seed(s) for s in p.seeds]
        shape = (len(generator), shared.sd_model.unet.config.in_channels, p.height // shared.sd_model.vae_scale_factor, p.width // shared.sd_model.vae_scale_factor)
        latents = randn_tensor(shape, generator=generator, device=shared.sd_model._execution_device, dtype=shared.sd_model.unet.dtype) # pylint: disable=protected-access
        var_generator = [torch.Generator(generator_device).manual_seed(ss) for ss in p.subseeds]
        var_latents = randn_tensor(shape, generator=var_generator, device=shared.sd_model._execution_device, dtype=shared.sd_model.unet.dtype) # pylint: disable=protected-access
        return latents, var_latents, generator, var_generator

    @staticmethod
    def set_slerp(p, latents, var_latents, generator, var_generator):
        from modules.processing_helpers import slerp
        p.init_latent = slerp(p.subseed_strength, latents, var_latents) if p.subseed_strength < 1 else var_latents
        p.generator = generator if p.subseed_strength <= 0.5 else var_generator


    def process_batch(self, p: processing.StableDiffusionProcessing, *args, **kwargs): # pylint: disable=arguments-differ
        from modules.processing_helpers import create_random_tensors
        if not shared.native:
            return
        args = list(args)
        if p.subseed_strength != 0 and getattr(shared.sd_model, '_execution_device', None) is not None:
            # alt method using slerp
            # latents, var_latents, generator, var_generator = self.get_latents(p)
            # self.set_slerp(p, latents, var_latents, generator, var_generator)
            p.init_latent = create_random_tensors(
                shape=[shared.sd_model.unet.config.in_channels, p.height // shared.sd_model.vae_scale_factor, p.width // shared.sd_model.vae_scale_factor],
                seeds=p.seeds,
                subseeds=p.subseeds,
                subseed_strength=p.subseed_strength,
                p=p
            )
            p.init_latent = p.init_latent.to(device=shared.sd_model._execution_device, dtype=shared.sd_model.unet.dtype) # pylint: disable=protected-access
