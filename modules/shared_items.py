def postprocessing_scripts():
    import modules.scripts
    return modules.scripts.scripts_postproc.scripts


def sd_vae_items():
    import modules.sd_vae
    return ["Automatic", "None"] + list(modules.sd_vae.vae_dict)


def refresh_vae_list():
    import modules.sd_vae
    modules.sd_vae.refresh_vae_list()


def sd_unet_items():
    import modules.sd_unet
    return ["None"] + list(modules.sd_unet.unet_dict)


def refresh_unet_list():
    import modules.sd_unet
    modules.sd_unet.refresh_unet_list()


def sd_te_items():
    import modules.model_te
    predefined = ['None', 'T5 FP4', 'T5 FP8', 'T5 INT8', 'T5 QINT8', 'T5 FP16']
    return predefined + list(modules.model_te.te_dict)


def refresh_te_list():
    import modules.model_te
    modules.model_te.refresh_te_list()


def list_crossattention(diffusers=False):
    if diffusers:
        return [
            "Disabled",
            "Scaled-Dot-Product",
            "xFormers",
            "Batch matrix-matrix",
            "Split attention",
            "Dynamic Attention BMM"
        ]
    else:
        return [
            "Disabled",
            "Scaled-Dot-Product",
            "xFormers",
            "Doggettx's",
            "InvokeAI's",
            "Sub-quadratic",
            "Split attention"
        ]

def get_pipelines():
    import diffusers
    from installer import log

    pipelines = { # note: not all pipelines can be used manually as they require prior pipeline next to decoder pipeline
        'Autodetect': None,
        'Stable Diffusion': getattr(diffusers, 'StableDiffusionPipeline', None),
        'Stable Diffusion 2': getattr(diffusers, 'StableDiffusionPipeline', None),
        'Stable Diffusion Inpaint': getattr(diffusers, 'StableDiffusionInpaintPipeline', None),
        'Stable Diffusion Img2Img': getattr(diffusers, 'StableDiffusionImg2ImgPipeline', None),
        'Stable Diffusion Instruct': getattr(diffusers, 'StableDiffusionInstructPix2PixPipeline', None),
        'Stable Diffusion Upscale': getattr(diffusers, 'StableDiffusionUpscalePipeline', None),
        'Stable Diffusion XL': getattr(diffusers, 'StableDiffusionXLPipeline', None),
        'Stable Diffusion XL Refiner': getattr(diffusers, 'StableDiffusionXLImg2ImgPipeline', None),
        'Stable Diffusion XL Img2Img': getattr(diffusers, 'StableDiffusionXLImg2ImgPipeline', None),
        'Stable Diffusion XL Inpaint': getattr(diffusers, 'StableDiffusionXLInpaintPipeline', None),
        'Stable Diffusion XL Instruct': getattr(diffusers, 'StableDiffusionXLInstructPix2PixPipeline', None),
        'Latent Consistency Model': getattr(diffusers, 'LatentConsistencyModelPipeline', None),
        'PixArt-Alpha': getattr(diffusers, 'PixArtAlphaPipeline', None),
        'UniDiffuser': getattr(diffusers, 'UniDiffuserPipeline', None),
        'Wuerstchen': getattr(diffusers, 'WuerstchenCombinedPipeline', None),
        'Kandinsky 2.1': getattr(diffusers, 'KandinskyPipeline', None),
        'Kandinsky 2.2': getattr(diffusers, 'KandinskyV22Pipeline', None),
        'Kandinsky 3': getattr(diffusers, 'Kandinsky3Pipeline', None),
        'DeepFloyd IF': getattr(diffusers, 'IFPipeline', None),
        'Custom Diffusers Pipeline': getattr(diffusers, 'DiffusionPipeline', None),
        'InstaFlow': getattr(diffusers, 'StableDiffusionPipeline', None), # dynamically redefined and loaded in sd_models.load_diffuser
        'SegMoE': getattr(diffusers, 'StableDiffusionPipeline', None), # dynamically redefined and loaded in sd_models.load_diffuser
        'Kolors': getattr(diffusers, 'KolorsPipeline', None),
        'AuraFlow': getattr(diffusers, 'AuraFlowPipeline', None),
        'CogView': getattr(diffusers, 'CogView3PlusPipeline', None),
        'Stable Cascade': getattr(diffusers, 'StableCascadeCombinedPipeline', None),
        'PixArt-Sigma': getattr(diffusers, 'PixArtSigmaPipeline', None),
        'HunyuanDiT': getattr(diffusers, 'HunyuanDiTPipeline', None),
        'Stable Diffusion 3': getattr(diffusers, 'StableDiffusion3Pipeline', None),
        'Stable Diffusion 3 Img2Img': getattr(diffusers, 'StableDiffusion3Img2ImgPipeline', None),
        'Lumina-Next': getattr(diffusers, 'LuminaText2ImgPipeline', None),
        'FLUX': getattr(diffusers, 'FluxPipeline', None),
        'Sana': getattr(diffusers, 'SanaPAGPipeline', None),
    }
    if hasattr(diffusers, 'OnnxStableDiffusionPipeline'):
        onnx_pipelines = {
            'ONNX Stable Diffusion': getattr(diffusers, 'OnnxStableDiffusionPipeline', None),
            'ONNX Stable Diffusion Img2Img': getattr(diffusers, 'OnnxStableDiffusionImg2ImgPipeline', None),
            'ONNX Stable Diffusion Inpaint': getattr(diffusers, 'OnnxStableDiffusionInpaintPipeline', None),
            'ONNX Stable Diffusion Upscale': getattr(diffusers, 'OnnxStableDiffusionUpscalePipeline', None),
        }
        pipelines.update(onnx_pipelines)
    if hasattr(diffusers, 'OnnxStableDiffusionXLPipeline'):
        onnx_pipelines = {
            'ONNX Stable Diffusion XL': getattr(diffusers, 'OnnxStableDiffusionXLPipeline', None),
            'ONNX Stable Diffusion XL Img2Img': getattr(diffusers, 'OnnxStableDiffusionXLImg2ImgPipeline', None),
        }
        pipelines.update(onnx_pipelines)

    # items that may rely on diffusers dev version
    """
    if hasattr(diffusers, 'FluxPipeline'):
        pipelines['FLUX'] = getattr(diffusers, 'FluxPipeline', None)
    """

    for k, v in pipelines.items():
        if k != 'Autodetect' and v is None:
            log.error(f'Not available: pipeline={k} diffusers={diffusers.__version__} path={diffusers.__file__}')
    return pipelines
