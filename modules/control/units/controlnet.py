import os
import time
from typing import Union
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, FluxPipeline, StableDiffusion3Pipeline, ControlNetModel
from modules.control.units import detect
from modules.shared import log, opts, listdir
from modules import errors, sd_models, devices, model_quant


what = 'ControlNet'
debug = log.trace if os.environ.get('SD_CONTROL_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: CONTROL')
predefined_sd15 = {
    'Canny': "lllyasviel/control_v11p_sd15_canny",
    'Depth': "lllyasviel/control_v11f1p_sd15_depth",
    'HED': "lllyasviel/sd-controlnet-hed",
    'IP2P': "lllyasviel/control_v11e_sd15_ip2p",
    'LineArt': "lllyasviel/control_v11p_sd15_lineart",
    'LineArt Anime': "lllyasviel/control_v11p_sd15s2_lineart_anime",
    'MLDS': "lllyasviel/control_v11p_sd15_mlsd",
    'NormalBae': "lllyasviel/control_v11p_sd15_normalbae",
    'OpenPose': "lllyasviel/control_v11p_sd15_openpose",
    'Scribble': "lllyasviel/control_v11p_sd15_scribble",
    'Segment': "lllyasviel/control_v11p_sd15_seg",
    'Shuffle': "lllyasviel/control_v11e_sd15_shuffle",
    'SoftEdge': "lllyasviel/control_v11p_sd15_softedge",
    'Tile': "lllyasviel/control_v11f1e_sd15_tile",
    'Depth Anything': 'vladmandic/depth-anything',
    'Canny FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_canny.safetensors',
    'Inpaint FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_inpaint.safetensors',
    'LineArt Anime FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_animeline.safetensors',
    'LineArt FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_lineart.safetensors',
    'MLSD FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_mlsd.safetensors',
    'NormalBae FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_normal.safetensors',
    'OpenPose FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_openpose.safetensors',
    'Pix2Pix FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_pix2pix.safetensors',
    'Scribble FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_scribble.safetensors',
    'Segment FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_seg.safetensors',
    'Shuffle FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_shuffle.safetensors',
    'SoftEdge FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_softedge.safetensors',
    'Tile FP16': 'Aptronym/SDNext/ControlNet11/controlnet11Models_tileE.safetensors',
    'CiaraRowles TemporalNet': "CiaraRowles/TemporalNet",
    'Ciaochaos Recolor': 'ioclab/control_v1p_sd15_brightness',
    'Ciaochaos Illumination': 'ioclab/control_v1u_sd15_illumination/illumination20000.safetensors',
}
predefined_sdxl = {
    'Canny Small XL': 'diffusers/controlnet-canny-sdxl-1.0-small',
    'Canny Mid XL': 'diffusers/controlnet-canny-sdxl-1.0-mid',
    'Canny XL': 'diffusers/controlnet-canny-sdxl-1.0',
    'Depth Zoe XL': 'diffusers/controlnet-zoe-depth-sdxl-1.0',
    'Depth Mid XL': 'diffusers/controlnet-depth-sdxl-1.0-mid',
    'OpenPose XL': 'thibaud/controlnet-openpose-sdxl-1.0/bin',
    'Xinsir Union XL': 'xinsir/controlnet-union-sdxl-1.0',
    'Xinsir OpenPose XL': 'xinsir/controlnet-openpose-sdxl-1.0',
    'Xinsir Canny XL': 'xinsir/controlnet-canny-sdxl-1.0',
    'Xinsir Depth XL': 'xinsir/controlnet-depth-sdxl-1.0',
    'Xinsir Scribble XL': 'xinsir/controlnet-scribble-sdxl-1.0',
    'Xinsir Anime Painter XL': 'xinsir/anime-painter',
    'NoobAI Canny XL': 'Eugeoter/noob-sdxl-controlnet-canny',
    'NoobAI Lineart Anime XL': 'Eugeoter/noob-sdxl-controlnet-lineart_anime',
    'NoobAI Depth XL': 'Eugeoter/noob-sdxl-controlnet-depth',
    'NoobAI Normal XL': 'Eugeoter/noob-sdxl-controlnet-normal',
    'NoobAI SoftEdge XL': 'Eugeoter/noob-sdxl-controlnet-softedge_hed',
    'NoobAI OpenPose XL': 'Laxhar/noob_openpose/openpose_pre.safetensors',
    # 'StabilityAI Canny R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-canny-rank128.safetensors',
    # 'StabilityAI Depth R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-depth-rank128.safetensors',
    # 'StabilityAI Recolor R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-recolor-rank128.safetensors',
    # 'StabilityAI Sketch R128': 'stabilityai/control-lora/control-LoRAs-rank128/control-lora-sketch-rank128-metadata.safetensors',
    # 'StabilityAI Canny R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-canny-rank256.safetensors',
    # 'StabilityAI Depth R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-depth-rank256.safetensors',
    # 'StabilityAI Recolor R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors',
    # 'StabilityAI Sketch R256': 'stabilityai/control-lora/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors',
}
predefined_f1 = {
    "InstantX Union": 'InstantX/FLUX.1-dev-Controlnet-Union',
    "InstantX Canny": 'InstantX/FLUX.1-dev-Controlnet-Canny',
    "JasperAI Depth": 'jasperai/Flux.1-dev-Controlnet-Depth',
    "JasperAI Surface Normals": 'jasperai/Flux.1-dev-Controlnet-Surface-Normals',
    "JasperAI Upscaler": 'jasperai/Flux.1-dev-Controlnet-Upscaler',
    "Shakker-Labs Union": 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro',
    "Shakker-Labs Pose": 'Shakker-Labs/FLUX.1-dev-ControlNet-Pose',
    "Shakker-Labs Depth": 'Shakker-Labs/FLUX.1-dev-ControlNet-Depth',
    "XLabs-AI Canny": 'XLabs-AI/flux-controlnet-canny-diffusers',
    "XLabs-AI Depth": 'XLabs-AI/flux-controlnet-depth-diffusers',
    "XLabs-AI HED": 'XLabs-AI/flux-controlnet-hed-diffusers'
}
predefined_sd3 = {
    "InstantX Canny": 'InstantX/SD3-Controlnet-Canny',
    "InstantX Pose": 'InstantX/SD3-Controlnet-Pose',
    "InstantX Depth": 'InstantX/SD3-Controlnet-Depth',
    "InstantX Tile": 'InstantX/SD3-Controlnet-Tile',
    "Alimama Inpainting": 'alimama-creative/SD3-Controlnet-Inpainting',
    "Alimama SoftEdge": 'alimama-creative/SD3-Controlnet-Softedge',
}
models = {}
all_models = {}
all_models.update(predefined_sd15)
all_models.update(predefined_sdxl)
all_models.update(predefined_f1)
all_models.update(predefined_sd3)
cache_dir = 'models/control/controlnet'


def find_models():
    path = os.path.join(opts.control_dir, 'controlnet')
    files = listdir(path)
    folders = [f for f in files if os.path.isdir(f) if os.path.exists(os.path.join(f, 'config.json'))]
    files = [f for f in files if f.endswith('.safetensors')]
    downloaded_models = {}
    for f in files:
        basename = os.path.splitext(os.path.relpath(f, path))[0]
        downloaded_models[basename] = f
    for f in folders:
        basename = os.path.relpath(f, path)
        downloaded_models[basename] = f
    all_models.update(downloaded_models)
    return downloaded_models

find_models()

def list_models(refresh=False):
    import modules.shared
    global models # pylint: disable=global-statement
    if not refresh and len(models) > 0:
        return models
    models = {}
    if modules.shared.sd_model_type == 'none':
        models = ['None']
    elif modules.shared.sd_model_type == 'sdxl':
        models = ['None'] + list(predefined_sdxl) + sorted(find_models())
    elif modules.shared.sd_model_type == 'sd':
        models = ['None'] + list(predefined_sd15) + sorted(find_models())
    elif modules.shared.sd_model_type == 'f1':
        models = ['None'] + list(predefined_f1) + sorted(find_models())
    elif modules.shared.sd_model_type == 'sd3':
        models = ['None'] + list(predefined_sd3) + sorted(find_models())
    else:
        log.warning(f'Control {what} model list failed: unknown model type')
        models = ['None'] + sorted(predefined_sd15) + sorted(predefined_sdxl) + sorted(predefined_f1) + sorted(predefined_sd3) + sorted(find_models())
    debug(f'Control list {what}: path={cache_dir} models={models}')
    return models


class ControlNet():
    def __init__(self, model_id: str = None, device = None, dtype = None, load_config = None):
        self.model: ControlNetModel = None
        self.model_id: str = model_id
        self.device = device
        self.dtype = dtype
        self.load_config = { 'cache_dir': cache_dir }
        if load_config is not None:
            self.load_config.update(load_config)
        if model_id is not None:
            self.load()

    def reset(self):
        if self.model is not None:
            debug(f'Control {what} model unloaded')
        self.model = None
        self.model_id = None

    def get_class(self):
        import modules.shared
        if modules.shared.sd_model_type == 'sd':
            from diffusers import ControlNetModel as model_class # pylint: disable=reimported
        elif modules.shared.sd_model_type == 'sdxl':
            from diffusers import ControlNetModel as model_class # pylint: disable=reimported # sdxl shares same model class
        elif modules.shared.sd_model_type == 'f1':
            from diffusers import FluxControlNetModel as model_class
        elif modules.shared.sd_model_type == 'sd3':
            from diffusers import SD3ControlNetModel as model_class
        else:
            log.error(f'Control {what}: type={modules.shared.sd_model_type} unsupported model')
            return None
        return model_class

    def load_safetensors(self, model_path):
        name = os.path.splitext(model_path)[0]
        config_path = None
        if not os.path.exists(model_path):
            import huggingface_hub as hf
            parts = model_path.split('/')
            repo_id = f'{parts[0]}/{parts[1]}'
            filename = os.path.splitext('/'.join(parts[2:]))[0]
            model_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.safetensors', cache_dir=cache_dir)
            if config_path is None:
                try:
                    config_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.yaml', cache_dir=cache_dir)
                except Exception:
                    pass # no yaml file
            if config_path is None:
                try:
                    config_path = hf.hf_hub_download(repo_id=repo_id, filename=f'{filename}.json', cache_dir=cache_dir)
                except Exception:
                    pass # no yaml file
        elif os.path.exists(name + '.yaml'):
            config_path = f'{name}.yaml'
        elif os.path.exists(name + '.json'):
            config_path = f'{name}.json'
        if config_path is not None:
            self.load_config['original_config_file '] = config_path
        cls = self.get_class()
        self.model = cls.from_single_file(model_path, **self.load_config)

    def load(self, model_id: str = None, force: bool = True) -> str:
        try:
            t0 = time.time()
            model_id = model_id or self.model_id
            if model_id is None or model_id == 'None':
                self.reset()
                return
            if model_id not in all_models:
                log.error(f'Control {what} unknown model: id="{model_id}" available={list(all_models)}')
                return
            model_path = all_models[model_id]
            if model_path == '':
                return
            if model_path is None:
                log.error(f'Control {what} model load failed: id="{model_id}" error=unknown model id')
                return
            if model_id == self.model_id and not force:
                log.debug(f'Control {what} model: id="{model_id}" path="{model_path}" already loaded')
                return
            log.debug(f'Control {what} model loading: id="{model_id}" path="{model_path}"')
            if model_path.endswith('.safetensors'):
                self.load_safetensors(model_path)
            else:
                if '/bin' in model_path:
                    model_path = model_path.replace('/bin', '')
                    self.load_config['use_safetensors'] = False
                cls = self.get_class()
                if cls is None:
                    log.error(f'Control {what} model load failed: id="{model_id}" unknown base model')
                    return
                self.model = cls.from_pretrained(model_path, **self.load_config)
            if self.dtype is not None:
                self.model.to(self.dtype)
            if "ControlNet" in opts.nncf_compress_weights:
                try:
                    log.debug(f'Control {what} model NNCF Compress: id="{model_id}"')
                    from installer import install
                    install('nncf==2.7.0', quiet=True)
                    from modules.sd_models_compile import nncf_compress_model
                    self.model = nncf_compress_model(self.model)
                except Exception as e:
                    log.error(f'Control {what} model NNCF Compression failed: id="{model_id}" error={e}')
            elif "ControlNet" in opts.optimum_quanto_weights:
                try:
                    log.debug(f'Control {what} model Optimum Quanto: id="{model_id}"')
                    model_quant.load_quanto('Load model: type=ControlNet')
                    from modules.sd_models_compile import optimum_quanto_model
                    self.model = optimum_quanto_model(self.model)
                except Exception as e:
                    log.error(f'Control {what} model Optimum Quanto failed: id="{model_id}" error={e}')
            if self.device is not None:
                self.model.to(self.device)
            t1 = time.time()
            self.model_id = model_id
            log.debug(f'Control {what} model loaded: id="{model_id}" path="{model_path}" time={t1-t0:.2f}')
            return f'{what} loaded model: {model_id}'
        except Exception as e:
            log.error(f'Control {what} model load failed: id="{model_id}" error={e}')
            errors.display(e, f'Control {what} load')
            return f'{what} failed to load model: {model_id}'


class ControlNetPipeline():
    def __init__(self,
                 controlnet: Union[ControlNetModel, list[ControlNetModel]],
                 pipeline: Union[StableDiffusionXLPipeline, StableDiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline],
                 dtype = None,
                ):
        t0 = time.time()
        self.orig_pipeline = pipeline
        self.pipeline = None
        if pipeline is None:
            log.error('Control model pipeline: model not loaded')
            return
        elif detect.is_sdxl(pipeline):
            from diffusers import StableDiffusionXLControlNetPipeline
            self.pipeline = StableDiffusionXLControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                controlnet=controlnet, # can be a list
            )
        elif detect.is_sd15(pipeline):
            from diffusers import StableDiffusionControlNetPipeline
            self.pipeline = StableDiffusionControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                tokenizer=pipeline.tokenizer,
                unet=pipeline.unet,
                scheduler=pipeline.scheduler,
                feature_extractor=getattr(pipeline, 'feature_extractor', None),
                requires_safety_checker=False,
                safety_checker=None,
                controlnet=controlnet, # can be a list
            )
            sd_models.move_model(self.pipeline, pipeline.device)
        elif detect.is_f1(pipeline):
            from diffusers import FluxControlNetPipeline
            self.pipeline = FluxControlNetPipeline(
                vae=pipeline.vae.to(devices.device),
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                transformer=pipeline.transformer,
                scheduler=pipeline.scheduler,
                controlnet=controlnet, # can be a list
            )
        elif detect.is_sd3(pipeline):
            from diffusers import StableDiffusion3ControlNetPipeline
            self.pipeline = StableDiffusion3ControlNetPipeline(
                vae=pipeline.vae,
                text_encoder=pipeline.text_encoder,
                text_encoder_2=pipeline.text_encoder_2,
                text_encoder_3=pipeline.text_encoder_3,
                tokenizer=pipeline.tokenizer,
                tokenizer_2=pipeline.tokenizer_2,
                tokenizer_3=pipeline.tokenizer_3,
                transformer=pipeline.transformer,
                scheduler=pipeline.scheduler,
                controlnet=controlnet, # can be a list
            )
        else:
            log.error(f'Control {what} pipeline: class={pipeline.__class__.__name__} unsupported model type')
            return

        if self.pipeline is None:
            log.error(f'Control {what} pipeline: not initialized')
            return
        if dtype is not None:
            self.pipeline = self.pipeline.to(dtype)
        if opts.diffusers_offload_mode == 'none':
            sd_models.move_model(self.pipeline, devices.device)
        from modules.sd_models import set_diffuser_offload
        set_diffuser_offload(self.pipeline, 'model')

        t1 = time.time()
        log.debug(f'Control {what} pipeline: class={self.pipeline.__class__.__name__} time={t1-t0:.2f}')

    def restore(self):
        self.pipeline = None
        return self.orig_pipeline
