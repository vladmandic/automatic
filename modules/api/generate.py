from threading import Lock
from fastapi.responses import JSONResponse
from modules import errors, shared, scripts, ui
from modules.api import models, script, helpers
from modules.processing import StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, process_images


errors.install()


class APIGenerate():
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg_txt2img = []
        self.default_script_arg_img2img = []

    def sanitize_args(self, args: dict):
        args = vars(args)
        args.pop('include_init_images', None) # this is meant to be done by "exclude": True in model
        args.pop('script_name', None)
        args.pop('script_args', None) # will refeed them to the pipeline directly after initializing them
        args.pop('alwayson_scripts', None)
        args.pop('face', None)
        args.pop('face_id', None)
        args.pop('ip_adapter', None)
        args.pop('save_images', None)
        return args

    def sanitize_b64(self, request):
        def sanitize_str(args: list):
            for idx in range(0, len(args)):
                if isinstance(args[idx], str) and len(args[idx]) >= 1000:
                    args[idx] = f"<str {len(args[idx])}>"

        if hasattr(request, "alwayson_scripts") and request.alwayson_scripts:
            for script_name in request.alwayson_scripts.keys():
                script_obj = request.alwayson_scripts[script_name]
                if script_obj and "args" in script_obj and script_obj["args"]:
                    sanitize_str(script_obj["args"])
        if hasattr(request, "script_args") and request.script_args:
            sanitize_str(request.script_args)

    def prepare_face_module(self, request):
        if hasattr(request, "face") and request.face and not request.script_name and (not request.alwayson_scripts or "face" not in request.alwayson_scripts.keys()):
            request.script_name = "face"
            request.script_args = [
                request.face.mode,
                request.face.source_images,
                request.face.ip_model,
                request.face.ip_override_sampler,
                request.face.ip_cache_model,
                request.face.ip_strength,
                request.face.ip_structure,
                request.face.id_strength,
                request.face.id_conditioning,
                request.face.id_cache,
                request.face.pm_trigger,
                request.face.pm_strength,
                request.face.pm_start,
                request.face.fs_cache
            ]
            del request.face

    def post_text2img(self, txt2imgreq: models.ReqTxt2Img):
        self.prepare_face_module(txt2imgreq)
        script_runner = scripts.scripts_txt2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui(None)
        if not self.default_script_arg_txt2img:
            self.default_script_arg_txt2img = script.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = script.get_selectable_script(txt2imgreq.script_name, script_runner)
        populate = txt2imgreq.copy(update={  # Override __init__ params
            "sampler_name": helpers.validate_sampler_name(txt2imgreq.sampler_name or txt2imgreq.sampler_index),
            "do_not_save_samples": not txt2imgreq.save_images,
            "do_not_save_grid": not txt2imgreq.save_images,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on
        args = self.sanitize_args(populate)
        send_images = args.pop('send_images', True)
        with self.queue_lock:
            p = StableDiffusionProcessingTxt2Img(sd_model=shared.sd_model, **args)
            p.scripts = script_runner
            p.outpath_grids = shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids
            p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples
            shared.state.begin('api-txt2img', api=True)
            script_args = script.init_script_args(p, txt2imgreq, self.default_script_arg_txt2img, selectable_scripts, selectable_script_idx, script_runner)
            if selectable_scripts is not None:
                processed = scripts.scripts_txt2img.run(p, *script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end(api=False)
        b64images = list(map(helpers.encode_pil_to_base64, processed.images)) if send_images else []
        self.sanitize_b64(txt2imgreq)
        return models.ResTxt2Img(images=b64images, parameters=vars(txt2imgreq), info=processed.js())

    def post_img2img(self, img2imgreq: models.ReqImg2Img):
        self.prepare_face_module(img2imgreq)
        init_images = img2imgreq.init_images
        if init_images is None:
            return JSONResponse(status_code=400, content={"error": "Init image is none"})
        mask = img2imgreq.mask
        if mask:
            mask = helpers.decode_base64_to_image(mask)
        script_runner = scripts.scripts_img2img
        if not script_runner.scripts:
            script_runner.initialize_scripts(True)
            ui.create_ui(None)
        if not self.default_script_arg_img2img:
            self.default_script_arg_img2img = script.init_default_script_args(script_runner)
        selectable_scripts, selectable_script_idx = script.get_selectable_script(img2imgreq.script_name, script_runner)
        populate = img2imgreq.copy(update={  # Override __init__ params
            "sampler_name": helpers.validate_sampler_name(img2imgreq.sampler_name or img2imgreq.sampler_index),
            "do_not_save_samples": not img2imgreq.save_images,
            "do_not_save_grid": not img2imgreq.save_images,
            "mask": mask,
        })
        if populate.sampler_name:
            populate.sampler_index = None  # prevent a warning later on
        args = self.sanitize_args(populate)
        send_images = args.pop('send_images', True)
        with self.queue_lock:
            p = StableDiffusionProcessingImg2Img(sd_model=shared.sd_model, **args)
            p.init_images = [helpers.decode_base64_to_image(x) for x in init_images]
            p.scripts = script_runner
            p.outpath_grids = shared.opts.outdir_img2img_grids
            p.outpath_samples = shared.opts.outdir_img2img_samples
            shared.state.begin('api-img2img', api=True)
            script_args = script.init_script_args(p, img2imgreq, self.default_script_arg_img2img, selectable_scripts, selectable_script_idx, script_runner)
            if selectable_scripts is not None:
                processed = scripts.scripts_img2img.run(p, *script_args) # Need to pass args as list here
            else:
                p.script_args = tuple(script_args) # Need to pass args as tuple here
                processed = process_images(p)
            shared.state.end(api=False)
        b64images = list(map(helpers.encode_pil_to_base64, processed.images)) if send_images else []
        if not img2imgreq.include_init_images:
            img2imgreq.init_images = None
            img2imgreq.mask = None
        self.sanitize_b64(img2imgreq)
        return models.ResImg2Img(images=b64images, parameters=vars(img2imgreq), info=processed.js())
