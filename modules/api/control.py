from typing import Optional, List
from threading import Lock
from pydantic import BaseModel, Field # pylint: disable=no-name-in-module
from modules import errors, shared, scripts, ui
from modules.api import script, helpers
from modules.processing import StableDiffusionProcessingControl
from modules.control import run as run_control

# TODO control api
# should use control.run, not process_images directly

errors.install()


class ReqControl(BaseModel):
    pass

class ResControl(BaseModel):
    images: List[str] = Field(default=None, title="Image", description="The generated images in base64 format.")
    params: dict = Field(default={}, title="Settings", description="Process settings")
    info: str = Field(default="", title="Info", description="Process info")


class APIControl():
    def __init__(self, queue_lock: Lock):
        self.queue_lock = queue_lock
        self.default_script_arg = []

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

    def post_control(self, req: ReqControl):
        self.prepare_face_module(req)

        # prepare script
        script_runner = scripts.scripts_control
        if not script_runner.scripts:
            script_runner.initialize_scripts(False)
            ui.create_ui(None)
        if not self.default_script_arg:
            self.default_script_arg = script.init_default_script_args(script_runner)

        # prepare args
        args = req.copy(update={  # Override __init__ params
            "sampler_name": helpers.validate_sampler_name(req.sampler_name or req.sampler_index),
            "sampler_index": None,
            "do_not_save_samples": not req.save_images,
            "do_not_save_grid": not req.save_images,
            "init_images": [helpers.decode_base64_to_image(x) for x in req.init_images] if req.init_images else None,
            "mask": helpers.decode_base64_to_image(req.mask) if req.mask else None,
        })
        args = self.sanitize_args(args)
        send_images = args.pop('send_images', True)

        # run
        with self.queue_lock:
            shared.state.begin('api-control', api=True)

            # selectable_scripts, selectable_script_idx = script.get_selectable_script(req.script_name, script_runner)
            # script_args = script.init_script_args(p, req, self.default_script_arg, selectable_scripts, selectable_script_idx, script_runner)
            # output_images, _processed_images, output_info = run_control(**args, **script_args)
            output_images = None
            output_info = None

            shared.state.end(api=False)

        # return
        b64images = list(map(helpers.encode_pil_to_base64, output_images)) if send_images else []
        self.sanitize_b64(req)
        return ResControl(images=b64images, params=vars(req), info=output_info)
