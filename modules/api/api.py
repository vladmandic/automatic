from typing import List, Optional
from threading import Lock
from secrets import compare_digest
from fastapi import FastAPI, APIRouter, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.exceptions import HTTPException
from modules import errors, shared, postprocessing
from modules.api import models, endpoints, script, helpers, server, nvml, generate, process, control, gallery, docs


errors.install()


class Api:
    def __init__(self, app: FastAPI, queue_lock: Lock):
        self.credentials = {}
        if shared.cmd_opts.auth:
            for auth in shared.cmd_opts.auth.split(","):
                user, password = auth.split(":")
                self.credentials[user.replace('"', '').strip()] = password.replace('"', '').strip()
        if shared.cmd_opts.auth_file:
            with open(shared.cmd_opts.auth_file, 'r', encoding="utf8") as file:
                for line in file.readlines():
                    user, password = line.split(":")
                    self.credentials[user.replace('"', '').strip()] = password.replace('"', '').strip()
        self.router = APIRouter()
        if shared.cmd_opts.docs:
            docs.create_docs(app)
            docs.create_redocs(app)
        self.app = app
        self.queue_lock = queue_lock
        self.generate = generate.APIGenerate(queue_lock)
        self.process = process.APIProcess(queue_lock)
        self.control = control.APIControl(queue_lock)

        # server api
        self.add_api_route("/sdapi/v1/motd", server.get_motd, methods=["GET"], response_model=str)
        self.add_api_route("/sdapi/v1/log", server.get_log, methods=["GET"], response_model=List[str])
        self.add_api_route("/sdapi/v1/log", server.post_log, methods=["POST"])
        self.add_api_route("/sdapi/v1/start", self.get_session_start, methods=["GET"])
        self.add_api_route("/sdapi/v1/version", server.get_version, methods=["GET"])
        self.add_api_route("/sdapi/v1/status", server.get_status, methods=["GET"], response_model=models.ResStatus)
        self.add_api_route("/sdapi/v1/platform", server.get_platform, methods=["GET"])
        self.add_api_route("/sdapi/v1/progress", server.get_progress, methods=["GET"], response_model=models.ResProgress)
        self.add_api_route("/sdapi/v1/interrupt", server.post_interrupt, methods=["POST"])
        self.add_api_route("/sdapi/v1/skip", server.post_skip, methods=["POST"])
        self.add_api_route("/sdapi/v1/shutdown", server.post_shutdown, methods=["POST"])
        self.add_api_route("/sdapi/v1/memory", server.get_memory, methods=["GET"], response_model=models.ResMemory)
        self.add_api_route("/sdapi/v1/options", server.get_config, methods=["GET"], response_model=models.OptionsModel)
        self.add_api_route("/sdapi/v1/options", server.set_config, methods=["POST"])
        self.add_api_route("/sdapi/v1/cmd-flags", server.get_cmd_flags, methods=["GET"], response_model=models.FlagsModel)
        self.add_api_route("/sdapi/v1/nvml", nvml.get_nvml, methods=["GET"], response_model=List[models.ResNVML])

        # core api using locking
        self.add_api_route("/sdapi/v1/txt2img", self.generate.post_text2img, methods=["POST"], response_model=models.ResTxt2Img)
        self.add_api_route("/sdapi/v1/img2img", self.generate.post_img2img, methods=["POST"], response_model=models.ResImg2Img)
        self.add_api_route("/sdapi/v1/control", self.control.post_control, methods=["POST"], response_model=control.ResControl)
        self.add_api_route("/sdapi/v1/extra-single-image", self.extras_single_image_api, methods=["POST"], response_model=models.ResProcessImage)
        self.add_api_route("/sdapi/v1/extra-batch-images", self.extras_batch_images_api, methods=["POST"], response_model=models.ResProcessBatch)
        self.add_api_route("/sdapi/v1/preprocess", self.process.post_preprocess, methods=["POST"])
        self.add_api_route("/sdapi/v1/mask", self.process.post_mask, methods=["POST"])
        self.add_api_route("/sdapi/v1/detect", self.process.post_detect, methods=["POST"])

        # api dealing with optional scripts
        self.add_api_route("/sdapi/v1/scripts", script.get_scripts_list, methods=["GET"], response_model=models.ResScripts)
        self.add_api_route("/sdapi/v1/script-info", script.get_script_info, methods=["GET"], response_model=List[models.ItemScript])

        # enumerator api
        self.add_api_route("/sdapi/v1/preprocessors", self.process.get_preprocess, methods=["GET"], response_model=List[process.ItemPreprocess])
        self.add_api_route("/sdapi/v1/masking", self.process.get_mask, methods=["GET"], response_model=process.ItemMask)
        self.add_api_route("/sdapi/v1/interrogate", endpoints.get_interrogate, methods=["GET"], response_model=List[str])
        self.add_api_route("/sdapi/v1/samplers", endpoints.get_samplers, methods=["GET"], response_model=List[models.ItemSampler])
        self.add_api_route("/sdapi/v1/upscalers", endpoints.get_upscalers, methods=["GET"], response_model=List[models.ItemUpscaler])
        self.add_api_route("/sdapi/v1/sd-models", endpoints.get_sd_models, methods=["GET"], response_model=List[models.ItemModel])
        self.add_api_route("/sdapi/v1/hypernetworks", endpoints.get_hypernetworks, methods=["GET"], response_model=List[models.ItemHypernetwork])
        self.add_api_route("/sdapi/v1/face-restorers", endpoints.get_detailers, methods=["GET"], response_model=List[models.ItemDetailer])
        self.add_api_route("/sdapi/v1/prompt-styles", endpoints.get_prompt_styles, methods=["GET"], response_model=List[models.ItemStyle])
        self.add_api_route("/sdapi/v1/embeddings", endpoints.get_embeddings, methods=["GET"], response_model=models.ResEmbeddings)
        self.add_api_route("/sdapi/v1/sd-vae", endpoints.get_sd_vaes, methods=["GET"], response_model=List[models.ItemVae])
        self.add_api_route("/sdapi/v1/extensions", endpoints.get_extensions_list, methods=["GET"], response_model=List[models.ItemExtension])
        self.add_api_route("/sdapi/v1/extra-networks", endpoints.get_extra_networks, methods=["GET"], response_model=List[models.ItemExtraNetwork])

        # functional api
        self.add_api_route("/sdapi/v1/png-info", endpoints.post_pnginfo, methods=["POST"], response_model=models.ResImageInfo)
        self.add_api_route("/sdapi/v1/interrogate", endpoints.post_interrogate, methods=["POST"])
        self.add_api_route("/sdapi/v1/vqa", endpoints.post_vqa, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-checkpoints", endpoints.post_refresh_checkpoints, methods=["POST"])
        self.add_api_route("/sdapi/v1/unload-checkpoint", endpoints.post_unload_checkpoint, methods=["POST"])
        self.add_api_route("/sdapi/v1/reload-checkpoint", endpoints.post_reload_checkpoint, methods=["POST"])
        self.add_api_route("/sdapi/v1/refresh-vae", endpoints.post_refresh_vae, methods=["POST"])
        self.add_api_route("/sdapi/v1/history", endpoints.get_history, methods=["GET"], response_model=List[str])
        self.add_api_route("/sdapi/v1/history", endpoints.post_history, methods=["POST"], response_model=int)

        # lora api
        if shared.native:
            self.add_api_route("/sdapi/v1/loras", endpoints.get_loras, methods=["GET"], response_model=List[dict])
            self.add_api_route("/sdapi/v1/refresh-loras", endpoints.post_refresh_loras, methods=["POST"])

        # gallery api
        gallery.register_api(app)

    def add_api_route(self, path: str, endpoint, **kwargs):
        if (shared.cmd_opts.auth or shared.cmd_opts.auth_file) and shared.cmd_opts.api_only:
            return self.app.add_api_route(path, endpoint, dependencies=[Depends(self.auth)], **kwargs)
        return self.app.add_api_route(path, endpoint, **kwargs)

    def auth(self, credentials: HTTPBasicCredentials = Depends(HTTPBasic())):
        # this is only needed for api-only since otherwise auth is handled in gradio/routes.py
        if credentials.username in self.credentials:
            if compare_digest(credentials.password, self.credentials[credentials.username]):
                return True
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})

    def get_session_start(self, req: Request, agent: Optional[str] = None):
        token = req.cookies.get("access-token") or req.cookies.get("access-token-unsecure")
        user = self.app.tokens.get(token) if hasattr(self.app, 'tokens') else None
        shared.log.info(f'Browser session: user={user} client={req.client.host} agent={agent}')
        return {}

    def set_upscalers(self, req: dict):
        reqDict = vars(req)
        reqDict['extras_upscaler_1'] = reqDict.pop('upscaler_1', None)
        reqDict['extras_upscaler_2'] = reqDict.pop('upscaler_2', None)
        return reqDict

    def extras_single_image_api(self, req: models.ReqProcessImage):
        reqDict = self.set_upscalers(req)
        reqDict['image'] = helpers.decode_base64_to_image(reqDict['image'])
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=0, image_folder="", input_dir="", output_dir="", save_output=False, **reqDict)
        return models.ResProcessImage(image=helpers.encode_pil_to_base64(result[0][0]), html_info=result[1])

    def extras_batch_images_api(self, req: models.ReqProcessBatch):
        reqDict = self.set_upscalers(req)
        image_list = reqDict.pop('imageList', [])
        image_folder = [helpers.decode_base64_to_image(x.data) for x in image_list]
        with self.queue_lock:
            result = postprocessing.run_extras(extras_mode=1, image_folder=image_folder, image="", input_dir="", output_dir="", save_output=False, **reqDict)
        return models.ResProcessBatch(images=list(map(helpers.encode_pil_to_base64, result[0])), html_info=result[1])

    def launch(self):
        config = {
            "listen": shared.cmd_opts.listen,
            "port": shared.cmd_opts.port,
            "keyfile": shared.cmd_opts.tls_keyfile,
            "certfile": shared.cmd_opts.tls_certfile,
            "loop": "auto", # auto, asyncio, uvloop
            "http": "auto", # auto, h11, httptools
        }
        from modules.server import UvicornServer
        http_server = UvicornServer(self.app, **config)
        # from modules.server import HypercornServer
        # server = HypercornServer(self.app, **config)
        http_server.start()
        shared.log.info(f'API server: Uvicorn options={config}')
        return http_server


# compatibility items
decode_base64_to_image = helpers.decode_base64_to_image
encode_pil_to_base64 = helpers.encode_pil_to_base64
validate_sampler_name = helpers.validate_sampler_name
