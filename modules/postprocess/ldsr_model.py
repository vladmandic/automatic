import os
import sys
import traceback
from modules.upscaler import Upscaler, UpscalerData
from modules import shared, script_callbacks


class Dummy:
    pass

cls = Upscaler if not shared.native else Dummy

class UpscalerLDSR(cls):
    def __init__(self, user_path):
        self.name = "LDSR"
        self.user_path = user_path
        self.model_url = "https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1"
        self.yaml_url = "https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1"
        super().__init__()
        scaler_data = UpscalerData("LDSR", None, self)
        self.scalers = [scaler_data]

    def load_model(self, path: str):
        from modules.ldsr.ldsr_model_arch import LDSR
        import modules.ldsr.sd_hijack_autoencoder # pylint: disable=unused-import
        import modules.ldsr.sd_hijack_ddpm_v1 # pylint: disable=unused-import
        # Remove incorrect project.yaml file if too big
        yaml_path = os.path.join(self.model_path, "project.yaml")
        old_model_path = os.path.join(self.model_path, "model.pth")
        new_model_path = os.path.join(self.model_path, "model.ckpt")

        local_model_paths = self.find_models(ext_filter=[".ckpt", ".safetensors"])
        local_ckpt_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("model.ckpt")]), None)
        local_safetensors_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("model.safetensors")]), None)
        local_yaml_path = next(iter([local_model for local_model in local_model_paths if local_model.endswith("project.yaml")]), None)

        if os.path.exists(yaml_path):
            statinfo = os.stat(yaml_path)
            if statinfo.st_size >= 10485760:
                print("Removing invalid LDSR YAML file.")
                os.remove(yaml_path)

        if os.path.exists(old_model_path):
            print("Renaming model from model.pth to model.ckpt")
            os.rename(old_model_path, new_model_path)

        from modules.modelloader import load_file_from_url
        if local_safetensors_path is not None and os.path.exists(local_safetensors_path):
            model = local_safetensors_path
        else:
            model = local_ckpt_path if local_ckpt_path is not None else load_file_from_url(url=self.model_url, model_dir=self.model_download_path, file_name="model.ckpt", progress=True)

        yaml = local_yaml_path if local_yaml_path is not None else load_file_from_url(url=self.yaml_url, model_dir=self.model_download_path, file_name="project.yaml", progress=True)

        try:
            return LDSR(model, yaml)
        except Exception:
            print("Error importing LDSR:", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
        return None

    def do_upscale(self, img, selected_model):
        ldsr = self.load_model(selected_model)
        if ldsr is None:
            print("NO LDSR!")
            return img
        ddim_steps = shared.opts.ldsr_steps
        return ldsr.super_resolution(img, ddim_steps, self.scale)


def on_ui_settings():
    import gradio as gr
    shared.opts.add_option("ldsr_steps", shared.OptionInfo(100, "LDSR processing steps", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}, section=('postprocessing', "Postprocessing")))

script_callbacks.on_ui_settings(on_ui_settings)
