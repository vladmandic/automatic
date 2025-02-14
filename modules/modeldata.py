import sys
import threading
from modules import shared, errors


def get_model_type(pipe):
    name = pipe.__class__.__name__
    if not shared.native:
        model_type = 'ldm'
    elif "StableDiffusion3" in name:
        model_type = 'sd3'
    elif "StableDiffusionXL" in name:
        model_type = 'sdxl'
    elif "StableDiffusion" in name:
        model_type = 'sd'
    elif "LatentConsistencyModel" in name:
        model_type = 'sd' # lcm is compatible with sd
    elif "InstaFlowPipeline" in name:
        model_type = 'sd' # instaflow is compatible with sd
    elif "AnimateDiffPipeline" in name:
        model_type = 'sd' # animatediff is compatible with sd
    elif "Kandinsky" in name:
        model_type = 'kandinsky'
    elif "HunyuanDiT" in name:
        model_type = 'hunyuandit'
    elif "Cascade" in name:
        model_type = 'sc'
    elif "AuraFlow" in name:
        model_type = 'auraflow'
    elif "Flux" in name:
        model_type = 'f1'
    elif "Lumina" in name:
        model_type = 'lumina'
    elif "OmniGen" in name:
        model_type = 'omnigen'
    elif "CogVideo" in name:
        model_type = 'cogvideox'
    elif "Sana" in name:
        model_type = 'sana'
    else:
        model_type = name
    return model_type


class ModelData:
    def __init__(self):
        self.sd_model = None
        self.sd_refiner = None
        self.sd_dict = 'None'
        self.initial = True
        self.lock = threading.Lock()

    def get_sd_model(self):
        from modules.sd_models import reload_model_weights
        if self.sd_model is None and shared.opts.sd_model_checkpoint != 'None' and not self.lock.locked():
            with self.lock:
                try:
                    # note: reload_model_weights directly updates model_data.sd_model and returns it at the end
                    self.sd_model = reload_model_weights(op='model')
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self.sd_model = None
        return self.sd_model

    def set_sd_model(self, v):
        self.sd_model = v

    def get_sd_refiner(self):
        from modules.sd_models import reload_model_weights
        if self.sd_refiner is None and shared.opts.sd_model_refiner != 'None' and not self.lock.locked():
            with self.lock:
                try:
                    self.sd_refiner = reload_model_weights(op='refiner')
                    self.initial = False
                except Exception as e:
                    shared.log.error("Failed to load stable diffusion model")
                    errors.display(e, "loading stable diffusion model")
                    self.sd_refiner = None
        return self.sd_refiner

    def set_sd_refiner(self, v):
        self.sd_refiner = v


# provides shared.sd_model field as a property
class Shared(sys.modules[__name__].__class__):
    @property
    def sd_loaded(self):
        import modules.sd_models # pylint: disable=W0621
        return modules.sd_models.model_data.sd_model is not None

    @property
    def sd_model(self):
        import modules.sd_models # pylint: disable=W0621
        if modules.sd_models.model_data.sd_model is None:
            fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            shared.log.debug(f'Model requested: fn={fn}') # pylint: disable=protected-access
        return modules.sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        import modules.sd_models # pylint: disable=W0621
        modules.sd_models.model_data.set_sd_model(value)

    @property
    def sd_refiner(self):
        import modules.sd_models # pylint: disable=W0621
        return modules.sd_models.model_data.get_sd_refiner()

    @sd_refiner.setter
    def sd_refiner(self, value):
        import modules.sd_models # pylint: disable=W0621
        modules.sd_models.model_data.set_sd_refiner(value)

    @property
    def sd_model_type(self):
        try:
            import modules.sd_models # pylint: disable=W0621
            if modules.sd_models.model_data.sd_model is None:
                model_type = 'none'
                return model_type
            model_type = get_model_type(self.sd_model)
        except Exception:
            model_type = 'unknown'
        return model_type

    @property
    def sd_refiner_type(self):
        try:
            import modules.sd_models # pylint: disable=W0621
            if modules.sd_models.model_data.sd_refiner is None:
                model_type = 'none'
                return model_type
            model_type = get_model_type(self.sd_refiner)
        except Exception:
            model_type = 'unknown'
        return model_type


model_data = ModelData()
