import os
import re
import torch
import numpy as np
from PIL import Image
from modules import modelloader, paths, deepbooru_model, devices, images, shared

re_special = re.compile(r'([\\()])')


class DeepDanbooru:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is not None:
            return
        model_path = os.path.join(paths.models_path, "DeepDanbooru")
        shared.log.debug(f'Load interrogate model: type=DeepDanbooru folder="{model_path}"')
        files = modelloader.load_models(
            model_path=model_path,
            model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',
            ext_filter=[".pt"],
            download_name='model-resnet_custom_v3.pt',
        )

        self.model = deepbooru_model.DeepDanbooruModel()
        self.model.load_state_dict(torch.load(files[0], map_location="cpu"))

        self.model.eval()
        self.model.to(devices.cpu, devices.dtype)

    def start(self):
        self.load()
        self.model.to(devices.device)

    def stop(self):
        if not shared.opts.interrogate_keep_models_in_memory:
            self.model.to(devices.cpu)
            devices.torch_gc()

    def tag(self, pil_image):
        self.start()
        res = self.tag_multi(pil_image)
        self.stop()

        return res

    def tag_multi(self, pil_image, force_disable_ranks=False):
        if isinstance(pil_image, list):
            pil_image = pil_image[0] if len(pil_image) > 0 else None
        if isinstance(pil_image, dict) and 'name' in pil_image:
            pil_image = Image.open(pil_image['name'])
        if pil_image is None:
            return ''
        pic = images.resize_image(2, pil_image.convert("RGB"), 512, 512)
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
        with devices.inference_context(), devices.autocast():
            x = torch.from_numpy(a).to(devices.device)
            y = self.model(x)[0].detach().float().cpu().numpy()
        probability_dict = {}
        for tag, probability in zip(self.model.tags, y):
            if probability < shared.opts.interrogate_deepbooru_score_threshold:
                continue
            if tag.startswith("rating:"):
                continue
            probability_dict[tag] = probability
        if shared.opts.deepbooru_sort_alpha:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]
        res = []
        filtertags = {x.strip().replace(' ', '_') for x in shared.opts.deepbooru_filter_tags.split(",")}
        for tag in [x for x in tags if x not in filtertags]:
            probability = probability_dict[tag]
            tag_outformat = tag
            if shared.opts.deepbooru_use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            if shared.opts.deepbooru_escape:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            if shared.opts.interrogate_return_ranks and not force_disable_ranks:
                tag_outformat = f"({tag_outformat}:{probability:.3f})"
            res.append(tag_outformat)
        return ", ".join(res)


model = DeepDanbooru()
