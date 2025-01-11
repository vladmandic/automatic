import os
import sys
import time
from collections import namedtuple
from pathlib import Path
import re
import torch
import torch.hub # pylint: disable=ungrouped-imports
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules import devices, paths, shared, lowvram, errors


config = {
    "caption_max_length": 64,
    "chunk_size": 1024,
    "flavor_intermediate_count": 1024,
    "min_flavors": 2,
    "max_flavors": 8,
    "clip_offload": True,
    "caption_offload": True,
}
caption_models = {
    'blip-base': 'Salesforce/blip-image-captioning-base',
    'blip-large': 'Salesforce/blip-image-captioning-large',
    'blip2-opt-2.7b': 'Salesforce/blip2-opt-2.7b-coco',
    'blip2-opt-6.7b': 'Salesforce/blip2-opt-6.7b',
    'blip2-flip-t5-xl': 'Salesforce/blip2-flan-t5-xl',
    'blip2-flip-t5-xxl': 'Salesforce/blip2-flan-t5-xxl',
}
ci = None
blip_image_eval_size = 384
clip_model_name = 'ViT-L/14'
Category = namedtuple("Category", ["name", "topn", "items"])
re_topn = re.compile(r"\.top(\d+)\.")


def category_types():
    return [f.stem for f in Path(shared.interrogator.content_dir).glob('*.txt')]


def download_default_clip_interrogate_categories(content_dir):
    shared.log.info("Downloading CLIP categories...")
    tmpdir = f"{content_dir}_tmp"
    cat_types = ["artists", "flavors", "mediums", "movements"]
    try:
        os.makedirs(tmpdir, exist_ok=True)
        for category_type in cat_types:
            torch.hub.download_url_to_file(f"https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/{category_type}.txt", os.path.join(tmpdir, f"{category_type}.txt"))
        os.rename(tmpdir, content_dir)
    except Exception as e:
        errors.display(e, "downloading default CLIP interrogate categories")
    finally:
        if os.path.exists(tmpdir):
            os.removedirs(tmpdir)


class InterrogateModels:
    blip_model = None
    clip_model = None
    clip_preprocess = None
    dtype = None
    running_on_cpu = None

    def __init__(self, content_dir):
        self.loaded_categories = None
        self.skip_categories = []
        self.content_dir = content_dir
        self.running_on_cpu = False

    def categories(self):
        if not os.path.exists(self.content_dir):
            download_default_clip_interrogate_categories(self.content_dir)
        if self.loaded_categories is not None and self.skip_categories == shared.opts.interrogate_clip_skip_categories:
            return self.loaded_categories
        self.loaded_categories = []

        if os.path.exists(self.content_dir):
            self.skip_categories = shared.opts.interrogate_clip_skip_categories
            cat_types = []
            for filename in Path(self.content_dir).glob('*.txt'):
                cat_types.append(filename.stem)
                if filename.stem in self.skip_categories:
                    continue
                m = re_topn.search(filename.stem)
                topn = 1 if m is None else int(m.group(1))
                with open(filename, "r", encoding="utf8") as file:
                    lines = [x.strip() for x in file.readlines()]
                self.loaded_categories.append(Category(name=filename.stem, topn=topn, items=lines))
        return self.loaded_categories

    def create_fake_fairscale(self):
        class FakeFairscale:
            def checkpoint_wrapper(self):
                pass
        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale

    def load_blip_model(self):
        self.create_fake_fairscale()
        from repositories.blip import models # pylint: disable=unused-import
        from repositories.blip.models import blip
        import modules.modelloader as modelloader
        model_path = os.path.join(paths.models_path, "BLIP")
        download_name='model_base_caption_capfilt_large.pth'
        shared.log.debug(f'Model interrogate load: type=BLiP model={download_name} path={model_path}')
        files = modelloader.load_models(
            model_path=model_path,
            model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
            ext_filter=[".pth"],
            download_name=download_name,
        )
        blip_model = blip.blip_decoder(pretrained=files[0], image_size=blip_image_eval_size, vit='base', med_config=os.path.join(paths.paths["BLIP"], "configs", "med_config.json")) # pylint: disable=c-extension-no-member
        blip_model.eval()

        return blip_model

    def load_clip_model(self):
        shared.log.debug(f'Model interrogate load: type=CLiP model={clip_model_name} path={shared.opts.clip_models_path}')
        import clip
        if self.running_on_cpu:
            model, preprocess = clip.load(clip_model_name, device="cpu", download_root=shared.opts.clip_models_path)
        else:
            model, preprocess = clip.load(clip_model_name, download_root=shared.opts.clip_models_path)
        model.eval()
        model = model.to(devices.device)
        return model, preprocess

    def load(self):
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            if not shared.opts.no_half and not self.running_on_cpu:
                self.blip_model = self.blip_model.half()
        self.blip_model = self.blip_model.to(devices.device)
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = self.load_clip_model()
            if not shared.opts.no_half and not self.running_on_cpu:
                self.clip_model = self.clip_model.half()
        self.clip_model = self.clip_model.to(devices.device)
        self.dtype = next(self.clip_model.parameters()).dtype

    def send_clip_to_ram(self):
        if not shared.opts.interrogate_keep_models_in_memory:
            if self.clip_model is not None:
                self.clip_model = self.clip_model.to(devices.cpu)

    def send_blip_to_ram(self):
        if not shared.opts.interrogate_keep_models_in_memory:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(devices.cpu)

    def unload(self):
        self.send_clip_to_ram()
        self.send_blip_to_ram()
        devices.torch_gc()

    def rank(self, image_features, text_array, top_count=1):
        import clip
        devices.torch_gc()
        if shared.opts.interrogate_clip_dict_limit != 0:
            text_array = text_array[0:int(shared.opts.interrogate_clip_dict_limit)]
        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize(list(text_array), truncate=True).to(devices.device)
        text_features = self.clip_model.encode_text(text_tokens).type(self.dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = torch.zeros((1, len(text_array))).to(devices.device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]
        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(devices.device)
        with devices.inference_context():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=shared.opts.interrogate_clip_num_beams, min_length=shared.opts.interrogate_clip_min_length, max_length=shared.opts.interrogate_clip_max_length)
        return caption[0]

    def interrogate(self, pil_image):
        res = ""
        shared.state.begin('Interrogate')
        try:
            if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
                lowvram.send_everything_to_cpu()
                devices.torch_gc()
            self.load()
            if isinstance(pil_image, list):
                pil_image = pil_image[0] if len(pil_image) > 0 else None
            if isinstance(pil_image, dict) and 'name' in pil_image:
                pil_image = Image.open(pil_image['name'])
            if pil_image is None:
                return ''
            pil_image = pil_image.convert("RGB")
            caption = self.generate_caption(pil_image)
            self.send_blip_to_ram()
            devices.torch_gc()
            res = caption
            clip_image = self.clip_preprocess(pil_image).unsqueeze(0).type(self.dtype).to(devices.device)
            with devices.inference_context(), devices.autocast():
                image_features = self.clip_model.encode_image(clip_image).type(self.dtype)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                for _name, topn, items in self.categories():
                    matches = self.rank(image_features, items, top_count=topn)
                    for match, score in matches:
                        if shared.opts.interrogate_return_ranks:
                            res += f", ({match}:{score/100:.3f})"
                        else:
                            res += f", {match}"
        except Exception as e:
            errors.display(e, 'interrogate')
            res += "<error>"
        self.unload()
        shared.state.end()
        return res

# --------- interrrogate ui

class BatchWriter:
    def __init__(self, folder):
        self.folder = folder
        self.csv, self.file = None, None

    def add(self, file, prompt):
        txt_file = os.path.splitext(file)[0] + ".txt"
        with open(os.path.join(self.folder, txt_file), 'w', encoding='utf-8') as f:
            f.write(prompt)

    def close(self):
        if self.file is not None:
            self.file.close()


def update_interrogate_params(caption_max_length, chunk_size, min_flavors, max_flavors, flavor_intermediate_count):
    config["caption_max_length"] = int(caption_max_length)
    config["chunk_size"] = int(chunk_size)
    config["min_flavors"] = int(min_flavors)
    config["max_flavors"] = int(max_flavors)
    config["flavor_intermediate_count"] = int(flavor_intermediate_count)
    if ci is not None:
        ci.config.caption_max_length = config["caption_max_length"]
        ci.config.chunk_size = config["chunk_size"]
        ci.config.flavor_intermediate_count = config["flavor_intermediate_count"]
    shared.log.debug(f'Interrogate params: {config}')

def get_clip_models():
    import open_clip
    models = sorted(open_clip.list_pretrained())
    shared.log.info(f'Interrogate: pkg=openclip version={open_clip.__version__} models={len(models)}')
    return ['/'.join(x) for x in models]


def load_interrogator(clip_model, blip_model):
    from installer import install
    install('clip_interrogator==0.6.0')
    import clip_interrogator
    clip_interrogator.clip_interrogator.CAPTION_MODELS = caption_models
    global ci # pylint: disable=global-statement
    if ci is None:
        interrogator_config = clip_interrogator.Config(
            device=devices.get_optimal_device(),
            cache_path=os.path.join(paths.models_path, 'Interrogator'),
            clip_model_name=clip_model,
            caption_model_name=blip_model,
            quiet=True,
            caption_max_length=config['caption_max_length'],
            chunk_size=config['chunk_size'],
            flavor_intermediate_count=config['flavor_intermediate_count'],
            clip_offload=config['clip_offload'],
            caption_offload=config['caption_offload'],
        )
        t0 = time.time()
        ci = clip_interrogator.Interrogator(interrogator_config)
        t1 = time.time()
        shared.log.info(f'Interrogate load: config={ci.config} min_flavors={config["min_flavors"]} max_flavors={config["max_flavors"]} time={t1-t0:.2f}')
    elif clip_model != ci.config.clip_model_name or blip_model != ci.config.caption_model_name:
        t0 = time.time()
        ci.config.clip_model_name = clip_model
        ci.config.clip_model = None
        ci.load_clip_model()
        ci.config.caption_model_name = blip_model
        ci.config.caption_model = None
        ci.load_caption_model()
        t1 = time.time()
        shared.log.info(f'Interrogate reload: config={ci.config} min_flavors={config["min_flavors"]} max_flavors={config["max_flavors"]} time={t1-t0:.2f}')


def unload_clip_model():
    if ci is not None:
        shared.log.debug('Interrogate offload')
        ci.caption_model = ci.caption_model.to(devices.cpu)
        ci.clip_model = ci.clip_model.to(devices.cpu)
        ci.caption_offloaded = True
        ci.clip_offloaded = True
        devices.torch_gc()


def interrogate(image, mode, caption=None):
    shared.log.info(f'Interrogate: mode={mode} image={image}')
    t0 = time.time()
    if mode == 'best':
        prompt = ci.interrogate(image, caption=caption, min_flavors=config["min_flavors"], max_flavors=config["max_flavors"])
    elif mode == 'caption':
        prompt = ci.generate_caption(image) if caption is None else caption
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image, caption=caption, max_flavors=config["max_flavors"])
    elif mode == 'fast':
        prompt = ci.interrogate_fast(image, caption=caption, max_flavors=config["max_flavors"])
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image, max_flavors=config["max_flavors"])
    else:
        raise RuntimeError(f"Unknown mode {mode}")
    t1 = time.time()
    shared.log.debug(f'Interrogate: prompt="{prompt}" time={t1-t0:.2f}')
    return prompt


def interrogate_image(image, clip_model, blip_model, mode):
    shared.state.begin('Interrogate')
    try:
        if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
            lowvram.send_everything_to_cpu()
            devices.torch_gc()
        load_interrogator(clip_model, blip_model)
        image = image.convert('RGB')
        prompt = interrogate(image, mode)
        devices.torch_gc()
    except Exception as e:
        prompt = f"Exception {type(e)}"
        shared.log.error(f'Interrogate: {e}')
        errors.display(e, 'Interrogate')
    shared.state.end()
    return prompt


def interrogate_batch(batch_files, batch_folder, batch_str, clip_model, blip_model, mode, write):
    files = []
    if batch_files is not None:
        files += [f.name for f in batch_files]
    if batch_folder is not None:
        files += [f.name for f in batch_folder]
    if batch_str is not None and len(batch_str) > 0 and os.path.exists(batch_str) and os.path.isdir(batch_str):
        files += [os.path.join(batch_str, f) for f in os.listdir(batch_str) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    if len(files) == 0:
        shared.log.error('Interrogate batch no images')
        return ''
    shared.state.begin('Batch interrogate')
    prompts = []
    try:
        if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
            lowvram.send_everything_to_cpu()
            devices.torch_gc()
        load_interrogator(clip_model, blip_model)
        shared.log.info(f'Interrogate batch: images={len(files)} mode={mode} config={ci.config}')
        captions = []
        # first pass: generate captions
        for file in files:
            caption = ""
            try:
                if shared.state.interrupted:
                    break
                image = Image.open(file).convert('RGB')
                caption = ci.generate_caption(image)
            except Exception as e:
                shared.log.error(f'Interrogate caption: {e}')
            finally:
                captions.append(caption)
        # second pass: interrogate
        if write:
            writer = BatchWriter(os.path.dirname(files[0]))
        for idx, file in enumerate(files):
            try:
                if shared.state.interrupted:
                    break
                image = Image.open(file).convert('RGB')
                prompt = interrogate(image, mode, caption=captions[idx])
                prompts.append(prompt)
                if write:
                    writer.add(file, prompt)
            except OSError as e:
                shared.log.error(f'Interrogate batch: {e}')
        if write:
            writer.close()
        ci.config.quiet = False
        unload_clip_model()
    except Exception as e:
        shared.log.error(f'Interrogate batch: {e}')
    shared.state.end()
    return '\n\n'.join(prompts)


def analyze_image(image, clip_model, blip_model):
    load_interrogator(clip_model, blip_model)
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)
    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)
    medium_ranks = dict(zip(top_mediums, ci.similarities(image_features, top_mediums)))
    artist_ranks = dict(zip(top_artists, ci.similarities(image_features, top_artists)))
    movement_ranks = dict(zip(top_movements, ci.similarities(image_features, top_movements)))
    trending_ranks = dict(zip(top_trendings, ci.similarities(image_features, top_trendings)))
    flavor_ranks = dict(zip(top_flavors, ci.similarities(image_features, top_flavors)))
    return medium_ranks, artist_ranks, movement_ranks, trending_ranks, flavor_ranks
