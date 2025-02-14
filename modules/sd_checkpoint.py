import os
import re
import time
import json
import collections
from modules import shared, paths, modelloader, hashes, sd_hijack_accelerate


checkpoints_list = {}
checkpoint_aliases = {}
checkpoints_loaded = collections.OrderedDict()
model_dir = "Stable-diffusion"
model_path = os.path.abspath(os.path.join(paths.models_path, model_dir))
sd_metadata_file = os.path.join(paths.data_path, "metadata.json")
sd_metadata = None
sd_metadata_pending = 0
sd_metadata_timer = 0


class CheckpointInfo:
    def __init__(self, filename, sha=None):
        self.name = None
        self.hash = sha
        self.filename = filename
        self.type = ''
        relname = filename
        app_path = os.path.abspath(paths.script_path)

        def rel(fn, path):
            try:
                return os.path.relpath(fn, path)
            except Exception:
                return fn

        if relname.startswith('..'):
            relname = os.path.abspath(relname)
        if relname.startswith(shared.opts.ckpt_dir):
            relname = rel(filename, shared.opts.ckpt_dir)
        elif relname.startswith(shared.opts.diffusers_dir):
            relname = rel(filename, shared.opts.diffusers_dir)
        elif relname.startswith(model_path):
            relname = rel(filename, model_path)
        elif relname.startswith(paths.script_path):
            relname = rel(filename, paths.script_path)
        elif relname.startswith(app_path):
            relname = rel(filename, app_path)
        else:
            relname = os.path.abspath(relname)
        relname, ext = os.path.splitext(relname)
        ext = ext.lower()[1:]

        if os.path.isfile(filename): # ckpt or safetensor
            self.name = relname
            self.filename = filename
            self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{relname}")
            self.type = ext
            if 'nf4' in filename:
                self.type = 'transformer'
        else: # maybe a diffuser
            if self.hash is None:
                repo = [r for r in modelloader.diffuser_repos if self.filename == r['name']]
            else:
                repo = [r for r in modelloader.diffuser_repos if self.hash == r['hash']]
            if len(repo) == 0:
                self.name = filename
                self.filename = filename
                self.sha256 = None
                self.type = 'unknown'
            else:
                self.name = os.path.join(os.path.basename(shared.opts.diffusers_dir), repo[0]['name'])
                self.filename = repo[0]['path']
                self.sha256 = repo[0]['hash']
                self.type = 'diffusers'

        self.shorthash = self.sha256[0:10] if self.sha256 else None
        self.title = self.name if self.shorthash is None else f'{self.name} [{self.shorthash}]'
        self.path = self.filename
        self.model_name = os.path.basename(self.name)
        self.metadata = read_metadata_from_safetensors(filename)
        # shared.log.debug(f'Checkpoint: type={self.type} name={self.name} filename={self.filename} hash={self.shorthash} title={self.title}')

    def register(self):
        checkpoints_list[self.title] = self
        for i in [self.name, self.filename, self.shorthash, self.title]:
            if i is not None:
                checkpoint_aliases[i] = self

    def calculate_shorthash(self):
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return None
        self.shorthash = self.sha256[0:10]
        if self.title in checkpoints_list:
            checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()
        return self.shorthash

    def __str__(self):
        return f'checkpoint: type={self.type} title="{self.title}" path="{self.path}"'


def setup_model():
    list_models()
    sd_hijack_accelerate.hijack_hfhub()
    # sd_hijack_accelerate.hijack_torch_conv()
    if not shared.native:
        enable_midas_autodownload()


def checkpoint_titles(use_short=False): # pylint: disable=unused-argument
    def convert(name):
        return int(name) if name.isdigit() else name.lower()
    def alphanumeric_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted([x.title for x in checkpoints_list.values()], key=alphanumeric_key)


def list_models():
    t0 = time.time()
    global checkpoints_list # pylint: disable=global-statement
    checkpoints_list.clear()
    checkpoint_aliases.clear()
    ext_filter = [".safetensors"] if shared.opts.sd_disable_ckpt or shared.native else [".ckpt", ".safetensors"]
    model_list = list(modelloader.load_models(model_path=model_path, model_url=None, command_path=shared.opts.ckpt_dir, ext_filter=ext_filter, download_name=None, ext_blacklist=[".vae.ckpt", ".vae.safetensors"]))
    safetensors_list = []
    for filename in sorted(model_list, key=str.lower):
        checkpoint_info = CheckpointInfo(filename)
        safetensors_list.append(checkpoint_info)
        if checkpoint_info.name is not None:
            checkpoint_info.register()
    diffusers_list = []
    if shared.native:
        for repo in modelloader.load_diffusers_models(clear=True):
            checkpoint_info = CheckpointInfo(repo['name'], sha=repo['hash'])
            diffusers_list.append(checkpoint_info)
            if checkpoint_info.name is not None:
                checkpoint_info.register()
    if shared.cmd_opts.ckpt is not None:
        if not os.path.exists(shared.cmd_opts.ckpt) and not shared.native:
            if shared.cmd_opts.ckpt.lower() != "none":
                shared.log.warning(f'Load model: path="{shared.cmd_opts.ckpt}" not found')
        else:
            checkpoint_info = CheckpointInfo(shared.cmd_opts.ckpt)
            if checkpoint_info.name is not None:
                checkpoint_info.register()
                shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    elif shared.cmd_opts.ckpt != shared.default_sd_model_file and shared.cmd_opts.ckpt is not None:
        shared.log.warning(f'Load model: path="{shared.cmd_opts.ckpt}" not found')
    shared.log.info(f'Available Models: items={len(checkpoints_list)} safetensors="{shared.opts.ckpt_dir}":{len(safetensors_list)} diffusers="{shared.opts.diffusers_dir}":{len(diffusers_list)} time={time.time()-t0:.2f}')
    checkpoints_list = dict(sorted(checkpoints_list.items(), key=lambda cp: cp[1].filename))

def update_model_hashes():
    txt = []
    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.hash is None]
    # shared.log.info(f'Models list: short hash missing for {len(lst)} out of {len(checkpoints_list)} models')
    for ckpt in lst:
        ckpt.hash = model_hash(ckpt.filename)
        # txt.append(f'Calculated short hash: <b>{ckpt.title}</b> {ckpt.hash}')
    # txt.append(f'Updated short hashes for <b>{len(lst)}</b> out of <b>{len(checkpoints_list)}</b> models')
    lst = [ckpt for ckpt in checkpoints_list.values() if ckpt.sha256 is None or ckpt.shorthash is None]
    shared.log.info(f'Models list: hash missing={len(lst)} total={len(checkpoints_list)}')
    for ckpt in lst:
        ckpt.sha256 = hashes.sha256(ckpt.filename, f"checkpoint/{ckpt.name}")
        ckpt.shorthash = ckpt.sha256[0:10] if ckpt.sha256 is not None else None
        if ckpt.sha256 is not None:
            txt.append(f'Hash: <b>{ckpt.title}</b> {ckpt.shorthash}')
    txt.append(f'Updated hashes for <b>{len(lst)}</b> out of <b>{len(checkpoints_list)}</b> models')
    txt = '<br>'.join(txt)
    return txt


def get_closet_checkpoint_match(s: str):
    if s.startswith('https://huggingface.co/'):
        model_name = s.replace('https://huggingface.co/', '')
        checkpoint_info = CheckpointInfo(model_name) # create a virutal model info
        checkpoint_info.type = 'huggingface'
        return checkpoint_info
    if s.startswith('huggingface/'):
        model_name = s.replace('huggingface/', '')
        checkpoint_info = CheckpointInfo(model_name) # create a virutal model info
        checkpoint_info.type = 'huggingface'
        return checkpoint_info

    # alias search
    checkpoint_info = checkpoint_aliases.get(s, None)
    if checkpoint_info is not None:
        return checkpoint_info

    # models search
    found = sorted([info for info in checkpoints_list.values() if os.path.basename(info.title).lower().startswith(s.lower())], key=lambda x: len(x.title))
    if found and len(found) == 1:
        return found[0]

    # absolute path
    if s.endswith('.safetensors') and os.path.isfile(s):
        checkpoint_info = CheckpointInfo(s)
        return checkpoint_info

    # reference search
    """
    found = sorted([info for info in shared.reference_models.values() if os.path.basename(info['path']).lower().startswith(s.lower())], key=lambda x: len(x['path']))
    if found and len(found) == 1:
        checkpoint_info = CheckpointInfo(found[0]['path']) # create a virutal model info
        checkpoint_info.type = 'huggingface'
        return checkpoint_info
    """

    # huggingface search
    if shared.opts.sd_checkpoint_autodownload and s.count('/') == 1:
        modelloader.hf_login()
        found = modelloader.find_diffuser(s, full=True)
        found = [f for f in found if f == s]
        shared.log.info(f'HF search: model="{s}" results={found}')
        if found is not None and len(found) == 1:
            checkpoint_info = CheckpointInfo(s)
            checkpoint_info.type = 'huggingface'
            return checkpoint_info

    # civitai search
    if shared.opts.sd_checkpoint_autodownload and s.startswith("https://civitai.com/api/download/models"):
        fn = modelloader.download_civit_model_thread(model_name=None, model_url=s, model_path='', model_type='Model', token=None)
        if fn is not None:
            checkpoint_info = CheckpointInfo(fn)
            return checkpoint_info

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""
    try:
        with open(filename, "rb") as file:
            import hashlib
            # t0 = time.time()
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            shorthash = m.hexdigest()[0:8]
            # t1 = time.time()
            # shared.log.debug(f'Calculating short hash: {filename} hash={shorthash} time={(t1-t0):.2f}')
            return shorthash
    except FileNotFoundError:
        return 'NOFILE'
    except Exception:
        return 'NOHASH'


def select_checkpoint(op='model'):
    if op == 'dict':
        model_checkpoint = shared.opts.sd_model_dict
    elif op == 'refiner':
        model_checkpoint = shared.opts.data.get('sd_model_refiner', None)
    else:
        model_checkpoint = shared.opts.sd_model_checkpoint
    if len(model_checkpoint) < 3:
        return None
    if model_checkpoint is None or model_checkpoint == 'None':
        return None
    checkpoint_info = get_closet_checkpoint_match(model_checkpoint)
    if checkpoint_info is not None:
        shared.log.info(f'Load {op}: select="{checkpoint_info.title if checkpoint_info is not None else None}"')
        return checkpoint_info
    if len(checkpoints_list) == 0:
        shared.log.warning("Cannot generate without a checkpoint")
        shared.log.info("Set system paths to use existing folders")
        shared.log.info("  or use --models-dir <path-to-folder> to specify base folder with all models")
        shared.log.info("  or use --ckpt-dir <path-to-folder> to specify folder with sd models")
        shared.log.info("  or use --ckpt <path-to-checkpoint> to force using specific model")
        return None
    # checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        if model_checkpoint != 'model.safetensors' and model_checkpoint != 'stabilityai/stable-diffusion-xl-base-1.0':
            shared.log.info(f'Load {op}: search="{model_checkpoint}" not found')
        else:
            shared.log.info("Selecting first available checkpoint")
        # shared.log.warning(f"Loading fallback checkpoint: {checkpoint_info.title}")
        # shared.opts.data['sd_model_checkpoint'] = checkpoint_info.title
    else:
        shared.log.info(f'Load {op}: select="{checkpoint_info.title if checkpoint_info is not None else None}"')
    return checkpoint_info


def init_metadata():
    global sd_metadata # pylint: disable=global-statement
    if sd_metadata is None:
        sd_metadata = shared.readfile(sd_metadata_file, lock=True) if os.path.isfile(sd_metadata_file) else {}


def read_metadata_from_safetensors(filename):
    global sd_metadata # pylint: disable=global-statement
    if sd_metadata is None:
        sd_metadata = shared.readfile(sd_metadata_file, lock=True) if os.path.isfile(sd_metadata_file) else {}
    res = sd_metadata.get(filename, None)
    if res is not None:
        return res
    if not filename.endswith(".safetensors"):
        return {}
    if shared.cmd_opts.no_metadata:
        return {}
    res = {}
    # try:
    t0 = time.time()
    with open(filename, mode="rb") as file:
        try:
            metadata_len = file.read(8)
            metadata_len = int.from_bytes(metadata_len, "little")
            json_start = file.read(2)
            if metadata_len <= 2 or json_start not in (b'{"', b"{'"):
                shared.log.error(f'Model metadata invalid: file="{filename}"')
            json_data = json_start + file.read(metadata_len-2)
            json_obj = json.loads(json_data)
            for k, v in json_obj.get("__metadata__", {}).items():
                if v.startswith("data:"):
                    v = 'data'
                if k == 'format' and v == 'pt':
                    continue
                large = True if len(v) > 2048 else False
                if large and k in ['ss_datasets', 'workflow', 'prompt', 'ss_bucket_info', 'sd_metadata_file']:
                    continue
                if v[0:1] == '{':
                    try:
                        v = json.loads(v)
                        if large and k == 'ss_tag_frequency':
                            v = { i: len(j) for i, j in v.items() }
                        if large and k == 'sd_merge_models':
                            scrub_dict(v, ['sd_merge_recipe'])
                    except Exception:
                        pass
                res[k] = v
        except Exception as e:
            shared.log.error(f'Model metadata: file="{filename}" {e}')
    sd_metadata[filename] = res
    global sd_metadata_pending # pylint: disable=global-statement
    sd_metadata_pending += 1
    t1 = time.time()
    global sd_metadata_timer # pylint: disable=global-statement
    sd_metadata_timer += (t1 - t0)
    # except Exception as e:
    #    shared.log.error(f"Error reading metadata from: {filename} {e}")
    return res


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """
    from urllib import request
    import ldm.modules.midas.api
    midas_path = os.path.join(paths.models_path, 'midas')
    for k, v in ldm.modules.midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        ldm.modules.midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)
    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }
    ldm.modules.midas.api.load_model_inner = ldm.modules.midas.api.load_model

    def load_model_wrapper(model_type):
        path = ldm.modules.midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                os.mkdir(midas_path)
            shared.log.info(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            shared.log.info(f"{model_type} downloaded")
        return ldm.modules.midas.api.load_model_inner(model_type)

    ldm.modules.midas.api.load_model = load_model_wrapper


def scrub_dict(dict_obj, keys):
    for key in list(dict_obj.keys()):
        if not isinstance(dict_obj, dict):
            continue
        if key in keys:
            dict_obj.pop(key, None)
        elif isinstance(dict_obj[key], dict):
            scrub_dict(dict_obj[key], keys)
        elif isinstance(dict_obj[key], list):
            for item in dict_obj[key]:
                scrub_dict(item, keys)


def write_metadata():
    global sd_metadata_pending # pylint: disable=global-statement
    if sd_metadata_pending == 0:
        shared.log.debug(f'Model metadata: file="{sd_metadata_file}" no changes')
        return
    shared.writefile(sd_metadata, sd_metadata_file)
    shared.log.info(f'Model metadata saved: file="{sd_metadata_file}" items={sd_metadata_pending} time={sd_metadata_timer:.2f}')
    sd_metadata_pending = 0
