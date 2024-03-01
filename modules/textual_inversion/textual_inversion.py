from typing import List, Union
import os
import time
from collections import namedtuple
import torch
import safetensors.torch
from PIL import Image
from modules import shared, devices, sd_models, errors
from modules.textual_inversion.image_embedding import embedding_from_b64, extract_image_data_embed
from modules.files_cache import directory_files, directory_mtime, extension_filter


debug = shared.log.trace if os.environ.get('SD_TI_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: TEXTUAL INVERSION')
TokenToAdd = namedtuple("TokenToAdd", ["clip_l", "clip_g"])
TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"])
textual_inversion_templates = {}


def list_textual_inversion_templates():
    textual_inversion_templates.clear()
    for root, _dirs, fns in os.walk(shared.opts.embeddings_templates_dir):
        for fn in fns:
            path = os.path.join(root, fn)
            textual_inversion_templates[fn] = TextualInversionTemplate(fn, path)
    return textual_inversion_templates


def list_embeddings(*dirs):
    is_ext = extension_filter(['.SAFETENSORS', '.PT' ] + ( ['.PNG', '.WEBP', '.JXL', '.AVIF', '.BIN' ] if shared.backend != shared.Backend.DIFFUSERS else [] ))
    is_not_preview = lambda fp: not next(iter(os.path.splitext(fp))).upper().endswith('.PREVIEW') # pylint: disable=unnecessary-lambda-assignment
    return list(filter(lambda fp: is_ext(fp) and is_not_preview(fp) and os.stat(fp).st_size > 0, directory_files(*dirs)))


class Embedding:
    def __init__(self, vec, name, filename=None, step=None):
        self.vec = vec
        self.name = name
        self.tag = name
        self.step = step
        self.filename = filename
        self.basename = os.path.relpath(filename, shared.opts.embeddings_dir) if filename is not None else None
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }
        torch.save(embedding_data, filename)
        if shared.opts.save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, f"{filename}.optim")

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum
        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r
        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False
        return directory_mtime(self.path) != self.mtime

    def update(self):
        if not os.path.isdir(self.path):
            return
        self.mtime = directory_mtime(self.path)


def convert_embedding(tensor, text_encoder, text_encoder_2):
    with torch.no_grad():
        vectors = []
        clip_l_embeds = text_encoder.get_input_embeddings().weight.data.clone().to(device=devices.device)
        tensor = tensor.to(device=devices.device)
        for vec in tensor:
            values, indices = torch.max(torch.nan_to_num(torch.cosine_similarity(vec.unsqueeze(0), clip_l_embeds)), 0)
            if values < 0.707:  # Arbitrary similarity to cutoff, here 45 degrees
                indices *= 0  # Use SDXL padding vector 0
            vectors.append(indices)
        vectors = torch.stack(vectors)
        output = text_encoder_2.get_input_embeddings().weight.data[vectors]
    return output


class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}
        self.previously_displayed_embeddings = ()
        self.embeddings_used = []

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding
        if hasattr(model, 'cond_stage_model'):
            ids = model.cond_stage_model.tokenize([embedding.name])[0]
        elif hasattr(model, 'tokenizer'):
            ids = model.tokenizer.convert_tokens_to_ids(embedding.name)
        if type(ids) != list:
            ids = [ids]
        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []
        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)
        return embedding

    def get_expected_shape(self):
        if shared.backend == shared.Backend.DIFFUSERS:
            return 0
        if shared.sd_model is None:
            shared.log.error('Model not loaded')
            return 0
        vec = shared.sd_model.cond_stage_model.encode_embedding_init_text(",", 1)
        return vec.shape[1]

    def load_diffusers_embedding(self, filename: Union[str, List[str]]):
        _loaded_pre = len(self.word_embeddings)
        embeddings_to_load = []
        loaded_embeddings = {}
        skipped_embeddings = []
        if shared.sd_model is None:
            return 0
        tokenizer   = getattr(shared.sd_model, 'tokenizer',   None)
        tokenizer_2 = getattr(shared.sd_model, 'tokenizer_2', None)
        clip_l = getattr(shared.sd_model, 'text_encoder',   None)
        clip_g = getattr(shared.sd_model, 'text_encoder_2', None)
        if clip_g and tokenizer_2:
            model_type = 'SDXL'
        elif clip_l and tokenizer:
            model_type = 'SD'
        else:
            return 0
        filenames = list(filename)
        exts = [".SAFETENSORS", '.BIN', '.PT', '.PNG', '.WEBP', '.JXL', '.AVIF']
        for filename in filenames:
            # debug(f'Embedding check: {filename}')
            fullname = filename
            filename = os.path.basename(fullname)
            fn, ext = os.path.splitext(filename)
            name = os.path.basename(fn)
            embedding = Embedding(vec=None, name=name, filename=fullname)
            tokenizer_vocab = tokenizer.get_vocab()
            try:
                if ext.upper() not in exts:
                    raise ValueError(f'extension `{ext}` is invalid, expected one of: {exts}')
                if name in tokenizer.get_vocab() or f"{name}_1" in tokenizer.get_vocab():
                    loaded_embeddings[name] = embedding
                    debug(f'Embedding already loaded: {name}')
                embeddings_to_load.append(embedding)
            except Exception as e:
                skipped_embeddings.append(embedding)
                debug(f'Embedding skipped: "{name}" {e}')
                continue
            embeddings_to_load = sorted(embeddings_to_load, key=lambda e: exts.index(os.path.splitext(e.filename)[1].upper()))

        tokens_to_add = {}
        for embedding in embeddings_to_load:
            try:
                if embedding.name in tokens_to_add or embedding.name in loaded_embeddings:
                    raise ValueError('duplicate token')
                embeddings_dict = {}
                _, ext = os.path.splitext(embedding.filename)
                if ext.upper() in ['.SAFETENSORS']:
                    with safetensors.torch.safe_open(embedding.filename, framework="pt") as f: # type: ignore
                        for k in f.keys():
                            embeddings_dict[k] = f.get_tensor(k)
                else:  # fallback for sd1.5 pt embeddings
                    embeddings_dict["clip_l"] = self.load_from_file(embedding.filename, embedding.filename)
                if 'clip_l' not in embeddings_dict:
                    raise ValueError('Invalid Embedding, dict missing required key `clip_l`')
                if 'clip_g' not in embeddings_dict and model_type == "SDXL" and shared.opts.diffusers_convert_embed:
                    embeddings_dict["clip_g"] = convert_embedding(embeddings_dict["clip_l"], clip_l, clip_g)
                if 'clip_g' in embeddings_dict:
                    embedding_type = 'SDXL'
                else:
                    embedding_type = 'SD'
                if embedding_type != model_type:
                    raise ValueError(f'Unable to load {embedding_type} Embedding "{embedding.name}" into {model_type} Model')
                _tokens_to_add = {}
                for i in range(len(embeddings_dict["clip_l"])):
                    if len(clip_l.get_input_embeddings().weight.data[0]) == len(embeddings_dict["clip_l"][i]):
                        token = embedding.name if i == 0 else f"{embedding.name}_{i}"
                        if token in tokenizer_vocab:
                            raise RuntimeError(f'Multi-Vector Embedding would add pre-existing Token in Vocabulary: {token}')
                        if token in tokens_to_add:
                            raise RuntimeError(f'Multi-Vector Embedding would add duplicate Token to Add: {token}')
                        _tokens_to_add[token] = TokenToAdd(
                            embeddings_dict["clip_l"][i],
                            embeddings_dict["clip_g"][i] if 'clip_g' in embeddings_dict else None
                        )
                if not _tokens_to_add:
                    raise ValueError('no valid tokens to add')
                tokens_to_add.update(_tokens_to_add)
                loaded_embeddings[embedding.name] = embedding
            except Exception as e:
                debug(f"Embedding loading: {embedding.filename} {e}")
                continue
        if len(tokens_to_add) > 0:
            tokenizer.add_tokens(list(tokens_to_add.keys()))
            clip_l.resize_token_embeddings(len(tokenizer))
            if model_type == 'SDXL':
                tokenizer_2.add_tokens(list(tokens_to_add.keys())) # type: ignore
                clip_g.resize_token_embeddings(len(tokenizer_2)) # type: ignore
            unk_token_id = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
            for token, data in tokens_to_add.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id > unk_token_id:
                    clip_l.get_input_embeddings().weight.data[token_id] = data.clip_l
                    if model_type == 'SDXL':
                        clip_g.get_input_embeddings().weight.data[token_id] = data.clip_g # type: ignore

        for embedding in loaded_embeddings.values():
            if not embedding:
                continue
            self.register_embedding(embedding, shared.sd_model)
            if embedding in embeddings_to_load:
                embeddings_to_load.remove(embedding)
        skipped_embeddings.extend(embeddings_to_load)
        for embedding in skipped_embeddings:
            if loaded_embeddings.get(embedding.name, None) == embedding:
                continue
            self.skipped_embeddings[embedding.name] = embedding
        try:
            if model_type == 'SD':
                debug(f"Embeddings loaded: text-encoder={shared.sd_model.text_encoder.get_input_embeddings().weight.data.shape[0]}")
            if model_type == 'SDXL':
                debug(f"Embeddings loaded: text-encoder-1={shared.sd_model.text_encoder.get_input_embeddings().weight.data.shape[0]} text-encoder-2={shared.sd_model.text_encoder_2.get_input_embeddings().weight.data.shape[0]}")
        except Exception:
            pass
        return len(self.word_embeddings) - _loaded_pre

    def load_from_file(self, path, filename):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()

        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            if '.preview' in filename.lower():
                return
            embed_image = Image.open(path)
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
            else:
                data = extract_image_data_embed(embed_image)
                if not data: # if data is None, means this is not an embeding, just a preview image
                    return
        elif ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location="cpu")
        elif ext in ['.SAFETENSORS']:
            data = safetensors.torch.load_file(path, device="cpu")
        else:
            return

        # textual inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            param_dict = getattr(param_dict, '_parameters', param_dict)  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            if len(data.keys()) != 1:
                self.skipped_embeddings[name] = Embedding(None, name=name, filename=path)
                return
            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise RuntimeError(f"Couldn't identify {filename} as textual inversion embedding")

        if shared.backend == shared.Backend.DIFFUSERS:
            return emb

        vec = emb.detach().to(devices.device, dtype=torch.float32)
        # name = data.get('name', name)
        embedding = Embedding(vec=vec, name=name, filename=path)
        embedding.tag = data.get('name', None)
        embedding.step = data.get('step', None)
        embedding.sd_checkpoint = data.get('sd_checkpoint', None)
        embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
        embedding.vectors = vec.shape[0]
        embedding.shape = vec.shape[-1]
        if self.expected_shape == -1 or self.expected_shape == embedding.shape:
            self.register_embedding(embedding, shared.sd_model)
        else:
            self.skipped_embeddings[name] = embedding

    def load_from_dir(self, embdir):
        if sd_models.model_data.sd_model is None:
            shared.log.info('Skipping embeddings load: model not loaded')
            return
        if not os.path.isdir(embdir.path):
            return
        file_paths = list_embeddings(embdir.path)
        if shared.backend == shared.Backend.DIFFUSERS:
            self.load_diffusers_embedding(file_paths)
        else:
            for file_path in file_paths:
                try:
                    fn = os.path.basename(file_path)
                    self.load_from_file(file_path, fn)
                except Exception as e:
                    errors.display(e, f'Load embeding={fn}')
                    continue

    def load_textual_inversion_embeddings(self, force_reload=False):
        if shared.sd_model is None:
            return
        t0 = time.time()
        if not force_reload:
            need_reload = False
            for embdir in self.embedding_dirs.values():
                if embdir.has_changed():
                    need_reload = True
                    break
            if not need_reload:
                return
        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.embeddings_used.clear()
        self.expected_shape = self.get_expected_shape()
        for embdir in self.embedding_dirs.values():
            self.load_from_dir(embdir)
            embdir.update()

        # re-sort word_embeddings because load_from_dir may not load in alphabetic order.
        # using a temporary copy so we don't reinitialize self.word_embeddings in case other objects have a reference to it.
        sorted_word_embeddings = {e.name: e for e in sorted(self.word_embeddings.values(), key=lambda e: e.name.lower())}
        self.word_embeddings.clear()
        self.word_embeddings.update(sorted_word_embeddings)

        displayed_embeddings = (tuple(self.word_embeddings.keys()), tuple(self.skipped_embeddings.keys()))
        if self.previously_displayed_embeddings != displayed_embeddings:
            self.previously_displayed_embeddings = displayed_embeddings
            t1 = time.time()
            shared.log.info(f"Load embeddings: loaded={len(self.word_embeddings)} skipped={len(self.skipped_embeddings)} time={t1-t0:.2f}")


    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)
        if possible_matches is None:
            return None, None
        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)
        return None, None
