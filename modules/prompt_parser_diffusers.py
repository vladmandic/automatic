import os
import math
import time
import typing
from collections import OrderedDict
import torch
from compel.embeddings_provider import BaseTextualInversionManager, EmbeddingsProvider
from transformers import PreTrainedTokenizer
from modules import shared, prompt_parser, devices, sd_models
from modules.prompt_parser_xhinker import get_weighted_text_embeddings_sd15, get_weighted_text_embeddings_sdxl_2p, get_weighted_text_embeddings_sd3, get_weighted_text_embeddings_flux1

debug_enabled = os.environ.get('SD_PROMPT_DEBUG', None)
debug = shared.log.trace if os.environ.get('SD_PROMPT_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROMPT')
orig_encode_token_ids_to_embeddings = EmbeddingsProvider._encode_token_ids_to_embeddings # pylint: disable=protected-access
token_dict = None # used by helper get_tokens
token_type = None # used by helper get_tokens
cache = OrderedDict()
last_attention = None
embedder = None


def prompt_compatible(pipe = None):
    pipe = pipe or shared.sd_model
    if (
        'StableDiffusion' not in pipe.__class__.__name__ and
        'DemoFusion' not in pipe.__class__.__name__ and
        'StableCascade' not in pipe.__class__.__name__ and
        'Flux' not in pipe.__class__.__name__
    ):
        shared.log.warning(f"Prompt parser not supported: {pipe.__class__.__name__}")
        return False
    return True


def prepare_model(pipe = None):
    pipe = pipe or shared.sd_model
    if not hasattr(pipe, "text_encoder") and hasattr(shared.sd_model, "pipe"):
        pipe = pipe.pipe
    if not hasattr(pipe, "text_encoder"):
        return None
    elif hasattr(pipe, "maybe_free_model_hooks"):
        pipe.maybe_free_model_hooks()
        devices.torch_gc()
    return pipe


class PromptEmbedder:
    def __init__(self, prompts, negative_prompts, steps, clip_skip, p):
        t0 = time.time()
        self.prompts = prompts
        self.negative_prompts = negative_prompts
        self.batchsize = len(self.prompts)
        self.attention = last_attention
        self.allsame = self.compare_prompts()  # collapses batched prompts to single prompt if possible
        self.steps = steps
        self.clip_skip = clip_skip
        # All embeds are nested lists, outer list batch length, inner schedule length
        self.prompt_embeds = [[]] * self.batchsize
        self.positive_pooleds = [[]] * self.batchsize
        self.negative_prompt_embeds = [[]] * self.batchsize
        self.negative_pooleds = [[]] * self.batchsize
        self.positive_schedule = None
        self.negative_schedule = None
        self.scheduled_prompt = False
        if hasattr(p, 'dummy'):
            return
        earlyout = self.checkcache(p)
        if earlyout:
            return
        pipe = prepare_model(p.sd_model)
        if pipe is None:
            shared.log.error("Prompt encode: cannot find text encoder in model")
            return
        # per prompt in batch
        for batchidx, (prompt, negative_prompt) in enumerate(zip(self.prompts, self.negative_prompts)):
            self.prepare_schedule(prompt, negative_prompt)
            if self.scheduled_prompt:
                self.scheduled_encode(pipe, batchidx)
            else:
                self.encode(pipe, prompt, negative_prompt, batchidx)
        self.checkcache(p)
        debug(f"Prompt encode: time={(time.time() - t0):.3f}")

    def checkcache(self, p):
        if shared.opts.sd_textencoder_cache_size == 0:
            return False
        if self.attention != shared.opts.prompt_attention:
            debug(f"Prompt change: parser={shared.opts.prompt_attention}")
            cache.clear()
            return False

        def flatten(xss):
            return [x for xs in xss for x in xs]

        # unpack EN data in case of TE LoRA
        en_data = p.network_data
        en_data = [idx.items for item in en_data.values() for idx in item]
        effective_batch = 1 if self.allsame else self.batchsize
        key = str([self.prompts, self.negative_prompts, effective_batch, self.clip_skip, self.steps, en_data])
        item = cache.get(key)
        if not item:
            if not any(flatten(emb) for emb in [self.prompt_embeds,
                                                self.negative_prompt_embeds,
                                                self.positive_pooleds,
                                                self.negative_pooleds]):
                return False
            else:
                cache[key] = {'prompt_embeds': self.prompt_embeds,
                              'negative_prompt_embeds': self.negative_prompt_embeds,
                              'positive_pooleds': self.positive_pooleds,
                              'negative_pooleds': self.negative_pooleds,
                              }
                debug(f"Prompt cache: add={key}")
                while len(cache) > int(shared.opts.sd_textencoder_cache_size):
                    cache.popitem(last=False)
                return True
        if item:
            self.__dict__.update(cache[key])
            cache.move_to_end(key)
            if self.allsame and len(self.prompt_embeds) < self.batchsize:
                self.prompt_embeds = [self.prompt_embeds[0]] * self.batchsize
                self.positive_pooleds = [self.positive_pooleds[0]] * self.batchsize
                self.negative_prompt_embeds = [self.negative_prompt_embeds[0]] * self.batchsize
                self.negative_pooleds = [self.negative_pooleds[0]] * self.batchsize
            debug(f"Prompt cache: get={key}")
            return True

    def compare_prompts(self):
        same = (self.prompts == [self.prompts[0]] * len(self.prompts) and self.negative_prompts == [self.negative_prompts[0]] * len(self.negative_prompts))
        if same:
            self.prompts = [self.prompts[0]]
            self.negative_prompts = [self.negative_prompts[0]]
        return same

    def prepare_schedule(self, prompt, negative_prompt):
        self.positive_schedule, scheduled = get_prompt_schedule(prompt, self.steps)
        self.negative_schedule, neg_scheduled = get_prompt_schedule(negative_prompt, self.steps)
        self.scheduled_prompt = scheduled or neg_scheduled
        debug(f"Prompt schedule: positive={self.positive_schedule} negative={self.negative_schedule} scheduled={scheduled}")

    def scheduled_encode(self, pipe, batchidx):
        prompt_dict = {}  # index cache
        for i in range(max(len(self.positive_schedule), len(self.negative_schedule))):
            positive_prompt = self.positive_schedule[i % len(self.positive_schedule)]
            negative_prompt = self.negative_schedule[i % len(self.negative_schedule)]
            # skip repeated scheduled subprompts
            idx = prompt_dict.get(positive_prompt+negative_prompt)
            if idx is not None:
                self.extend_embeds(batchidx, idx)
                continue
            self.encode(pipe, positive_prompt, negative_prompt, batchidx)
            prompt_dict[positive_prompt+negative_prompt] = i

    def extend_embeds(self, batchidx, idx):  # Extends scheduled prompt via index
        if len(self.prompt_embeds[batchidx]) > 0:
            self.prompt_embeds[batchidx].append(self.prompt_embeds[batchidx][idx])
        if len(self.negative_prompt_embeds[batchidx]) > 0:
            self.negative_prompt_embeds[batchidx].append(self.negative_prompt_embeds[batchidx][idx])
        if len(self.positive_pooleds[batchidx]) > 0:
            self.positive_pooleds[batchidx].append(self.positive_pooleds[batchidx][idx])
        if len(self.negative_pooleds[batchidx]) > 0:
            self.negative_pooleds[batchidx].append(self.negative_pooleds[batchidx][idx])

    def encode(self, pipe, positive_prompt, negative_prompt, batchidx):
        global last_attention # pylint: disable=global-statement
        self.attention = shared.opts.prompt_attention
        last_attention = self.attention
        if self.attention == "xhinker":
            prompt_embed, positive_pooled, negative_embed, negative_pooled = get_xhinker_text_embeddings(pipe, positive_prompt, negative_prompt, self.clip_skip)
        else:
            prompt_embed, positive_pooled, negative_embed, negative_pooled = get_weighted_text_embeddings(pipe, positive_prompt, negative_prompt, self.clip_skip)
        if prompt_embed is not None:
            self.prompt_embeds[batchidx].append(prompt_embed)
        if negative_embed is not None:
            self.negative_prompt_embeds[batchidx].append(negative_embed)
        if positive_pooled is not None:
            self.positive_pooleds[batchidx].append(positive_pooled)
        if negative_pooled is not None:
            self.negative_pooleds[batchidx].append(negative_pooled)

        if debug_enabled:
            get_tokens(pipe, 'positive', positive_prompt)
            get_tokens(pipe, 'negative', negative_prompt)

    def __call__(self, key, step=0):
        batch = getattr(self, key)
        res = []
        for i in range(self.batchsize):
            if len(batch[i]) == 0:  # if asking for a null key, ie pooled on SD1.5
                return None
            try:
                res.append(batch[i][step])
            except IndexError:
                res.append(batch[i][0])  # if not scheduled, return default
        return torch.cat(res)


def compel_hijack(self, token_ids: torch.Tensor, attention_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    needs_hidden_states = self.returned_embeddings_type != 1
    text_encoder_output = self.text_encoder(token_ids, attention_mask, output_hidden_states=needs_hidden_states, return_dict=True)

    if not needs_hidden_states:
        return text_encoder_output.last_hidden_state
    try:
        normalized = self.returned_embeddings_type > 0
        clip_skip = math.floor(abs(self.returned_embeddings_type))
        interpolation = abs(self.returned_embeddings_type) - clip_skip
    except Exception:
        normalized = False
        clip_skip = 1
        interpolation = False
    if interpolation:
        hidden_state = (1 - interpolation) * text_encoder_output.hidden_states[-clip_skip] + interpolation * text_encoder_output.hidden_states[-(clip_skip+1)]
    else:
        hidden_state = text_encoder_output.hidden_states[-clip_skip]
    if normalized:
        hidden_state = self.text_encoder.text_model.final_layer_norm(hidden_state)
    return hidden_state


def sd3_compel_hijack(self, token_ids: torch.Tensor, attention_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    needs_hidden_states = True
    text_encoder_output = self.text_encoder(token_ids, attention_mask, output_hidden_states=needs_hidden_states, return_dict=True)
    clip_skip = int(self.returned_embeddings_type)
    hidden_state = text_encoder_output.hidden_states[-(clip_skip+1)]
    return hidden_state


def insert_parser_highjack(pipename):
    if "StableDiffusion3" in pipename:
        EmbeddingsProvider._encode_token_ids_to_embeddings = sd3_compel_hijack # pylint: disable=protected-access
        debug("Load SD3 Parser hijack")
    else:
        EmbeddingsProvider._encode_token_ids_to_embeddings = compel_hijack # pylint: disable=protected-access
        debug("Load Standard Parser hijack")


insert_parser_highjack("Initialize")


# from https://github.com/damian0815/compel/blob/main/src/compel/diffusers_textual_inversion_manager.py
class DiffusersTextualInversionManager(BaseTextualInversionManager):
    def __init__(self, pipe, tokenizer):
        self.pipe = pipe
        self.tokenizer = tokenizer
        if hasattr(self.pipe, 'embedding_db'):
            self.pipe.embedding_db.embeddings_used.clear()

    # code from
    # https://github.com/huggingface/diffusers/blob/705c592ea98ba4e288d837b9cba2767623c78603/src/diffusers/loaders.py
    def maybe_convert_prompt(self, prompt: typing.Union[str, typing.List[str]], tokenizer: PreTrainedTokenizer):
        prompts = [prompt] if not isinstance(prompt, typing.List) else prompt
        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]
        if not isinstance(prompt, typing.List):
            return prompts[0]
        return prompts

    def _maybe_convert_prompt(self, prompt: str, tokenizer: PreTrainedTokenizer):
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                if hasattr(self.pipe, 'embedding_db'):
                    self.pipe.embedding_db.embeddings_used.append(token)
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1
                prompt = prompt.replace(token, replacement)
        if hasattr(self.pipe, 'embedding_db'):
            self.pipe.embedding_db.embeddings_used = list(set(self.pipe.embedding_db.embeddings_used))
        debug(f'Prompt: convert="{prompt}"')
        return prompt

    def expand_textual_inversion_token_ids_if_necessary(self, token_ids: typing.List[int]) -> typing.List[int]:
        if len(token_ids) == 0:
            return token_ids
        prompt = self.pipe.tokenizer.decode(token_ids)
        prompt = self.maybe_convert_prompt(prompt, self.pipe.tokenizer)
        debug(f'Prompt: expand="{prompt}"')
        return self.pipe.tokenizer.encode(prompt, add_special_tokens=False)


def get_prompt_schedule(prompt, steps):
    t0 = time.time()
    temp = []
    schedule = prompt_parser.get_learned_conditioning_prompt_schedules([prompt], steps)[0]
    if all(x == schedule[0] for x in schedule):
        return [schedule[0][1]], False
    for chunk in schedule:
        for s in range(steps):
            if len(temp) < s + 1 <= chunk[0]:
                temp.append(chunk[1])
    debug(f'Prompt: schedule={temp} time={(time.time() - t0):.3f}')
    return temp, len(schedule) > 1


def get_tokens(pipe, msg, prompt):
    global token_dict, token_type # pylint: disable=global-statement
    if not shared.native:
        return 0
    if shared.sd_loaded and hasattr(pipe, 'tokenizer') and pipe.tokenizer is not None:
        if token_dict is None or token_type != shared.sd_model_type:
            token_type = shared.sd_model_type
            fn = pipe.tokenizer.name_or_path
            if fn.endswith('tokenizer'):
                fn = os.path.join(pipe.tokenizer.name_or_path, 'vocab.json')
            else:
                fn = os.path.join(pipe.tokenizer.name_or_path, 'tokenizer', 'vocab.json')
            token_dict = shared.readfile(fn, silent=True)
            for k, v in pipe.tokenizer.added_tokens_decoder.items():
                token_dict[str(v)] = k
            shared.log.debug(f'Tokenizer: words={len(token_dict)} file="{fn}"')
        has_bos_token = pipe.tokenizer.bos_token_id is not None
        has_eos_token = pipe.tokenizer.eos_token_id is not None
        ids = pipe.tokenizer(prompt)
        ids = getattr(ids, 'input_ids', [])
        tokens = []
        for i in ids:
            try:
                key = list(token_dict.keys())[list(token_dict.values()).index(i)]
                tokens.append(key)
            except Exception:
                tokens.append(f'UNK_{i}')
        token_count = len(ids) - int(has_bos_token) - int(has_eos_token)
        debug(f'Prompt tokenizer: type={msg} tokens={token_count} {tokens}')
    return token_count


def normalize_prompt(pairs: list):
    num_words = 0
    total_weight = 0
    for section in pairs:
        words = len(section[0].split())
        if section[1] == -1: # control tokens
            continue
        num_words += words
        total_weight += section[1] * words
    avg_weight = round(100 * total_weight / num_words) / 100 if num_words > 0 else 1
    debug(f'Prompt stats: words={num_words} weight={avg_weight}')
    for section in pairs:
        section[1] = section[1] / avg_weight if section[1] != -1 else -1 # skip control tokens
    debug(f'Prompt normalized: {pairs}')
    return pairs


def get_prompts_with_weights(pipe, prompt: str):
    t0 = time.time()
    manager = DiffusersTextualInversionManager(pipe, pipe.tokenizer or pipe.tokenizer_2)
    prompt = manager.maybe_convert_prompt(prompt, pipe.tokenizer or pipe.tokenizer_2)
    texts_and_weights = prompt_parser.parse_prompt_attention(prompt)
    if shared.opts.prompt_mean_norm:
        texts_and_weights = normalize_prompt(texts_and_weights)
    texts, text_weights = zip(*texts_and_weights)
    if debug_enabled:
        all_tokens = 0
        for text in texts:
            tokens = get_tokens(pipe, 'section', text)
            all_tokens += tokens
        debug(f'Prompt tokenizer: parser={shared.opts.prompt_attention} tokens={all_tokens}')
    debug(f'Prompt: weights={texts_and_weights} time={(time.time() - t0):.3f}')
    return texts, text_weights


def prepare_embedding_providers(pipe, clip_skip) -> list[EmbeddingsProvider]:
    device = devices.device
    embeddings_providers = []
    if 'StableCascade' in pipe.__class__.__name__:
        embedding_type = -(clip_skip)
    elif 'XL' in pipe.__class__.__name__:
        embedding_type = -(clip_skip + 1)
    else:
        embedding_type = clip_skip
    embedding_args = {
        'truncate': False,
        'returned_embeddings_type': embedding_type,
        'device': device,
        'dtype_for_device_getter': lambda device: devices.dtype,
    }
    if getattr(pipe, "prior_pipe", None) is not None and getattr(pipe.prior_pipe, "tokenizer", None) is not None and getattr(pipe.prior_pipe, "text_encoder", None) is not None:
        provider = EmbeddingsProvider(padding_attention_mask_value=0, tokenizer=pipe.prior_pipe.tokenizer, text_encoder=pipe.prior_pipe.text_encoder, **embedding_args)
        embeddings_providers.append(provider)
        no_mask_provider = EmbeddingsProvider(padding_attention_mask_value=1 if "sote" in pipe.sd_checkpoint_info.name.lower() else 0, tokenizer=pipe.prior_pipe.tokenizer, text_encoder=pipe.prior_pipe.text_encoder, **embedding_args)
        embeddings_providers.append(no_mask_provider)
    elif getattr(pipe, "tokenizer", None) is not None and getattr(pipe, "text_encoder", None) is not None:
        if pipe.text_encoder.__class__.__name__.startswith('CLIP'):
            sd_models.move_model(pipe.text_encoder, devices.device, force=True)
        provider = EmbeddingsProvider(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, **embedding_args)
        embeddings_providers.append(provider)
    if getattr(pipe, "tokenizer_2", None) is not None and getattr(pipe, "text_encoder_2", None) is not None:
        if pipe.text_encoder_2.__class__.__name__.startswith('CLIP'):
            sd_models.move_model(pipe.text_encoder_2, devices.device, force=True)
        provider = EmbeddingsProvider(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2, **embedding_args)
        embeddings_providers.append(provider)
    if getattr(pipe, "tokenizer_3", None) is not None and getattr(pipe, "text_encoder_3", None) is not None:
        if pipe.text_encoder_3.__class__.__name__.startswith('CLIP'):
            sd_models.move_model(pipe.text_encoder_3, devices.device, force=True)
        provider = EmbeddingsProvider(tokenizer=pipe.tokenizer_3, text_encoder=pipe.text_encoder_3, **embedding_args)
        embeddings_providers.append(provider)
    return embeddings_providers


def pad_to_same_length(pipe, embeds, empty_embedding_providers=None):
    if not hasattr(pipe, 'encode_prompt') and 'StableCascade' not in pipe.__class__.__name__:
        return embeds
    device = devices.device
    if shared.opts.diffusers_zeros_prompt_pad or 'StableDiffusion3' in pipe.__class__.__name__:
        empty_embed = [torch.zeros((1, 77, embeds[0].shape[2]), device=device, dtype=embeds[0].dtype)]
    else:
        try:
            if 'StableCascade' in pipe.__class__.__name__:
                empty_embed = empty_embedding_providers[0].get_embeddings_for_weighted_prompt_fragments(text_batch=[[""]], fragment_weights_batch=[[1]], should_return_tokens=False, device=device)
                empty_embed = [empty_embed]
            else:
                empty_embed = pipe.encode_prompt("")
        except TypeError:  # SD1.5
            empty_embed = pipe.encode_prompt("", device, 1, False)
    max_token_count = max([embed.shape[1] for embed in embeds])
    repeats = max_token_count - min([embed.shape[1] for embed in embeds])
    empty_batched = empty_embed[0].to(embeds[0].device).repeat(embeds[0].shape[0], repeats // empty_embed[0].shape[1], 1)
    for i, embed in enumerate(embeds):
        if embed.shape[1] < max_token_count:
            embed = torch.cat([embed, empty_batched], dim=1)
            embeds[i] = embed
    return embeds


def split_prompts(pipe, prompt, SD3 = False):
    if prompt.find("TE2:") != -1:
        prompt, prompt2 = prompt.split("TE2:")
    else:
        prompt2 = prompt

    if prompt.find("TE3:") != -1:
        prompt, prompt3 = prompt.split("TE3:")
    elif prompt2.find("TE3:") != -1:
        prompt2, prompt3 = prompt2.split("TE3:")
    else:
        prompt3 = prompt

    prompt = prompt.strip()
    prompt2 = " " if prompt2.strip() == "" else prompt2.strip()
    prompt3 = " " if prompt3.strip() == "" else prompt3.strip()

    if SD3 and prompt3 != " ":
        ps, _ws = get_prompts_with_weights(pipe, prompt3)
        prompt3 = " ".join(ps)
    return prompt, prompt2, prompt3


def get_weighted_text_embeddings(pipe, prompt: str = "", neg_prompt: str = "", clip_skip: int = None):
    device = devices.device
    SD3 = hasattr(pipe, 'text_encoder_3')
    prompt, prompt_2, prompt_3 = split_prompts(pipe, prompt, SD3)
    neg_prompt, neg_prompt_2, neg_prompt_3 = split_prompts(pipe, neg_prompt, SD3)

    if prompt != prompt_2:
        ps = [get_prompts_with_weights(pipe, p) for p in [prompt, prompt_2]]
        ns = [get_prompts_with_weights(pipe, p) for p in [neg_prompt, neg_prompt_2]]
    else:
        ps = 2 * [get_prompts_with_weights(pipe, prompt)]
        ns = 2 * [get_prompts_with_weights(pipe, neg_prompt)]

    positives, positive_weights = zip(*ps)
    negatives, negative_weights = zip(*ns)
    if hasattr(pipe, "tokenizer_2") and not hasattr(pipe, "tokenizer"):
        positives.pop(0)
        positive_weights.pop(0)
        negatives.pop(0)
        negative_weights.pop(0)

    if "Flux" in pipe.__class__.__name__: # clip is only used for the pooled embeds
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(prompt=prompt, prompt_2=prompt_2, device=device, num_images_per_prompt=1)
        return prompt_embeds, pooled_prompt_embeds, None, None # no negative support

    embedding_providers = prepare_embedding_providers(pipe, clip_skip)
    empty_embedding_providers = None
    if 'StableCascade' in pipe.__class__.__name__:
        empty_embedding_providers = [embedding_providers[1]]
        embedding_providers = [embedding_providers[0]]

    prompt_embeds = []
    negative_prompt_embeds = []
    pooled_prompt_embeds = []
    negative_pooled_prompt_embeds = []
    for i in range(len(embedding_providers)):
        if i >= len(positives): # te may be missing/unloaded
            break
        t0 = time.time()
        text = list(positives[i])
        weights = list(positive_weights[i])
        text.append('BREAK')
        weights.append(-1)
        provider_embed = []
        ptokens = 0
        while 'BREAK' in text:
            pos = text.index('BREAK')
            debug(f'Prompt: section="{text[:pos]}" len={len(text[:pos])} weights={weights[:pos]}')
            if len(text[:pos]) > 0:
                embed, ptokens = embedding_providers[i].get_embeddings_for_weighted_prompt_fragments(text_batch=[text[:pos]], fragment_weights_batch=[weights[:pos]], device=device, should_return_tokens=True)
                provider_embed.append(embed)
            text = text[pos + 1:]
            weights = weights[pos + 1:]
        prompt_embeds.append(torch.cat(provider_embed, dim=1))
        # negative prompt has no keywords
        embed, ntokens = embedding_providers[i].get_embeddings_for_weighted_prompt_fragments(text_batch=[negatives[i]], fragment_weights_batch=[negative_weights[i]], device=device, should_return_tokens=True)
        negative_prompt_embeds.append(embed)
        debug(f'Prompt: unpadded={prompt_embeds[0].shape} TE{i+1} ptokens={torch.count_nonzero(ptokens)} ntokens={torch.count_nonzero(ntokens)} time={(time.time() - t0):.3f}')
    if SD3:
        t0 = time.time()
        pooled_prompt_embeds.append(embedding_providers[0].get_pooled_embeddings(texts=positives[0] if len(positives[0]) == 1 else [" ".join(positives[0])], device=device))
        pooled_prompt_embeds.append(embedding_providers[1].get_pooled_embeddings(texts=positives[-1] if len(positives[-1]) == 1 else [" ".join(positives[-1])], device=device))
        negative_pooled_prompt_embeds.append(embedding_providers[0].get_pooled_embeddings(texts=negatives[0] if len(negatives[0]) == 1 else [" ".join(negatives[0])], device=device))
        negative_pooled_prompt_embeds.append(embedding_providers[1].get_pooled_embeddings(texts=negatives[-1] if len(negatives[-1]) == 1 else [" ".join(negatives[-1])], device=device))
        pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=-1)
        negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds, dim=-1)
        debug(f'Prompt: pooled={pooled_prompt_embeds[0].shape} time={(time.time() - t0):.3f}')
    elif prompt_embeds[-1].shape[-1] > 768:
        t0 = time.time()
        if shared.opts.diffusers_pooled == "weighted":
            pooled_prompt_embeds = embedding_providers[-1].text_encoder.text_projection(prompt_embeds[-1][
                torch.arange(prompt_embeds[-1].shape[0], device=device),
                (ptokens.to(dtype=torch.int, device=device) == 49407)
                .int()
                .argmax(dim=-1),
            ])
            negative_pooled_prompt_embeds = embedding_providers[-1].text_encoder.text_projection(negative_prompt_embeds[-1][
                torch.arange(negative_prompt_embeds[-1].shape[0], device=device),
                (ntokens.to(dtype=torch.int, device=device) == 49407)
                .int()
                .argmax(dim=-1),
            ])
        else:
            try:
                pooled_prompt_embeds = embedding_providers[-1].get_pooled_embeddings(texts=[prompt_2], device=device) if prompt_embeds[-1].shape[-1] > 768 else None
                negative_pooled_prompt_embeds = embedding_providers[-1].get_pooled_embeddings(texts=[neg_prompt_2], device=device) if negative_prompt_embeds[-1].shape[-1] > 768 else None
            except Exception:
                pooled_prompt_embeds = None
                negative_pooled_prompt_embeds = None
        debug(f'Prompt: pooled shape={pooled_prompt_embeds[0].shape if pooled_prompt_embeds is not None else None} time={(time.time() - t0):.3f}')

    prompt_embeds = torch.cat(prompt_embeds, dim=-1) if len(prompt_embeds) > 1 else prompt_embeds[0]
    negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=-1) if len(negative_prompt_embeds) > 1 else \
        negative_prompt_embeds[0]
    if pooled_prompt_embeds == []:
        pooled_prompt_embeds = None
    if negative_pooled_prompt_embeds == []:
        negative_pooled_prompt_embeds = None
    debug(f'Prompt: positive={prompt_embeds.shape if prompt_embeds is not None else None} pooled={pooled_prompt_embeds.shape if pooled_prompt_embeds is not None else None} negative={negative_prompt_embeds.shape if negative_prompt_embeds is not None else None} pooled={negative_pooled_prompt_embeds.shape if negative_pooled_prompt_embeds is not None else None}')
    if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
        [prompt_embeds, negative_prompt_embeds] = pad_to_same_length(pipe, [prompt_embeds, negative_prompt_embeds], empty_embedding_providers=empty_embedding_providers)
    if SD3:
        device = devices.device
        t5_prompt_embed = pipe._get_t5_prompt_embeds( # pylint: disable=protected-access
            prompt=prompt_3,
            num_images_per_prompt=prompt_embeds.shape[0],
            device=device,
        )
        prompt_embeds = torch.nn.functional.pad(
            prompt_embeds, (0, t5_prompt_embed.shape[-1] - prompt_embeds.shape[-1])
        ).to(device)
        prompt_embeds = torch.cat([prompt_embeds, t5_prompt_embed], dim=-2)
        t5_negative_prompt_embed = pipe._get_t5_prompt_embeds( # pylint: disable=protected-access
            prompt=neg_prompt_3,
            num_images_per_prompt=prompt_embeds.shape[0],
            device=device,
        )
        negative_prompt_embeds = torch.nn.functional.pad(
            negative_prompt_embeds, (0, t5_negative_prompt_embed.shape[-1] - negative_prompt_embeds.shape[-1])
        ).to(device)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds, t5_negative_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds


def get_xhinker_text_embeddings(pipe, prompt: str = "", neg_prompt: str = "", clip_skip: int = None):
    is_sd3 = hasattr(pipe, 'text_encoder_3')
    prompt, prompt_2, _prompt_3 = split_prompts(pipe, prompt, is_sd3)
    neg_prompt, neg_prompt_2, _neg_prompt_3 = split_prompts(pipe, neg_prompt, is_sd3)
    try:
        prompt = pipe.maybe_convert_prompt(prompt, pipe.tokenizer)
        neg_prompt = pipe.maybe_convert_prompt(neg_prompt, pipe.tokenizer)
        prompt_2 = pipe.maybe_convert_prompt(prompt_2, pipe.tokenizer_2)
        neg_prompt_2 = pipe.maybe_convert_prompt(neg_prompt_2, pipe.tokenizer_2)
    except Exception:
        pass
    prompt_embed = positive_pooled = negative_embed = negative_pooled = None

    te1_device, te2_device, te3_device = None, None, None
    if hasattr(pipe, "text_encoder") and pipe.text_encoder.device != devices.device:
        te1_device = pipe.text_encoder.device
        sd_models.move_model(pipe.text_encoder, devices.device, force=True)
    if hasattr(pipe, "text_encoder_2") and pipe.text_encoder_2.device != devices.device:
        te2_device = pipe.text_encoder_2.device
        sd_models.move_model(pipe.text_encoder_2, devices.device, force=True)
    if hasattr(pipe, "text_encoder_3") and pipe.text_encoder_3.device != devices.device:
        te3_device = pipe.text_encoder_3.device
        sd_models.move_model(pipe.text_encoder_3, devices.device, force=True)

    if 'StableDiffusion3' in pipe.__class__.__name__:
        prompt_embed, negative_embed, positive_pooled, negative_pooled = get_weighted_text_embeddings_sd3(pipe=pipe, prompt=prompt, neg_prompt=neg_prompt, use_t5_encoder=bool(pipe.text_encoder_3))
    elif 'Flux' in pipe.__class__.__name__:
        prompt_embed, positive_pooled = get_weighted_text_embeddings_flux1(pipe=pipe, prompt=prompt, prompt2=prompt_2, device=devices.device)
    elif 'XL' in pipe.__class__.__name__:
        prompt_embed, negative_embed, positive_pooled, negative_pooled = get_weighted_text_embeddings_sdxl_2p(pipe=pipe, prompt=prompt, prompt_2=prompt_2, neg_prompt=neg_prompt, neg_prompt_2=neg_prompt_2)
    else:
        prompt_embed, negative_embed = get_weighted_text_embeddings_sd15(pipe=pipe, prompt=prompt, neg_prompt=neg_prompt, clip_skip=clip_skip)

    if te1_device is not None:
        sd_models.move_model(pipe.text_encoder, te1_device, force=True)
    if te2_device is not None:
        sd_models.move_model(pipe.text_encoder_2, te1_device, force=True)
    if te3_device is not None:
        sd_models.move_model(pipe.text_encoder_3, te1_device, force=True)

    return prompt_embed, positive_pooled, negative_embed, negative_pooled
