import os
import math
import time
import typing
import torch
from compel.embeddings_provider import BaseTextualInversionManager, EmbeddingsProvider
from transformers import PreTrainedTokenizer
from modules import shared, prompt_parser, devices


debug_enabled = os.environ.get('SD_PROMPT_DEBUG', None)
debug = shared.log.trace if os.environ.get('SD_PROMPT_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROMPT')
orig_encode_token_ids_to_embeddings = EmbeddingsProvider._encode_token_ids_to_embeddings # pylint: disable=protected-access
token_dict = None
token_type = None
cache = {}
cache_type = None


def compel_hijack(self, token_ids: torch.Tensor,
                  attention_mask: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
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


EmbeddingsProvider._encode_token_ids_to_embeddings = compel_hijack # pylint: disable=protected-access


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


def get_tokens(msg, prompt):
    global token_dict, token_type # pylint: disable=global-statement
    if shared.backend != shared.Backend.DIFFUSERS:
        return
    if shared.sd_loaded and hasattr(shared.sd_model, 'tokenizer') and shared.sd_model.tokenizer is not None:
        if token_dict is None or token_type != shared.sd_model_type:
            token_type = shared.sd_model_type
            fn = os.path.join(shared.sd_model.tokenizer.name_or_path, 'tokenizer', 'vocab.json')
            token_dict = shared.readfile(fn, silent=True)
            for k, v in shared.sd_model.tokenizer.added_tokens_decoder.items():
                token_dict[str(v)] = k
            shared.log.debug(f'Tokenizer: words={len(token_dict)} file="{fn}"')
        has_bos_token = shared.sd_model.tokenizer.bos_token_id is not None
        has_eos_token = shared.sd_model.tokenizer.eos_token_id is not None
        ids = shared.sd_model.tokenizer(prompt)
        ids = getattr(ids, 'input_ids', [])
        tokens = []
        for i in ids:
            try:
                key = list(token_dict.keys())[list(token_dict.values()).index(i)]
                tokens.append(key)
            except Exception:
                tokens.append(f'UNK_{i}')
        token_count = len(ids) - int(has_bos_token) - int(has_eos_token)
        shared.log.trace(f'Prompt tokenizer: type={msg} tokens={token_count} {tokens}')


def encode_prompts(pipe, p, prompts: list, negative_prompts: list, steps: int, clip_skip: typing.Optional[int] = None):
    if 'StableDiffusion' not in pipe.__class__.__name__ and 'DemoFusion' not in pipe.__class__.__name__ and 'StableCascade' not in pipe.__class__.__name__:
        shared.log.warning(f"Prompt parser not supported: {pipe.__class__.__name__}")
        return
    elif prompts == cache.get('prompts', None) and negative_prompts == cache.get('negative_prompts', None) and clip_skip == cache.get('clip_skip', None) and cache.get('model_type', None) == shared.sd_model_type:
        p.prompt_embeds = cache.get('prompt_embeds', None)
        p.positive_pooleds = cache.get('positive_pooleds', None)
        p.negative_embeds = cache.get('negative_embeds', None)
        p.negative_pooleds = cache.get('negative_pooleds', None)
        p.scheduled_prompt = cache.get('scheduled_prompt', None)
        debug("Prompt encode: cached")
        return
    else:
        t0 = time.time()
        positive_schedule, scheduled = get_prompt_schedule(prompts[0], steps)
        negative_schedule, neg_scheduled = get_prompt_schedule(negative_prompts[0], steps)
        p.scheduled_prompt = scheduled or neg_scheduled
        p.prompt_embeds = []
        p.positive_pooleds = []
        p.negative_embeds = []
        p.negative_pooleds = []

        for i in range(max(len(positive_schedule), len(negative_schedule))):
            positive_prompt = positive_schedule[i % len(positive_schedule)]
            negative_prompt = negative_schedule[i % len(negative_schedule)]
            if cache.get('model_type', None) != shared.sd_model_type:
                cache[positive_prompt + negative_prompt] = None
                results = None
            elif clip_skip == cache.get('clip_skip', None):
                results = cache.get(positive_prompt + negative_prompt, None)
            else:
                results = None

            if results is None:
                results = get_weighted_text_embeddings(pipe, positive_prompt, negative_prompt, clip_skip)
                cache[positive_prompt + negative_prompt] = results

            prompt_embed, positive_pooled, negative_embed, negative_pooled = results
            if prompt_embed is not None:
                p.prompt_embeds.append(torch.cat([prompt_embed] * len(prompts), dim=0))
                cache['prompt_embeds'] = p.prompt_embeds
            if negative_embed is not None:
                p.negative_embeds.append(torch.cat([negative_embed] * len(negative_prompts), dim=0))
                cache['negative_embeds'] = p.negative_embeds
            if positive_pooled is not None:
                p.positive_pooleds.append(torch.cat([positive_pooled] * len(prompts), dim=0))
                cache['positive_pooleds'] = p.positive_pooleds
            if negative_pooled is not None:
                p.negative_pooleds.append(torch.cat([negative_pooled] * len(negative_prompts), dim=0))
                cache['negative_pooleds'] = p.negative_pooleds

        cache['prompts'] = prompts
        cache['negative_prompts'] = negative_prompts
        cache['clip_skip'] = clip_skip
        cache['model_type'] = shared.sd_model_type
        if debug_enabled:
            get_tokens('positive', prompts[0])
            get_tokens('negative', negative_prompts[0])
        debug(f"Prompt encode: time={(time.time() - t0):.3f}")
        return


def get_prompts_with_weights(prompt: str):
    t0 = time.time()
    manager = DiffusersTextualInversionManager(shared.sd_model,
                                               shared.sd_model.tokenizer or shared.sd_model.tokenizer_2)
    prompt = manager.maybe_convert_prompt(prompt, shared.sd_model.tokenizer or shared.sd_model.tokenizer_2)
    texts_and_weights = prompt_parser.parse_prompt_attention(prompt)
    texts, text_weights = zip(*texts_and_weights)
    debug(f'Prompt: weights={texts_and_weights} time={(time.time() - t0):.3f}')
    return texts, text_weights


def prepare_embedding_providers(pipe, clip_skip) -> list[EmbeddingsProvider]:
    device = pipe.device if str(pipe.device) != 'meta' else devices.device
    embeddings_providers = []
    if 'StableCascade' in pipe.__class__.__name__:
        embedding_type = -(clip_skip)
    elif 'XL' in pipe.__class__.__name__:
        embedding_type = -(clip_skip + 1)
    else:
        embedding_type = clip_skip
    if getattr(pipe, "prior_pipe", None) is not None and getattr(pipe.prior_pipe, "tokenizer", None) is not None and getattr(pipe.prior_pipe, "text_encoder", None) is not None:
        provider = EmbeddingsProvider(tokenizer=pipe.prior_pipe.tokenizer, text_encoder=pipe.prior_pipe.text_encoder, truncate=False, returned_embeddings_type=embedding_type, device=device)
        embeddings_providers.append(provider)
    elif getattr(pipe, "tokenizer", None) is not None and getattr(pipe, "text_encoder", None) is not None:
        provider = EmbeddingsProvider(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate=False, returned_embeddings_type=embedding_type, device=device)
        embeddings_providers.append(provider)
    if getattr(pipe, "tokenizer_2", None) is not None and getattr(pipe, "text_encoder_2", None) is not None:
        provider = EmbeddingsProvider(tokenizer=pipe.tokenizer_2, text_encoder=pipe.text_encoder_2, truncate=False, returned_embeddings_type=embedding_type, device=device)
        embeddings_providers.append(provider)
    return embeddings_providers


def pad_to_same_length(pipe, embeds):
    if not hasattr(pipe, 'encode_prompt') and not (hasattr(pipe, "prior_pipe") and hasattr(pipe.prior_pipe, "encode_prompt")):
        return embeds
    device = pipe.device if str(pipe.device) != 'meta' else devices.device
    try:
        if getattr(pipe, "prior_pipe", None) and getattr(pipe.prior_pipe, "text_encoder", None) is not None: # Cascade
            empty_embed = pipe.prior_pipe.encode_prompt(device, 1, 1, False, "")
            empty_embed = [torch.zeros(empty_embed[0].shape, device=empty_embed[0].device, dtype=empty_embed[0].dtype)]
        else: # SDXL
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


def get_weighted_text_embeddings(pipe, prompt: str = "", neg_prompt: str = "", clip_skip: int = None):
    device = pipe.device if str(pipe.device) != 'meta' else devices.device
    prompt_split = prompt.split("TE2:")
    prompt = prompt_split[0]
    prompt_2 = prompt_split[-1]
    neg_prompt_split = neg_prompt.split("TE2:")
    neg_prompt_2 = neg_prompt_split[-1]
    neg_prompt = neg_prompt_split[0]

    if prompt != prompt_2:
        ps = [get_prompts_with_weights(p) for p in [prompt, prompt_2]]
        ns = [get_prompts_with_weights(p) for p in [neg_prompt, neg_prompt_2]]
    else:
        ps = 2 * [get_prompts_with_weights(prompt)]
        ns = 2 * [get_prompts_with_weights(neg_prompt)]

    positives, positive_weights = zip(*ps)
    negatives, negative_weights = zip(*ns)
    if hasattr(pipe, "tokenizer_2") and not hasattr(pipe, "tokenizer"):
        positives.pop(0)
        positive_weights.pop(0)
        negatives.pop(0)
        negative_weights.pop(0)

    embedding_providers = prepare_embedding_providers(pipe, clip_skip)
    prompt_embeds = []
    negative_prompt_embeds = []
    pooled_prompt_embeds = None
    negative_pooled_prompt_embeds = None
    for i in range(len(embedding_providers)):
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
                embed, ptokens = embedding_providers[i].get_embeddings_for_weighted_prompt_fragments(
                    text_batch=[text[:pos]], fragment_weights_batch=[weights[:pos]], device=device,
                    should_return_tokens=True)
                provider_embed.append(embed)
            text = text[pos + 1:]
            weights = weights[pos + 1:]
        prompt_embeds.append(torch.cat(provider_embed, dim=1))
        # negative prompt has no keywords
        embed, ntokens = embedding_providers[i].get_embeddings_for_weighted_prompt_fragments(text_batch=[negatives[i]], fragment_weights_batch=[negative_weights[i]], device=device, should_return_tokens=True)
        negative_prompt_embeds.append(embed)
        debug(f'Prompt: unpadded shape={prompt_embeds[0].shape} TE{i+1} ptokens={torch.count_nonzero(ptokens)} ntokens={torch.count_nonzero(ntokens)} time={(time.time() - t0):.3f}')

    if prompt_embeds[-1].shape[-1] > 768:
        t0 = time.time()
        if shared.opts.diffusers_pooled == "weighted":
            pooled_prompt_embeds = prompt_embeds[-1][
                torch.arange(prompt_embeds[-1].shape[0], device=device),
                (ptokens.to(dtype=torch.int, device=device) == 49407)
                .int()
                .argmax(dim=-1),
            ]
            negative_pooled_prompt_embeds = negative_prompt_embeds[-1][
                torch.arange(negative_prompt_embeds[-1].shape[0], device=device),
                (ntokens.to(dtype=torch.int, device=device) == 49407)
                .int()
                .argmax(dim=-1),
            ]
        else:
            try:
                pooled_prompt_embeds = embedding_providers[-1].get_pooled_embeddings(texts=[prompt_2], device=device) if prompt_embeds[-1].shape[-1] > 768 else None
                negative_pooled_prompt_embeds = embedding_providers[-1].get_pooled_embeddings(texts=[neg_prompt_2], device=device) if negative_prompt_embeds[-1].shape[-1] > 768 else None
            except Exception:
                pooled_prompt_embeds = None
                negative_pooled_prompt_embeds = None
        debug(f'Prompt: pooled shape={pooled_prompt_embeds[0].shape} time={(time.time() - t0):.3f}')

    prompt_embeds = torch.cat(prompt_embeds, dim=-1) if len(prompt_embeds) > 1 else prompt_embeds[0]
    negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=-1) if len(negative_prompt_embeds) > 1 else \
        negative_prompt_embeds[0]
    debug(f'Prompt: positive={prompt_embeds.shape if prompt_embeds is not None else None} pooled={pooled_prompt_embeds.shape if pooled_prompt_embeds is not None else None} negative={negative_prompt_embeds.shape if negative_prompt_embeds is not None else None} pooled={negative_pooled_prompt_embeds.shape if negative_pooled_prompt_embeds is not None else None}')
    if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
        [prompt_embeds, negative_prompt_embeds] = pad_to_same_length(pipe, [prompt_embeds, negative_prompt_embeds])
    return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds
