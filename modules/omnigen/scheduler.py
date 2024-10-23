import torch
from tqdm import tqdm
from transformers.cache_utils import Cache, DynamicCache

class OmniGenScheduler:
    def __init__(self, num_steps: int=50, time_shifting_factor: int=1):
        self.num_steps = num_steps
        self.time_shift = time_shifting_factor

        t = torch.linspace(0, 1, num_steps+1)
        t = t / (t + time_shifting_factor - time_shifting_factor * t)
        self.sigma = t
    
    def crop_kv_cache(self, past_key_values, num_tokens_for_img):
        crop_past_key_values = ()
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx][:2]
            crop_past_key_values += ((key_states[..., :-(num_tokens_for_img+1), :], value_states[..., :-(num_tokens_for_img+1), :], ),)
        return crop_past_key_values
        # return DynamicCache.from_legacy_cache(crop_past_key_values)

    def crop_position_ids_for_cache(self, position_ids, num_tokens_for_img):
        if isinstance(position_ids, list):
            for i in range(len(position_ids)):
                position_ids[i] = position_ids[i][:, -(num_tokens_for_img+1):]
        else:
            position_ids = position_ids[:, -(num_tokens_for_img+1):]
        return position_ids

    def crop_attention_mask_for_cache(self, attention_mask, num_tokens_for_img):
        if isinstance(attention_mask, list):
            return [x[..., -(num_tokens_for_img+1):, :] for x in attention_mask]
        return attention_mask[..., -(num_tokens_for_img+1):, :]

    def __call__(self, z, func, model_kwargs, use_kv_cache: bool=True):
        past_key_values = None
        for i in tqdm(range(self.num_steps)):
            timesteps = torch.zeros(size=(len(z), )).to(z.device) + self.sigma[i]
            pred, temp_past_key_values = func(z, timesteps, past_key_values=past_key_values, **model_kwargs)
            sigma_next = self.sigma[i+1]
            sigma = self.sigma[i]
            z = z + (sigma_next - sigma) * pred
            if i == 0 and use_kv_cache:
                num_tokens_for_img = z.size(-1)*z.size(-2) // 4
                if isinstance(temp_past_key_values, list):
                    past_key_values = [self.crop_kv_cache(x, num_tokens_for_img) for x in temp_past_key_values]
                    model_kwargs['input_ids'] = [None] * len(temp_past_key_values)
                else:
                    past_key_values = self.crop_kv_cache(temp_past_key_values, num_tokens_for_img)
                    model_kwargs['input_ids'] = None

                model_kwargs['position_ids'] = self.crop_position_ids_for_cache(model_kwargs['position_ids'], num_tokens_for_img)
                model_kwargs['attention_mask'] = self.crop_attention_mask_for_cache(model_kwargs['attention_mask'], num_tokens_for_img)
        return z

