
from functools import cache, wraps
import torch
from diffusers.utils import USE_PEFT_BACKEND # pylint: disable=unused-import
from modules import shared, devices


@cache
def find_slice_size(slice_size, slice_block_size, slice_rate=4):
    while (slice_size * slice_block_size) > slice_rate:
        slice_size = slice_size // 2
        if slice_size <= 1:
            slice_size = 1
            break
    return slice_size

@cache
def find_slice_sizes(query_shape, query_element_size, slice_rate=4):
    if len(query_shape) == 3:
        batch_size_attention, query_tokens, shape_three = query_shape
        shape_four = 1
    else:
        batch_size_attention, query_tokens, shape_three, shape_four = query_shape

    slice_block_size = query_tokens * shape_three * shape_four / 1024 / 1024 * query_element_size
    block_size = batch_size_attention * slice_block_size

    split_slice_size = batch_size_attention
    split_2_slice_size = query_tokens
    split_3_slice_size = shape_three

    do_split = False
    do_split_2 = False
    do_split_3 = False

    if block_size > slice_rate:
        do_split = True
        split_slice_size = find_slice_size(split_slice_size, slice_block_size, slice_rate=slice_rate)
        if split_slice_size * slice_block_size > slice_rate:
            slice_2_block_size = split_slice_size * shape_three * shape_four / 1024 / 1024 * query_element_size
            do_split_2 = True
            split_2_slice_size = find_slice_size(split_2_slice_size, slice_2_block_size, slice_rate=slice_rate)
            if split_2_slice_size * slice_2_block_size > slice_rate:
                slice_3_block_size = split_slice_size * split_2_slice_size * shape_four / 1024 / 1024 * query_element_size
                do_split_3 = True
                split_3_slice_size = find_slice_size(split_3_slice_size, slice_3_block_size, slice_rate=slice_rate)

    return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size

if devices.sdpa_pre_dyanmic_atten is None:
    devices.sdpa_pre_dyanmic_atten = torch.nn.functional.scaled_dot_product_attention
@wraps(devices.sdpa_pre_dyanmic_atten)
def sliced_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_slice_sizes(query.shape, query.element_size(), slice_rate=shared.opts.dynamic_attention_slice_rate)

    # Slice SDPA
    if do_split:
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
        if attn_mask is not None and attn_mask.shape[:-1] != query.shape[:-1]:
            if len(query.shape) == 4:
                attn_mask = attn_mask.expand((query.shape[0], query.shape[1], query.shape[2], key.shape[-2]))
            else:
                attn_mask = attn_mask.expand((query.shape[0], query.shape[1], key.shape[-2]))
        for i in range(batch_size_attention // split_slice_size):
            start_idx = i * split_slice_size
            end_idx = (i + 1) * split_slice_size
            if do_split_2:
                for i2 in range(query_tokens // split_2_slice_size): # pylint: disable=invalid-name
                    start_idx_2 = i2 * split_2_slice_size
                    end_idx_2 = (i2 + 1) * split_2_slice_size
                    if do_split_3:
                        for i3 in range(shape_three // split_3_slice_size): # pylint: disable=invalid-name
                            start_idx_3 = i3 * split_3_slice_size
                            end_idx_3 = (i3 + 1) * split_3_slice_size
                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = devices.sdpa_pre_dyanmic_atten(
                                query[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                key[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                value[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3],
                                attn_mask=attn_mask[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] if attn_mask is not None else attn_mask,
                                dropout_p=dropout_p, is_causal=is_causal, **kwargs
                            )
                    else:
                        hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = devices.sdpa_pre_dyanmic_atten(
                            query[start_idx:end_idx, start_idx_2:end_idx_2],
                            key[start_idx:end_idx, start_idx_2:end_idx_2],
                            value[start_idx:end_idx, start_idx_2:end_idx_2],
                            attn_mask=attn_mask[start_idx:end_idx, start_idx_2:end_idx_2] if attn_mask is not None else attn_mask,
                            dropout_p=dropout_p, is_causal=is_causal, **kwargs
                        )
            else:
                hidden_states[start_idx:end_idx] = devices.sdpa_pre_dyanmic_atten(
                    query[start_idx:end_idx],
                    key[start_idx:end_idx],
                    value[start_idx:end_idx],
                    attn_mask=attn_mask[start_idx:end_idx] if attn_mask is not None else attn_mask,
                    dropout_p=dropout_p, is_causal=is_causal, **kwargs
                )
        if devices.backend != "directml":
            getattr(torch, query.device.type).synchronize()
    else:
        return devices.sdpa_pre_dyanmic_atten(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, **kwargs)
    return hidden_states


class DynamicAttnProcessorBMM:
    r"""
    dynamically slices attention queries in order to keep them under the slice rate
    slicing will not get triggered if the query size is smaller than the slice rate to gain performance

    slice rate is in GB
    based on AttnProcessor V1
    """

    def __call__(self, attn, hidden_states: torch.Tensor, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs) -> torch.Tensor: # pylint: disable=too-many-statements, too-many-locals, too-many-branches, keyword-arg-before-vararg

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        ####################################################################
        # Slicing parts:
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
        do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_slice_sizes(query.shape, query.element_size(), slice_rate=shared.opts.dynamic_attention_slice_rate)

        if do_split:
            for i in range(batch_size_attention // split_slice_size):
                start_idx = i * split_slice_size
                end_idx = (i + 1) * split_slice_size
                if do_split_2:
                    for i2 in range(query_tokens // split_2_slice_size): # pylint: disable=invalid-name
                        start_idx_2 = i2 * split_2_slice_size
                        end_idx_2 = (i2 + 1) * split_2_slice_size
                        if do_split_3:
                            for i3 in range(shape_three // split_3_slice_size): # pylint: disable=invalid-name
                                start_idx_3 = i3 * split_3_slice_size
                                end_idx_3 = (i3 + 1) * split_3_slice_size

                                query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                                key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3]
                                attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] if attention_mask is not None else None

                                attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                                del query_slice
                                del key_slice
                                del attn_mask_slice
                                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3])

                                hidden_states[start_idx:end_idx, start_idx_2:end_idx_2, start_idx_3:end_idx_3] = attn_slice
                                del attn_slice
                        else:
                            query_slice = query[start_idx:end_idx, start_idx_2:end_idx_2]
                            key_slice = key[start_idx:end_idx, start_idx_2:end_idx_2]
                            attn_mask_slice = attention_mask[start_idx:end_idx, start_idx_2:end_idx_2] if attention_mask is not None else None

                            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                            del query_slice
                            del key_slice
                            del attn_mask_slice
                            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx, start_idx_2:end_idx_2])

                            hidden_states[start_idx:end_idx, start_idx_2:end_idx_2] = attn_slice
                            del attn_slice
                else:
                    query_slice = query[start_idx:end_idx]
                    key_slice = key[start_idx:end_idx]
                    attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

                    attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
                    del query_slice
                    del key_slice
                    del attn_mask_slice
                    attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

                    hidden_states[start_idx:end_idx] = attn_slice
                    del attn_slice
            if devices.backend != "directml":
                getattr(torch, query.device.type).synchronize()
        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
        ####################################################################
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
