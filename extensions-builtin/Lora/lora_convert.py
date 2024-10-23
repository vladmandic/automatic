import os
import re
import bisect
from typing import Dict
import torch
from modules import shared


debug = os.environ.get('SD_LORA_DEBUG', None) is not None
suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}
re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}


def make_unet_conversion_map() -> Dict[str, str]:
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3 * i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))
            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3 * i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3 * i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))
            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3 * i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3 * (i + 1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))
            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3 * i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2 * j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j + 1}."
        sd_time_embed_prefix = f"time_embed.{j * 2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j + 1}."
        sd_label_embed_prefix = f"label_emb.0.{j * 2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    sd_hf_conversion_map = {sd.replace(".", "_")[:-1]: hf.replace(".", "_")[:-1] for sd, hf in unet_conversion_map}
    return sd_hf_conversion_map


class KeyConvert:
    def __init__(self):
        if not shared.native:
            self.converter = self.original
            self.is_sd2 = 'model_transformer_resblocks' in shared.sd_model.network_layer_mapping
        else:
            self.converter = self.diffusers
            self.is_sdxl = True if shared.sd_model_type == "sdxl" else False
            self.UNET_CONVERSION_MAP = make_unet_conversion_map() if self.is_sdxl else None
            self.LORA_PREFIX_UNET = "lora_unet_"
            self.LORA_PREFIX_TEXT_ENCODER = "lora_te_"
            self.OFT_PREFIX_UNET = "oft_unet_"
            # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
            self.LORA_PREFIX_TEXT_ENCODER1 = "lora_te1_"
            self.LORA_PREFIX_TEXT_ENCODER2 = "lora_te2_"

    def original(self, key):
        key = convert_diffusers_name_to_compvis(key, self.is_sd2)
        sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = shared.sd_model.network_layer_mapping.get(m.group(1), None)
        # SDXL loras seem to already have correct compvis keys, so only need to replace "lora_unet" with "diffusion_model"
        if sd_module is None and "lora_unet" in key:
            key = key.replace("lora_unet", "diffusion_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        elif sd_module is None and "lora_te1_text_model" in key:
            key = key.replace("lora_te1_text_model", "0_transformer_text_model")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)
            # some SD1 Loras also have correct compvis keys
            if sd_module is None:
                key = key.replace("lora_te1_text_model", "transformer_text_model")
                sd_module = shared.sd_model.network_layer_mapping.get(key, None)

        # SegMoE begin
        expert_key = key + "_experts_0"
        expert_module = shared.sd_model.network_layer_mapping.get(expert_key, None)
        if expert_module is not None:
            sd_module = expert_module
            key = expert_key
        if sd_module is None:
            key = key.replace("_net_", "_experts_0_net_")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        key = key if isinstance(key, list) else [key]
        sd_module = sd_module if isinstance(sd_module, list) else [sd_module]
        if "_experts_0" in key[0]:
            i = expert_module = 1
            while expert_module is not None:
                expert_key = key[0].replace("_experts_0", f"_experts_{i}")
                expert_module = shared.sd_model.network_layer_mapping.get(expert_key, None)
                if expert_module is not None:
                    key.append(expert_key)
                    sd_module.append(expert_module)
                    i += 1
        # SegMoE end
        return key, sd_module

    def diffusers(self, key):
        if self.is_sdxl:
            if "diffusion_model" in key:  # Fix NTC Slider naming error
                key = key.replace("diffusion_model", "lora_unet")
            map_keys = list(self.UNET_CONVERSION_MAP.keys())  # prefix of U-Net modules
            map_keys.sort()
            search_key = key.replace(self.LORA_PREFIX_UNET, "").replace(self.OFT_PREFIX_UNET, "").replace(self.LORA_PREFIX_TEXT_ENCODER1, "").replace(self.LORA_PREFIX_TEXT_ENCODER2, "")
            position = bisect.bisect_right(map_keys, search_key)
            map_key = map_keys[position - 1]
            if search_key.startswith(map_key):
                key = key.replace(map_key, self.UNET_CONVERSION_MAP[map_key]).replace("oft", "lora") # pylint: disable=unsubscriptable-object
        if "lycoris" in key and "transformer" in key:
            key = key.replace("lycoris", "lora_transformer")
        sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        if sd_module is None:
            sd_module = shared.sd_model.network_layer_mapping.get(key.replace("guidance", "timestep"), None)  # FLUX1 fix
        # SegMoE begin
        expert_key = key + "_experts_0"
        expert_module = shared.sd_model.network_layer_mapping.get(expert_key, None)
        if expert_module is not None:
            sd_module = expert_module
            key = expert_key
        if sd_module is None:
            key = key.replace("_net_", "_experts_0_net_")
            sd_module = shared.sd_model.network_layer_mapping.get(key, None)
        key = key if isinstance(key, list) else [key]
        sd_module = sd_module if isinstance(sd_module, list) else [sd_module]
        if "_experts_0" in key[0]:
            i = expert_module = 1
            while expert_module is not None:
                expert_key = key[0].replace("_experts_0", f"_experts_{i}")
                expert_module = shared.sd_model.network_layer_mapping.get(expert_key, None)
                if expert_module is not None:
                    key.append(expert_key)
                    sd_module.append(expert_module)
                    i += 1
        # SegMoE end
        if debug and sd_module is None:
            raise RuntimeError(f"LoRA key not found in network_layer_mapping: key={key} mapping={shared.sd_model.network_layer_mapping.keys()}")
        return key, sd_module

    def __call__(self, key):
        return self.converter(key)


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex
        r = re.match(regex, key)
        if not r:
            return False
        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []
    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'
    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'
    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"
    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"
    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"
    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"
    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"
    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"
    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"
        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"
    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"
    return key


# Taken from https://github.com/huggingface/diffusers/blob/main/src/diffusers/loaders/lora_conversion_utils.py
# Modified from 'lora_A' and 'lora_B' to 'lora_down' and 'lora_up'
# Added early exit
# The utilities under `_convert_kohya_flux_lora_to_diffusers()`
# are taken from https://github.com/kohya-ss/sd-scripts/blob/a61cf73a5cb5209c3f4d1a3688dd276a4dfd1ecb/networks/convert_flux_lora.py
# All credits go to `kohya-ss`.
def _convert_kohya_flux_lora_to_diffusers(state_dict):
    def _convert_to_ai_toolkit(sds_sd, ait_sd, sds_key, ait_key):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")

        # scale weight by alpha and dim
        rank = down_weight.shape[0]
        alpha = sds_sd.pop(sds_key + ".alpha").item()  # alpha is scalar
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

        # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        ait_sd[ait_key + ".lora_down.weight"] = down_weight * scale_down
        ait_sd[ait_key + ".lora_up.weight"] = sds_sd.pop(sds_key + ".lora_up.weight") * scale_up

    def _convert_to_ai_toolkit_cat(sds_sd, ait_sd, sds_key, ait_keys, dims=None):
        if sds_key + ".lora_down.weight" not in sds_sd:
            return
        down_weight = sds_sd.pop(sds_key + ".lora_down.weight")
        up_weight = sds_sd.pop(sds_key + ".lora_up.weight")
        sd_lora_rank = down_weight.shape[0]

        # scale weight by alpha and dim
        alpha = sds_sd.pop(sds_key + ".alpha")
        scale = alpha / sd_lora_rank

        # calculate scale_down and scale_up
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        down_weight = down_weight * scale_down
        up_weight = up_weight * scale_up

        # calculate dims if not provided
        num_splits = len(ait_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # check upweight is sparse or not
        is_sparse = False
        if sd_lora_rank % num_splits == 0:
            ait_rank = sd_lora_rank // num_splits
            is_sparse = True
            i = 0
            for j in range(len(dims)):
                for k in range(len(dims)):
                    if j == k:
                        continue
                    is_sparse = is_sparse and torch.all(
                        up_weight[i : i + dims[j], k * ait_rank : (k + 1) * ait_rank] == 0
                    )
                i += dims[j]
            # if is_sparse:
            #     print(f"weight is sparse: {sds_key}")

        # make ai-toolkit weight
        ait_down_keys = [k + ".lora_down.weight" for k in ait_keys]
        ait_up_keys = [k + ".lora_up.weight" for k in ait_keys]
        if not is_sparse:
            # down_weight is copied to each split
            ait_sd.update({k: down_weight for k in ait_down_keys})

            # up_weight is split to each split
            ait_sd.update({k: v for k, v in zip(ait_up_keys, torch.split(up_weight, dims, dim=0))})  # noqa: C416 # pylint: disable=unnecessary-comprehension
        else:
            # down_weight is chunked to each split
            ait_sd.update({k: v for k, v in zip(ait_down_keys, torch.chunk(down_weight, num_splits, dim=0))})  # noqa: C416 # pylint: disable=unnecessary-comprehension

            # up_weight is sparse: only non-zero values are copied to each split
            i = 0
            for j in range(len(dims)):
                ait_sd[ait_up_keys[j]] = up_weight[i : i + dims[j], j * ait_rank : (j + 1) * ait_rank].contiguous()
                i += dims[j]

    def _convert_sd_scripts_to_ai_toolkit(sds_sd):
        ait_sd = {}
        for i in range(19):
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_0",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mlp_2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_img_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_0",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mlp_2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_double_blocks_{i}_txt_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            _convert_to_ai_toolkit_cat(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            _convert_to_ai_toolkit(
                sds_sd,
                ait_sd,
                f"lora_unet_single_blocks_{i}_modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        if len(sds_sd) > 0:
            return None

        return ait_sd

    return _convert_sd_scripts_to_ai_toolkit(state_dict)
