import os
import time
import torch
from safetensors.torch import save_file
import gradio as gr
from modules import shared, devices
from modules.ui_common import create_refresh_button


class SVDHandler:
    def __init__(self, maxrank=0, rank_ratio=1):
        self.network_name: str = None
        self.U: torch.Tensor = None
        self.S: torch.Tensor = None
        self.Vh: torch.Tensor = None
        self.maxrank: int = maxrank
        self.rank_ratio: float = rank_ratio
        self.rank: int = 0
        self.out_size: int = None
        self.in_size: int = None
        self.kernel_size: tuple[int, int] = None
        self.conv2d: bool = False

    def decompose(self, weight, backupweight):
        self.conv2d = len(weight.size()) == 4
        self.kernel_size = None if not self.conv2d else weight.size()[2:4]
        self.out_size, self.in_size = weight.size()[0:2]
        diffweight = weight.clone().to(devices.device)
        diffweight -= backupweight.to(devices.device)
        if self.conv2d:
            if self.conv2d and self.kernel_size != (1, 1):
                diffweight = diffweight.flatten(start_dim=1)
            else:
                diffweight = diffweight.squeeze()
        self.U, self.S, self.Vh = torch.svd_lowrank(diffweight.to(device=devices.device, dtype=torch.float), self.maxrank, 2)
        # del diffweight
        self.U = self.U.to(device=devices.cpu, dtype=torch.bfloat16)
        self.S = self.S.to(device=devices.cpu, dtype=torch.bfloat16)
        self.Vh = self.Vh.t().to(device=devices.cpu, dtype=torch.bfloat16)  # svd_lowrank outputs a transposed matrix

    def findrank(self):
        if self.rank_ratio < 1:
            S_squared = self.S.pow(2)
            S_fro_sq = float(torch.sum(S_squared))
            sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
            index = int(torch.searchsorted(sum_S_squared, self.rank_ratio ** 2)) + 1
            index = max(1, min(index, len(self.S) - 1))
            self.rank = index
            if self.maxrank > 0:
                self.rank = min(self.rank, self.maxrank)
        else:
            self.rank = min(self.in_size, self.out_size, self.maxrank)

    def makeweights(self):
        self.findrank()
        up = self.U[:, :self.rank] @ torch.diag(self.S[:self.rank])
        down = self.Vh[:self.rank, :]
        if self.conv2d and self.kernel_size is not None:
            up = up.reshape(self.out_size, self.rank, 1, 1)
            down = down.reshape(self.rank, self.in_size, self.kernel_size[0], self.kernel_size[1]) # pylint: disable=unsubscriptable-object
        return_dict = {f'{self.network_name}.lora_up.weight': up.contiguous(),
                       f'{self.network_name}.lora_down.weight': down.contiguous(),
                       f'{self.network_name}.alpha': torch.tensor(down.shape[0]),
                       }
        return return_dict


def loaded_lora():
    if not shared.sd_loaded:
        return ""
    loaded = set()
    if hasattr(shared.sd_model, 'unet'):
        for _name, module in shared.sd_model.unet.named_modules():
            current = getattr(module, "network_current_names", None)
            if current is not None:
                current = [item[0] for item in current]
                loaded.update(current)
    return ", ".join(list(loaded))


def make_lora(filename, maxrank, auto_rank, rank_ratio):
    if not shared.sd_loaded or not shared.native:
        return
    if loaded_lora() == "":
        shared.log.warning("LoRA extract: no LoRA detected")
        return
    if not filename:
        shared.log.warning("LoRA extract: target filename required")
        return
    t0 = time.time()
    maxrank = int(maxrank)
    rank_ratio = 1 if not auto_rank else rank_ratio

    if hasattr(shared.sd_model, 'text_encoder') and shared.sd_model.text_encoder is not None:
        for name, module in shared.sd_model.text_encoder.named_modules():
            weights_backup = getattr(module, "network_weights_backup", None)
            if weights_backup is None or getattr(module, "network_current_names", None) is None:
                continue
            prefix = "lora_te1_" if hasattr(shared.sd_model, 'text_encoder_2') else "lora_te_"
            module.svdhandler = SVDHandler(maxrank, rank_ratio)
            module.svdhandler.network_name = prefix + name.replace(".", "_")
            with devices.inference_context():
                module.svdhandler.decompose(module.weight, weights_backup)

    if hasattr(shared.sd_model, 'text_encoder_2'):
        for name, module in shared.sd_model.text_encoder_2.named_modules():
            weights_backup = getattr(module, "network_weights_backup", None)
            if weights_backup is None or getattr(module, "network_current_names", None) is None:
                continue
            module.svdhandler = SVDHandler(maxrank, rank_ratio)
            module.svdhandler.network_name = "lora_te2_" + name.replace(".", "_")
            with devices.inference_context():
                module.svdhandler.decompose(module.weight, weights_backup)

    if hasattr(shared.sd_model, 'unet'):
        for name, module in shared.sd_model.unet.named_modules():
            weights_backup = getattr(module, "network_weights_backup", None)
            if weights_backup is None or getattr(module, "network_current_names", None) is None:
                continue
            module.svdhandler = SVDHandler(maxrank, rank_ratio)
            module.svdhandler.network_name = "lora_unet_" + name.replace(".", "_")
            with devices.inference_context():
                module.svdhandler.decompose(module.weight, weights_backup)

    # TODO: Handle quant for Flux
    # if hasattr(shared.sd_model, 'transformer'):
    #     for name, module in shared.sd_model.transformer.named_modules():
    #         if "norm" in name and "linear" not in name:
    #             continue
    #         weights_backup = getattr(module, "network_weights_backup", None)
    #         if weights_backup is None:
    #             continue
    #         module.svdhandler = SVDHandler()
    #         module.svdhandler.network_name = "lora_transformer_" + name.replace(".", "_")
    #         module.svdhandler.decompose(module.weight, weights_backup)
    #         module.svdhandler.findrank(rank, rank_ratio)

    submodelname = ['text_encoder', 'text_encoder_2', 'unet', 'transformer']

    lora_state_dict = {}
    for sub in submodelname:
        submodel = getattr(shared.sd_model, sub, None)
        if submodel is not None:
            for _name, module in submodel.named_modules():
                if not hasattr(module, "svdhandler"):
                    continue
                lora_state_dict.update(module.svdhandler.makeweights())
                del module.svdhandler

    suffix = []
    if maxrank and auto_rank and rank_ratio != 1:
        suffix.append(f'maxrank{str(maxrank).replace(".","-")}')
    else:
        suffix.append(f'rank{str(maxrank).replace(".","-")}')
    if auto_rank and rank_ratio != 1:
        suffix.append(f'autorank{str(rank_ratio).replace(".","-")}')

    pathstr = str(os.path.join(shared.cmd_opts.lora_dir, filename+f'_{"_".join(suffix)}.safetensors'))
    save_file(lora_state_dict, pathstr)
    shared.log.info(f'LoRA extra: fn={pathstr} in {time.time()-t0} seconds')


def create_ui():
    def gr_show(visible=True):
        return {"visible": visible, "__type__": "update"}

    with gr.Tab(label="Extract LoRA"):
        with gr.Row():
            loaded = gr.Textbox(value="Press refresh to query loaded LoRA", label="Loaded LoRA", interactive=False)
            create_refresh_button(loaded, lambda: None, lambda: {'value': loaded_lora()}, "testid")
        with gr.Row():
            rank = gr.Slider(label="Maximum rank", value=32, minimum=1, maximum=256)
        with gr.Row():
            auto_rank = gr.Checkbox(value=False, label="Automatically determine rank")
        with gr.Row(visible=False) as rank_options:
            rank_ratio = gr.Slider(label="Autorank ratio", value=1, minimum=0, maximum=1, step=0.05, visible=True)
        with gr.Row():
            filename = gr.Textbox(label="LoRA target filename")
        with gr.Row():
            extract = gr.Button(value="Extract LoRA", variant='primary')

    auto_rank.change(fn=lambda x: gr_show(x), inputs=[auto_rank], outputs=[rank_options])
    extract.click(fn=make_lora, inputs=[filename, rank, auto_rank, rank_ratio], outputs=[])
