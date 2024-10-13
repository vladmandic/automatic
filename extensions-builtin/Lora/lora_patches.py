import sys
import torch
import networks
from modules import patches, shared, model_quant


class LoraPatches:
    def __init__(self):
        self.active = False
        self.Linear_forward = None
        self.Linear_load_state_dict = None
        self.Conv2d_forward = None
        self.Conv2d_load_state_dict = None
        self.GroupNorm_forward = None
        self.GroupNorm_load_state_dict = None
        self.LayerNorm_forward = None
        self.LayerNorm_load_state_dict = None
        self.MultiheadAttention_forward = None
        self.MultiheadAttention_load_state_dict = None
        # optional quant forwards
        self.Linear4bit_forward = None # bitsandbytes
        self.QLinear_forward = None # optimum.quanto
        self.QConv2d_forward = None # optimum.quanto

    def handle_quant(self, apply: bool):
        if 'bitsandbytes' in sys.modules: # lora should not be first to initialize quantization
            bnb = model_quant.load_bnb(silent=True)
            if bnb is not None:
                if apply:
                    self.Linear4bit_forward = patches.patch(__name__, bnb.nn.Linear4bit, 'forward', networks.network_Linear4bit_forward)
                else:
                    self.Linear4bit_forward = patches.undo(__name__, bnb.nn.Linear4bit, 'forward') # pylint: disable=E1128
        if 'optimum.quanto' in sys.modules:
            quanto = model_quant.load_quanto(silent=True)
            if quanto is not None:
                if apply:
                    self.QLinear_forward = patches.patch(__name__, quanto.nn.QLinear, 'forward', networks.network_QLinear_forward)
                    self.QConv2d_forward = patches.patch(__name__, quanto.nn.QConv2d, 'forward', networks.network_QConv2d_forward)
                else:
                    self.QLinear_forward = patches.undo(__name__, quanto.nn.QLinear, 'forward') # pylint: disable=E1128
                    self.QConv2d_forward = patches.undo(__name__, quanto.nn.QConv2d, 'forward') # pylint: disable=E1128

    def apply(self):
        if self.active or shared.opts.lora_force_diffusers:
            return
        self.Linear_forward = patches.patch(__name__, torch.nn.Linear, 'forward', networks.network_Linear_forward)
        self.Linear_load_state_dict = patches.patch(__name__, torch.nn.Linear, '_load_from_state_dict', networks.network_Linear_load_state_dict)
        self.Conv2d_forward = patches.patch(__name__, torch.nn.Conv2d, 'forward', networks.network_Conv2d_forward)
        self.Conv2d_load_state_dict = patches.patch(__name__, torch.nn.Conv2d, '_load_from_state_dict', networks.network_Conv2d_load_state_dict)
        self.GroupNorm_forward = patches.patch(__name__, torch.nn.GroupNorm, 'forward', networks.network_GroupNorm_forward)
        self.GroupNorm_load_state_dict = patches.patch(__name__, torch.nn.GroupNorm, '_load_from_state_dict', networks.network_GroupNorm_load_state_dict)
        self.LayerNorm_forward = patches.patch(__name__, torch.nn.LayerNorm, 'forward', networks.network_LayerNorm_forward)
        self.LayerNorm_load_state_dict = patches.patch(__name__, torch.nn.LayerNorm, '_load_from_state_dict', networks.network_LayerNorm_load_state_dict)
        self.MultiheadAttention_forward = patches.patch(__name__, torch.nn.MultiheadAttention, 'forward', networks.network_MultiheadAttention_forward)
        self.MultiheadAttention_load_state_dict = patches.patch(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict', networks.network_MultiheadAttention_load_state_dict)
        self.handle_quant(apply=True)
        networks.timer['load'] = 0
        networks.timer['apply'] = 0
        networks.timer['restore'] = 0
        self.active = True

    def undo(self):
        if not self.active or shared.opts.lora_force_diffusers:
            return
        self.Linear_forward = patches.undo(__name__, torch.nn.Linear, 'forward') # pylint: disable=E1128
        self.Linear_load_state_dict = patches.undo(__name__, torch.nn.Linear, '_load_from_state_dict') # pylint: disable=E1128
        self.Conv2d_forward = patches.undo(__name__, torch.nn.Conv2d, 'forward') # pylint: disable=E1128
        self.Conv2d_load_state_dict = patches.undo(__name__, torch.nn.Conv2d, '_load_from_state_dict') # pylint: disable=E1128
        self.GroupNorm_forward = patches.undo(__name__, torch.nn.GroupNorm, 'forward') # pylint: disable=E1128
        self.GroupNorm_load_state_dict = patches.undo(__name__, torch.nn.GroupNorm, '_load_from_state_dict') # pylint: disable=E1128
        self.LayerNorm_forward = patches.undo(__name__, torch.nn.LayerNorm, 'forward') # pylint: disable=E1128
        self.LayerNorm_load_state_dict = patches.undo(__name__, torch.nn.LayerNorm, '_load_from_state_dict') # pylint: disable=E1128
        self.MultiheadAttention_forward = patches.undo(__name__, torch.nn.MultiheadAttention, 'forward') # pylint: disable=E1128
        self.MultiheadAttention_load_state_dict = patches.undo(__name__, torch.nn.MultiheadAttention, '_load_from_state_dict') # pylint: disable=E1128
        self.handle_quant(apply=False)
        patches.originals.pop(__name__, None)
        self.active = False
