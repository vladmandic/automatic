import time
import numpy as np
import re
import networks
import lora_patches
from modules import extra_networks, shared

# from https://github.com/cheald/sd-webui-loractl/blob/master/loractl/lib/utils.py
def get_stepwise(param, step, steps):
    def sorted_positions(raw_steps):
        steps = [[float(s.strip()) for s in re.split("[@~]", x)]
                 for x in re.split("[,;]", str(raw_steps))]
        # If we just got a single number, just return it
        if len(steps[0]) == 1:
            return steps[0][0]

        # Add implicit 1s to any steps which don't have a weight
        steps = [[s[0], s[1] if len(s) == 2 else 1] for s in steps]

        # Sort by index
        steps.sort(key=lambda k: k[1])

        steps = [list(v) for v in zip(*steps)]
        return steps

    def calculate_weight(m, step, max_steps, step_offset=2):
        if isinstance(m, list):
            if m[1][-1] <= 1.0:
                if max_steps > 0:
                    step = (step) / (max_steps - step_offset)
                else:
                    step = 1.0
            else:
                step = step
            v = np.interp(step, m[1], m[0])
            return v
        else:
            return m
    return calculate_weight(sorted_positions(param), step, steps)


class ExtraNetworkLora(extra_networks.ExtraNetwork):

    def __init__(self):
        super().__init__('lora')
        self.active = False
        self.errors = {}
        networks.originals = lora_patches.LoraPatches()

        """mapping of network names to the number of errors the network had during operation"""

    def activate(self, p, params_list, step=0):
        t0 = time.time()
        self.errors.clear()
        if len(params_list) > 0:
            self.active = True
            networks.originals.apply() # apply patches
            if networks.debug:
                shared.log.debug("LoRA activate")
        names = []
        te_multipliers = []
        unet_multipliers = []
        dyn_dims = []
        for params in params_list:
            assert params.items
            names.append(params.positional[0])
            te_multiplier = params.named.get("te", params.positional[1] if len(params.positional) > 1 else 1.0)
            if isinstance(te_multiplier, str) and "@" in te_multiplier:
                te_multiplier = get_stepwise(te_multiplier, step, p.steps)
            else:
                te_multiplier = float(te_multiplier)
            unet_multiplier = [params.positional[2] if len(params.positional) > 2 else te_multiplier] * 3
            unet_multiplier = [params.named.get("unet", unet_multiplier[0])] * 3
            unet_multiplier[0] = params.named.get("in", unet_multiplier[0])
            unet_multiplier[1] = params.named.get("mid", unet_multiplier[1])
            unet_multiplier[2] = params.named.get("out", unet_multiplier[2])
            for i in range(len(unet_multiplier)):
                if isinstance(unet_multiplier[i], str) and "@" in unet_multiplier[i]:
                    unet_multiplier[i] = get_stepwise(unet_multiplier[i], step, p.steps)
                else:
                    unet_multiplier[i] = float(unet_multiplier[i])
            dyn_dim = int(params.positional[3]) if len(params.positional) > 3 else None
            dyn_dim = int(params.named["dyn"]) if "dyn" in params.named else dyn_dim
            te_multipliers.append(te_multiplier)
            unet_multipliers.append(unet_multiplier)
            dyn_dims.append(dyn_dim)
        t1 = time.time()
        networks.load_networks(names, te_multipliers, unet_multipliers, dyn_dims)
        t2 = time.time()
        if shared.opts.lora_add_hashes_to_infotext:
            network_hashes = []
            for item in networks.loaded_networks:
                shorthash = item.network_on_disk.shorthash
                if not shorthash:
                    continue
                alias = item.mentioned_name
                if not alias:
                    continue
                alias = alias.replace(":", "").replace(",", "")
                network_hashes.append(f"{alias}: {shorthash}")
            if network_hashes:
                p.extra_generation_params["Lora hashes"] = ", ".join(network_hashes)
        if len(names) > 0 and step == 0:
            shared.log.info(f'LoRA apply: {names} patch={t1-t0:.2f} load={t2-t1:.2f}')
        elif self.active:
            self.active = False

    def deactivate(self, p):
        if shared.backend == shared.Backend.DIFFUSERS and hasattr(shared.sd_model, "unload_lora_weights") and hasattr(shared.sd_model, "text_encoder"):
            if 'CLIP' in shared.sd_model.text_encoder.__class__.__name__ and not (shared.compiled_model_state is not None and shared.compiled_model_state.is_compiled is True):
                if shared.opts.lora_fuse_diffusers:
                    shared.sd_model.unfuse_lora()
                try:
                    shared.sd_model.unload_lora_weights() # fails for non-CLIP models
                except Exception:
                    pass
        if not self.active and getattr(networks, "originals", None ) is not None:
            networks.originals.undo() # remove patches
            if networks.debug:
                shared.log.debug("LoRA deactivate")
        if self.active and networks.debug:
            shared.log.debug(f"LoRA end: load={networks.timer['load']:.2f} apply={networks.timer['apply']:.2f} restore={networks.timer['restore']:.2f}")
        if self.errors:
            p.comment("Networks with errors: " + ", ".join(f"{k} ({v})" for k, v in self.errors.items()))
            for k, v in self.errors.items():
                shared.log.error(f'LoRA errors: file="{k}" errors={v}')
            self.errors.clear()
