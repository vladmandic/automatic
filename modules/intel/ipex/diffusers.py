from functools import wraps
import torch
import diffusers # pylint: disable=import-error

# pylint: disable=protected-access, missing-function-docstring, line-too-long


# Diffusers FreeU
# Diffusers is imported before ipex hijacks so fourier_filter needs hijacking too
original_fourier_filter = diffusers.utils.torch_utils.fourier_filter
@wraps(diffusers.utils.torch_utils.fourier_filter)
def fourier_filter(x_in, threshold, scale):
    return_dtype = x_in.dtype
    return original_fourier_filter(x_in.to(dtype=torch.float32), threshold, scale).to(dtype=return_dtype)


# fp64 error
class FluxPosEmbed(torch.nn.Module):
    def __init__(self, theta: int, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(n_axes):
            cos, sin = diffusers.models.embeddings.get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=torch.float32,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def ipex_diffusers(device_supports_fp64=False, can_allocate_plus_4gb=False):
    diffusers.utils.torch_utils.fourier_filter = fourier_filter
    if not device_supports_fp64:
        diffusers.models.embeddings.FluxPosEmbed = FluxPosEmbed
        diffusers.models.transformers.transformer_flux.FluxPosEmbed = FluxPosEmbed
        diffusers.models.controlnets.controlnet_flux.FluxPosEmbed = FluxPosEmbed
