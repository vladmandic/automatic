from __future__ import annotations
import sys
import math
import psutil
from functools import cache

import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

from modules import shared, errors, devices
from modules.hypernetworks import hypernetwork

from .sub_quadratic_attention import efficient_dot_product_attention # pylint: disable=relative-beyond-top-level

from diffusers.utils import USE_PEFT_BACKEND


if shared.opts.cross_attention_optimization == "xFormers":
    try:
        import xformers.ops # pylint: disable=import-error
        shared.xformers_available = True
    except Exception:
        pass
else:
    if sys.modules.get("xformers", None) is not None:
        shared.log.debug('Unloading xFormers')
    sys.modules["xformers"] = None
    sys.modules["xformers.ops"] = None


def get_available_vram():
    if shared.device.type == 'cuda' or shared.device.type == 'xpu':
        try:
            stats = torch.cuda.memory_stats(shared.device)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch
        except Exception:
            mem_free_total = 1024 * 1024 * 1024
        return mem_free_total
    elif shared.device.type == 'privateuseone':
        return torch.dml.mem_get_info(shared.device)[0]
    else:
        return psutil.virtual_memory().available


# see https://github.com/basujindal/stable-diffusion/pull/117 for discussion
def split_cross_attention_forward_v1(self, x, context=None, mask=None): # pylint: disable=unused-argument
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    del context, context_k, context_v, x

    q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], 2):
            end = i + 2
            s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
            s1 *= self.scale
            s2 = s1.softmax(dim=-1)
            del s1
            r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
            del s2
        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


# taken from https://github.com/Doggettx/stable-diffusion and modified
def split_cross_attention_forward(self, x, context=None, mask=None): # pylint: disable=unused-argument
    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    dtype = q_in.dtype
    if shared.opts.upcast_attn:
        q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k_in = k_in * self.scale
        del context, x
        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
        del q_in, k_in, v_in
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        mem_free_total = get_available_vram()
        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1
        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')
        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)
            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1
            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2
        del q, k, v
    r1 = r1.to(dtype)
    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


# -- Taken from https://github.com/invoke-ai/InvokeAI and modified --
mem_total_gb = psutil.virtual_memory().total // (1 << 30)

def einsum_op_compvis(q, k, v):
    s = einsum('b i d, b j d -> b i j', q, k)
    s = s.softmax(dim=-1, dtype=s.dtype)
    return einsum('b i j, b j d -> b i d', s, v)

def einsum_op_slice_0(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[0], slice_size):
        end = i + slice_size
        r[i:end] = einsum_op_compvis(q[i:end], k[i:end], v[i:end])
    return r

def einsum_op_slice_1(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        r[:, i:end] = einsum_op_compvis(q[:, i:end], k, v)
    return r

def einsum_op_mps_v1(q, k, v):
    if q.shape[0] * q.shape[1] <= 2**16: # (512x512) max q.shape[1]: 4096
        return einsum_op_compvis(q, k, v)
    else:
        slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
        if slice_size % 4096 == 0:
            slice_size -= 1
        return einsum_op_slice_1(q, k, v, slice_size)

def einsum_op_mps_v2(q, k, v):
    if mem_total_gb > 8 and q.shape[0] * q.shape[1] <= 2**16:
        return einsum_op_compvis(q, k, v)
    else:
        return einsum_op_slice_0(q, k, v, 1)

def einsum_op_tensor_mem(q, k, v, max_tensor_mb):
    size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
    if size_mb <= max_tensor_mb:
        return einsum_op_compvis(q, k, v)
    div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
    if div <= q.shape[0]:
        return einsum_op_slice_0(q, k, v, q.shape[0] // div)
    return einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1))

def einsum_op_cuda(q, k, v):
    try:
        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
    except Exception:
        mem_free_total = 1024 * 1024 * 1024
    # Divide factor of safety as there's copying and fragmentation
    return einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))

def einsum_op_dml(q, k, v):
    mem_free, mem_total = torch.dml.mem_get_info(q.device)
    mem_active = mem_total - mem_free
    mem_reserved = mem_total * 0.7
    return einsum_op_tensor_mem(q, k, v, (mem_reserved - mem_active) if mem_reserved > mem_active else 1)

def einsum_op(q, k, v):
    if q.device.type == 'cuda' or q.device.type == 'xpu':
        return einsum_op_cuda(q, k, v)

    if q.device.type == 'mps':
        if mem_total_gb >= 32 and q.shape[0] % 32 != 0 and q.shape[0] * q.shape[1] < 2**18:
            return einsum_op_mps_v1(q, k, v)
        return einsum_op_mps_v2(q, k, v)

    if q.device.type == 'privateuseone':
        return einsum_op_dml(q, k, v)

    # Smaller slices are faster due to L2/L3/SLC caches.
    # Tested on i7 with 8MB L3 cache.
    return einsum_op_tensor_mem(q, k, v, 32)

def split_cross_attention_forward_invokeAI(self, x, context=None, mask=None): # pylint: disable=unused-argument
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v if v.device.type == 'mps' else v.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k = k * self.scale
        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))
        r = einsum_op(q, k, v)
    r = r.to(dtype)
    return self.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))

# -- End of code from https://github.com/invoke-ai/InvokeAI --


# Based on Birch-san's modified implementation of sub-quadratic attention from https://github.com/Birch-san/diffusers/pull/1
# The sub_quad_attention_forward function is under the MIT License listed under Memory Efficient Attention in the Licenses section of the web UI interface
def sub_quad_attention_forward(self, x, context=None, mask=None):
    assert mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."

    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    q = q.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    k = k.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    v = v.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)

    if q.device.type == 'mps':
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k = q.float(), k.float()

    x = sub_quad_attention(q, k, v, q_chunk_size=shared.opts.sub_quad_q_chunk_size, kv_chunk_size=shared.opts.sub_quad_kv_chunk_size, chunk_threshold=shared.opts.sub_quad_chunk_threshold, use_checkpoint=self.training)

    x = x.to(dtype)

    x = x.unflatten(0, (-1, h)).transpose(1,2).flatten(start_dim=2)

    out_proj, dropout = self.to_out
    x = out_proj(x)
    x = dropout(x)

    return x

def sub_quad_attention(q, k, v, q_chunk_size=1024, kv_chunk_size=None, kv_chunk_size_min=None, chunk_threshold=None, use_checkpoint=True):
    bytes_per_token = torch.finfo(q.dtype).bits//8
    batch_x_heads, q_tokens, _ = q.shape
    _, k_tokens, _ = k.shape
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

    if chunk_threshold is None:
        chunk_threshold_bytes = int(get_available_vram() * 0.9) if q.device.type == 'mps' else int(get_available_vram() * 0.7)
    elif chunk_threshold == 0:
        chunk_threshold_bytes = None
    else:
        chunk_threshold_bytes = int(0.01 * chunk_threshold * get_available_vram())

    if kv_chunk_size_min is None and chunk_threshold_bytes is not None:
        kv_chunk_size_min = chunk_threshold_bytes // (batch_x_heads * bytes_per_token * (k.shape[2] + v.shape[2]))
    elif kv_chunk_size_min == 0:
        kv_chunk_size_min = None

    if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
        # the big matmul fits into our memory limit; do everything in 1 chunk,
        # i.e. send it down the unchunked fast-path
        kv_chunk_size = k_tokens

    with devices.without_autocast(disable=q.dtype == v.dtype):
        return efficient_dot_product_attention(
            q,
            k,
            v,
            query_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min = kv_chunk_size_min,
            use_checkpoint=use_checkpoint,
        )


def get_xformers_flash_attention_op(q, k, v):
    if 'xFormers enable flash Attention' not in shared.opts.cross_attention_options:
        return None

    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp # pylint: disable=used-before-assignment
        fw, _bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        errors.display_once(e, "enabling flash attention")

    return None


def xformers_attention_forward(self, x, context=None, mask=None): # pylint: disable=unused-argument
    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    q, k, v = (rearrange(t, 'b n (h d) -> b n h d', h=h) for t in (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=get_xformers_flash_attention_op(q, k, v))

    out = out.to(dtype)

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    return self.to_out(out)

# Based on Diffusers usage of scaled dot product attention from https://github.com/huggingface/diffusers/blob/c7da8fd23359a22d0df2741688b5b4f33c26df21/src/diffusers/models/cross_attention.py
# The scaled_dot_product_attention_forward function contains parts of code under Apache-2.0 license listed under Scaled Dot Product Attention in the Licenses section of the web UI interface
def scaled_dot_product_attention_forward(self, x, context=None, mask=None):
    batch_size, sequence_length, inner_dim = x.shape

    if mask is not None:
        mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
        mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])

    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    head_dim = inner_dim // h
    q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    del q_in, k_in, v_in
    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
    hidden_states = hidden_states.to(dtype)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states

def scaled_dot_product_no_mem_attention_forward(self, x, context=None, mask=None):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return scaled_dot_product_attention_forward(self, x, context, mask)

def cross_attention_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q1 = self.q(h_)
    k1 = self.k(h_)
    v = self.v(h_)

    # compute attention
    b, c, h, w = q1.shape

    q2 = q1.reshape(b, c, h*w)
    del q1

    q = q2.permute(0, 2, 1)   # b,hw,c
    del q2

    k = k1.reshape(b, c, h*w) # b,c,hw
    del k1

    h_ = torch.zeros_like(k, device=q.device)

    mem_free_total = get_available_vram()

    tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
    mem_required = tensor_size * 2.5
    steps = 1

    if mem_required > mem_free_total:
        steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

    slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size

        w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w2 = w1 * (int(c)**(-0.5))
        del w1
        w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
        del w2

        # attend to values
        v1 = v.reshape(b, c, h*w)
        w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        del w3

        h_[:, :, i:end] = torch.bmm(v1, w4)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        del v1, w4

    h2 = h_.reshape(b, c, h, w)
    del h_

    h3 = self.proj_out(h2)
    del h2

    h3 += x

    return h3

def xformers_attnblock_forward(self, x):
    try:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape # pylint: disable=unused-variable
        q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
        dtype = q.dtype
        if shared.opts.upcast_attn:
            q, k = q.float(), k.float()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return x + out
    except NotImplementedError:
        return cross_attention_attnblock_forward(self, x)

def sdp_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape # pylint: disable=unused-variable

    # SDP optimization kenels are built for operations with multiple attention heads.
    # Four dimensional tensors are required for mem_efficient and flash attention to work.
    # We add an attention head dimension `a` to allow these kernels to be used.
    q, k, v = (rearrange(t, '(b a) c h w -> b a (h w) c', a=1) for t in (q, k, v))
    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    out = out.to(dtype)
    out = rearrange(out, 'b a (h w) c -> (b a) c h w', h=h) # remove the one attention head dimension `a`
    out = self.proj_out(out)
    return x + out

def sdp_no_mem_attnblock_forward(self, x):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return sdp_attnblock_forward(self, x)

def sub_quad_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape # pylint: disable=unused-variable
    q, k, v = (rearrange(t, 'b c h w -> b (h w) c') for t in (q, k, v))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = sub_quad_attention(q, k, v, q_chunk_size=shared.opts.sub_quad_q_chunk_size, kv_chunk_size=shared.opts.sub_quad_kv_chunk_size, chunk_threshold=shared.opts.sub_quad_chunk_threshold, use_checkpoint=self.training)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out


@cache
def find_dynamic_v1_slice_size(slice_size, slice_block_size, slice_rate=4):
    while (slice_size * slice_block_size) > slice_rate:
        slice_size = slice_size // 2
        if slice_size <= 1:
            slice_size = 1
            break
    return slice_size

@cache
def find_dynamic_attention_v1_slice_sizes(query_shape, query_element_size, slice_rate=4):
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
        split_slice_size = find_dynamic_v1_slice_size(split_slice_size, slice_block_size, slice_rate=slice_rate)
        if split_slice_size * slice_block_size > slice_rate:
            slice_2_block_size = split_slice_size * shape_three * shape_four / 1024 / 1024 * query_element_size
            do_split_2 = True
            split_2_slice_size = find_dynamic_v1_slice_size(split_2_slice_size, slice_2_block_size, slice_rate=slice_rate)
            if split_2_slice_size * slice_2_block_size > slice_rate:
                slice_3_block_size = split_slice_size * split_2_slice_size * shape_four / 1024 / 1024 * query_element_size
                do_split_3 = True
                split_3_slice_size = find_dynamic_v1_slice_size(split_3_slice_size, slice_3_block_size, slice_rate=slice_rate)

    return do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size

class DynamicAttnProcessorV1:
    r"""
    dynamically slices attention queries based on query size and slice rate in GB
    slicing will not get triggered if the query size is smaller than the slice rate to gain performance
    based on AttnProcessor V1
    """

    def __call__(self, attn, hidden_states: torch.FloatTensor,
    encoder_hidden_states=None, attention_mask=None,
    temb=None, scale: float = 1.0) -> torch.Tensor: # pylint: disable=too-many-statements, too-many-locals, too-many-branches

        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

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

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        ####################################################################
        # Slicing parts:
        batch_size_attention, query_tokens, shape_three = query.shape[0], query.shape[1], query.shape[2]
        hidden_states = torch.zeros(query.shape, device=query.device, dtype=query.dtype)
        do_split, do_split_2, do_split_3, split_slice_size, split_2_slice_size, split_3_slice_size = find_dynamic_attention_v1_slice_sizes(query.shape, query.element_size(), slice_rate=shared.opts.dynamic_attention_slice_rate)

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
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
