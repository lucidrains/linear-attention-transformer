import torch
import torch.nn.functional as F
from torch import nn, einsum
import math
from operator import mul
from math import gcd
from collections import namedtuple
from functools import partial, reduce

from local_attention import LocalAttention
from linformer import LinformerSelfAttention

from product_key_memory import PKM
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer.reversible import ReversibleSequence, SequentialSequence

from einops import rearrange, repeat

# namedtuple settings

LinformerSettings = namedtuple('LinformerSettings', ['k'])
LinformerContextSettings = namedtuple('LinformerContextSettings', ['seq_len', 'k'])

# helper functions

def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def always(value):
    return lambda *args, **kwargs: value

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val
        
def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def lcm(*numbers):
    return int(reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class ProjectInOut(nn.Module):
    def __init__(self, fn, dim_in, dim_out, project_out = True):
        super().__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else nn.Identity()

    def forward(self, x, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, **kwargs)
        x = self.project_out(x)
        return x

# token shifting helper classes

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value = 0.)

class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)[None, :, :]

# rotary positional embeddings classes and helpers

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float)
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.register_buffer('emb', emb)

    def forward(self, x):
        return self.emb[None, :x.shape[1], :].to(x)

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

# position permutator classes

class PositionPermutator(nn.Module):
    def __init__(
        self,
        *,
        dim_head,
        heads,
        max_seq_len,
        causal = False
    ):
        super().__init__()
        self.causal = causal

        base_permutations = torch.stack([torch.randperm(dim_head) for _ in range(heads)])

        # calculate the permutation, for each head, across the entire sequence length
        seq_permutations = [base_permutations]
        for _ in range(max_seq_len - 1):
            curr_permutation = seq_permutations[-1]
            next_permutation = curr_permutation.gather(-1, base_permutations)
            seq_permutations.append(next_permutation)

        permutations = torch.stack(seq_permutations, dim = 1) # (heads, seq, feature)

        # calculate ratios
        ratios = torch.sigmoid(torch.arange(heads) / heads * 3 + 2) if causal else None

        self.register_buffer('permutations', permutations)
        self.register_buffer('ratios', ratios)

    def apply_ratio(self, q, k):
        if not self.causal:
            return t
        seq = torch.arange(q.shape[-2], device = q.device)
        ratios = rearrange(self.ratios, 'h -> () h () ()')
        seq = rearrange(seq, 'n -> () () n ()')
        q_ratios = ratios ** seq
        k_ratios = ratios ** -seq
        return q * q_ratios, k * k_ratios

    def forward(self, t):
        b, _, n, _ = t.shape
        permutations = repeat(self.permutations, '... -> b ...', b = b)
        permutations = permutations[:, :, :n]
        permuted_t = t.gather(-1, permutations)
        return permuted_t

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

# self attention layer

def linear_attn(q, k, v, kv_mask = None, apply_qk_fn = None):
    dim = q.shape[-1]

    if exists(kv_mask):
        mask_value = max_neg_value(q)
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)
        v = v.masked_fill_(~mask, 0.)
        del mask

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    if exists(apply_qk_fn):
        q, k = apply_qk_fn(q, k)

    q = q * dim ** -0.5

    context = einsum('bhnd,bhne->bhde', k, v)
    attn = einsum('bhnd,bhde->bhne', q, context)
    return attn.reshape(*q.shape)

def causal_linear_attn(q, k, v, kv_mask = None, bucket_size = None, apply_qk_fn = None, eps = 1e-3):
    b, h, n, e, dtype = *q.shape, q.dtype
    bucket_size = default(bucket_size, 64)
    bucket_size = max(bucket_size, 1)
    assert bucket_size == 0 or (n % bucket_size) == 0, f'sequence length {n} must be divisible by the bucket size {bucket_size} for causal linear attention'

    q = F.relu(q) + eps
    k = F.relu(k) + eps

    if exists(apply_qk_fn):
        q, k = apply_qk_fn(q, k)

    if exists(kv_mask):
        mask = kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, 0.)
        v = v.masked_fill_(~mask, 0.)
        del mask

    bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
    b_q, b_k, b_v = map(bucket_fn, (q, k, v))

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

    context = einsum('bhund,bhune->bhude', b_k, b_v)
    context = context.cumsum(dim = -3).type(dtype)

    if bucket_size > 1:
        context = F.pad(context, (0, 0, 0, 0, 1, 0), value = 0.)
        context, _ = split_at_index(2, -1, context)

        b_k_cumsum = F.pad(b_k_cumsum, (0, 0, 1, 0), value = 0.)
        b_k_cumsum, _ = split_at_index(2, -1, b_k_cumsum)

    D_inv = 1. / (einsum('bhud,bhund->bhun', b_k_cumsum, b_q))
    attn = einsum('bhund,bhude,bhun->bhune', b_q, context, D_inv)
    return attn.reshape(*q.shape)

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        causal = False,
        dim_head = None,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 128,
        receives_context = False,
        use_pos_permuted_qk = False,
        max_seq_len = None,
        dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        assert dim_head or (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = default(dim_head, dim // heads)

        self.heads = heads
        self.d_heads = d_heads
        self.receives_context = receives_context

        self.global_attn_heads = heads - n_local_attn_heads
        self.global_attn_fn = linear_attn if not causal else partial(causal_linear_attn, bucket_size = blindspot_size)

        self.local_attn_heads = n_local_attn_heads
        self.local_attn  = LocalAttention(local_attn_window_size, causal = causal, dropout = attn_dropout)

        self.to_q = nn.Linear(dim, d_heads * heads, bias = False)

        kv_heads = heads

        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)

        self.pos_permutator = None
        if use_pos_permuted_qk:
            assert exists(max_seq_len), 'max_seq_len must be supplied if position dependent permutation is turned on'
            self.pos_permutator = PositionPermutator(dim_head = d_heads, heads = self.global_attn_heads, max_seq_len = max_seq_len, causal = causal)

        self.to_out = nn.Linear(d_heads * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        input_mask = None,
        context = None,
        context_mask = None,
        pos_emb = None,
        **kwargs
    ):
        assert not (self.receives_context and not exists(context)), 'context must be supplied if self attention is in receives context mode'

        if not self.receives_context:
            q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        else:
            q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads

        merge_heads = lambda x: x.reshape(*x.shape[:2], -1, dh).transpose(1, 2)

        q, k, v = map(merge_heads, (q, k, v))

        if exists(pos_emb) and not self.receives_context:
            q, k = apply_rotary_pos_emb(q, k, pos_emb)

        out = []

        split_index_fn = partial(split_at_index, 1, self.local_attn_heads)

        (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))

        has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        if has_local:
            local_out = self.local_attn(lq, lk, lv, input_mask = input_mask)
            out.append(local_out)

        if has_global:
            kv_mask = input_mask if not self.receives_context else context_mask
            apply_qk_fn = self.pos_permutator.apply_ratio if exists(self.pos_permutator) else None

            if exists(self.pos_permutator) and not self.receives_context:
                q, k = map(self.pos_permutator, (q, k))

            global_out = self.global_attn_fn(q, k, v, kv_mask = kv_mask, apply_qk_fn = apply_qk_fn)
            out.append(global_out)

        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.dropout(self.to_out(attn))

# transformer and language model classes

class FoldAxially(nn.Module):
    def __init__(self, axial_dim, fn):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim
    def forward(self, x, input_mask = None, **kwargs):
        b, t, d, ax = *x.shape, self.axial_dim
        x = x.reshape(b, -1, ax, d).transpose(1, 2).reshape(b * ax, -1, d)

        mask = None
        if exists(input_mask):
            mask = input_mask.reshape(b, -1, ax).transpose(1, 2).reshape(b * ax, -1)

        x = self.fn(x, input_mask = mask, **kwargs)
        x = x.reshape(b, ax, -1, d).transpose(1, 2).reshape(b, t, d)
        return x

class LinearAttentionTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = None,
        bucket_size = 64,
        causal = False,
        ff_chunks = 1,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        attn_dropout = 0.,
        reversible = False,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 128,
        receives_context = False,
        attend_axially = False,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        linformer_settings = None,
        context_linformer_settings = None,
        shift_tokens = False,
        use_pos_permuted_qk = False
    ):
        super().__init__()
        assert not (causal and exists(linformer_settings)), 'Linformer self attention layer can only be used for non-causal networks'
        assert not exists(linformer_settings) or isinstance(linformer_settings, LinformerSettings), 'Linformer self-attention settings must be a LinformerSettings namedtuple'
        assert not exists(context_linformer_settings) or isinstance(context_linformer_settings, LinformerContextSettings), 'Linformer contextual self-attention settings must be a LinformerSettings namedtuple'

        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
        assert all([(local_heads <= heads) for local_heads in n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

        layers = nn.ModuleList([])

        for ind, local_heads in zip(range(depth), n_local_attn_heads):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)

            parallel_net = Chunk(ff_chunks, FeedForward(dim), along_dim = 1) if not use_pkm else PKM(dim)

            if not exists(linformer_settings):
                attn = SelfAttention(dim, heads, causal, dim_head = dim_head, blindspot_size = blindspot_size, n_local_attn_heads = local_heads, local_attn_window_size = local_attn_window_size, use_pos_permuted_qk = use_pos_permuted_qk, max_seq_len = max_seq_len, dropout = attn_layer_dropout, attn_dropout= attn_dropout)
            else:
                attn = LinformerSelfAttention(dim, max_seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, **linformer_settings._asdict())

            if shift_tokens:
                shifts = (1, 0, -1) if not causal else (1, 0)
                attn, parallel_net = map(partial(PreShiftTokens, shifts), (attn, parallel_net))

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, parallel_net)
            ]))

            if attend_axially:
                layers.append(nn.ModuleList([
                    PreNorm(dim, FoldAxially(local_attn_window_size, SelfAttention(dim, heads, causal, dropout = attn_layer_dropout, attn_dropout= attn_dropout))),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                ]))

            if receives_context:
                if not exists(context_linformer_settings):
                    attn = SelfAttention(dim, heads, dim_head = dim_head, dropout = attn_layer_dropout, attn_dropout= attn_dropout, receives_context = True)
                else:
                    attn = LinformerSelfAttention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, **context_linformer_settings._asdict())

                layers.append(nn.ModuleList([
                    PreNorm(dim, attn),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        axial_layer = ((True, False),) if attend_axially else tuple()
        attn_context_layer = ((True, False),) if receives_context else tuple()
        route_attn = ((True, False), *axial_layer, *attn_context_layer) * depth
        route_context = ((False, False), *axial_layer, *attn_context_layer) * depth

        context_route_map = {'context': route_context, 'context_mask': route_context} if receives_context else {}
        attn_route_map = {'input_mask': route_attn, 'pos_emb': route_attn}
        self.layers = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        self.pad_to_multiple = lcm(
            1 if not causal else blindspot_size,
            1 if all([(h == 0) for h in n_local_attn_heads]) else local_attn_window_size
        )

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)

class LinearAttentionTransformerLM(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        causal = False,
        emb_dim = None,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        ff_dropout = 0.,
        attn_layer_dropout = 0.,
        attn_dropout = 0.,
        blindspot_size = 1,
        n_local_attn_heads = 0,
        local_attn_window_size = 128,
        return_embeddings = False,
        receives_context = False,
        pkm_layers = tuple(),
        pkm_num_keys = 128,
        attend_axially = False,
        linformer_settings = None,
        context_linformer_settings = None,
        use_axial_pos_emb = True,
        use_rotary_emb = False,
        use_pos_permuted_qk = False,
        shift_tokens = False
    ):
        assert n_local_attn_heads == 0 or (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the local attention window size'
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)

        if use_rotary_emb:
            self.pos_emb = FixedPositionalEmbedding(emb_dim, max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, max_seq_len)
        elif use_axial_pos_emb:
            self.pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(math.ceil(max_seq_len / local_attn_window_size), local_attn_window_size))
            self.layer_pos_emb = always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len)
            self.layer_pos_emb = always(None)

        self.transformer = LinearAttentionTransformer(dim, depth, max_seq_len, heads = heads, dim_head = dim_head, causal = causal, ff_chunks = ff_chunks, ff_glu = ff_glu, ff_dropout = ff_dropout, attn_layer_dropout = attn_layer_dropout, attn_dropout = attn_dropout, reversible = reversible, blindspot_size = blindspot_size, n_local_attn_heads = n_local_attn_heads, local_attn_window_size = local_attn_window_size, receives_context = receives_context, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys, attend_axially = attend_axially, linformer_settings = linformer_settings, context_linformer_settings = context_linformer_settings, shift_tokens = shift_tokens, use_pos_permuted_qk = use_pos_permuted_qk)

        if emb_dim != dim:
            self.transformer = ProjectInOut(self.transformer, emb_dim, dim, project_out = not return_embeddings)

        self.norm = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, num_tokens) if not return_embeddings else nn.Identity()

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type(x.type())

        layer_pos_emb = self.layer_pos_emb(x)
        x = self.transformer(x, pos_emb = layer_pos_emb, **kwargs)
        x = self.norm(x)
        return self.out(x)
