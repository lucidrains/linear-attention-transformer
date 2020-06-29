import torch
import torch.nn.functional as F
from torch import nn
import math
from operator import mul
from fractions import gcd
from collections import namedtuple
from functools import partial, reduce

from linformer import LinformerSelfAttention
from product_key_memory import PKM
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer.reversible import ReversibleSequence, SequentialSequence

# constants

DEFAULT_PSI = lambda x: F.elu(x) + 1


# namedtuple settings

LinformerSettings = namedtuple('LinformerSettings', ['k'])
LinformerContextSettings = namedtuple('LinformerContextSettings', ['seq_len', 'k'])

# helper functions

def default(value, d):
    return d if value is None else value

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val
        
def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def lcm(*numbers):
    return int(reduce(lambda x, y: (x * y) / gcd(x, y), numbers, 1))

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def expand_dim(t, dim, k):
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

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

# local attention

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

class LocalAttention(nn.Module):
    def __init__(self, bucket_size, heads, head_dim, causal = False, look_backward = 1, look_forward = None, dropout = 0.):
        super().__init__()
        self.look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and self.look_forward > 0), 'you cannot look forward if causal'

        self.bucket_size = bucket_size
        self.causal = causal
        self.look_backward = look_backward

        self.heads = heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, input_mask = None):
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
        q, k, v = map(merge_into_batch, (q, k, v))

        b, t, e, h, device, dtype = *q.shape, self.heads, q.device, q.dtype
        bucket_size, causal, look_backward, look_forward = self.bucket_size, self.causal, self.look_backward, self.look_forward
        assert (t % bucket_size) == 0, f'sequence length {t} must be divisible by bucket size {bucket_size} for local attention'

        buckets = t // bucket_size

        ticker = torch.arange(t, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, buckets, bucket_size)

        bucket_fn = lambda t: t.reshape(b, buckets, bucket_size, -1)
        bq, bk, bv = map(bucket_fn, (q, k, v))

        look_around_kwargs = {'backward': look_backward, 'forward': look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (e ** -0.5)

        mask_value = max_neg_value(dots)

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]
            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            input_mask = input_mask.reshape(-1, buckets, bucket_size)
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = (mq[:, :, :, None] * mk[:, :, None, :])
            mask = merge_dims(0, 1, expand_dim(mask, 1, h))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(*shape)
        return out

# feedforward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

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

def linear_attn(q, k, v, kv_mask = None, one_kv_head = False):
    if kv_mask is not None:
        mask_value = max_neg_value(q)
        mask = kv_mask[:, :, None] if one_kv_head else kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, mask_value)

    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    context_einsum_eq = 'bhnd,bhne->bhde' if not one_kv_head else 'bnd,bne->bde'
    context = torch.einsum(context_einsum_eq, k, v)

    attn_einsum_eq = 'bhnd,bhde->bhne' if not one_kv_head else 'bhnd,bde->bhne'
    attn = torch.einsum(attn_einsum_eq, q, context)

    return attn.reshape(*q.shape)

def causal_linear_attn(q, k, v, kv_mask = None, psi = DEFAULT_PSI, one_kv_head = False, bucket_size = None):
    b, h, n, e, dtype = *q.shape, q.dtype
    bucket_size = default(bucket_size, 64)
    assert (n % bucket_size) == 0, f'sequence length {n} must be divisible by the bucket size {bucket_size} for causal linear attention'

    q = q.softmax(dim=-1)
    k = psi(k)

    if kv_mask is not None:
        mask = kv_mask[:, :, None] if one_kv_head else kv_mask[:, None, :, None]
        k = k.masked_fill_(~mask, 0.)

    bucket_fn = lambda x: x.reshape(*x.shape[:-2], -1, bucket_size, e)
    b_q, b_k, b_v = map(bucket_fn, (q, k, v))

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim=-2).type(dtype)

    context_einsum_eq = 'bhund,bhune->bhude' if not one_kv_head else 'bund,bune->bude'
    context = torch.einsum(context_einsum_eq, b_k, b_v)
    context_cumsum = context.cumsum(dim=-3).type(dtype)

    context = safe_div(context_cumsum, b_k_cumsum.unsqueeze(-1))

    if bucket_size != 1:
        context = F.pad(context, (0, 0, 0, 0, 1, 0), value=0.)
        seq_dim = 1 if one_kv_head else 2
        context, _ = split_at_index(seq_dim, -1, context)

    attn_einsum_eq = 'bhund,bhude->bhune' if not one_kv_head else 'bhund,bude->bhune'
    attn = torch.einsum(attn_einsum_eq, b_q, context)
    return attn.reshape(*q.shape)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, causal = False, one_kv_head = False, psi_fn = DEFAULT_PSI, blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128, receives_context = False, dropout = 0., attn_dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = dim // heads

        self.heads = heads
        self.d_heads = d_heads
        self.psi_fn = psi_fn
        self.receives_context = receives_context

        self.global_attn_fn = linear_attn if not causal else partial(causal_linear_attn, psi=psi_fn, bucket_size = blindspot_size)

        self.local_attn_heads = n_local_attn_heads
        self.local_attn  = LocalAttention(local_attn_window_size, n_local_attn_heads, d_heads, causal = causal, dropout = attn_dropout)

        self.to_q = nn.Linear(dim, dim, bias = False)

        kv_heads = 1 if one_kv_head else heads
        self.one_kv_head = one_kv_head
        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)

        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_mask = None, context = None, context_mask = None, **kwargs):
        assert not (self.receives_context and context is None), 'context must be supplied if self attention is in receives context mode'

        if not self.receives_context:
            q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))
        else:
            q, k, v = (self.to_q(x), self.to_k(context), self.to_v(context))

        b, t, e, h, dh = *q.shape, self.heads, self.d_heads
        merge_heads = lambda x: x.reshape(b, -1, h, dh).transpose(1, 2)

        q = merge_heads(q)

        if not self.one_kv_head:
            k, v = map(merge_heads, (k, v))

        out = []

        split_index_fn = partial(split_at_index, 1, self.local_attn_heads)

        if not self.one_kv_head:
            (lq, q), (lk, k), (lv, v) = map(split_index_fn, (q, k, v))
        else:
            lq, q = split_index_fn(q)
            lk = expand_dim(k, 1, self.local_attn_heads)
            lv = expand_dim(v, 1, self.local_attn_heads)

        has_local, has_global = map(lambda x: x.shape[1] > 0, (lq, q))

        if has_local:
            local_out = self.local_attn(lq, lk, lv, input_mask = input_mask)
            out.append(local_out)

        if has_global:
            kv_mask = input_mask if not self.receives_context else context_mask
            global_out = self.global_attn_fn(q, k, v, one_kv_head = self.one_kv_head, kv_mask = kv_mask)
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
        if input_mask is not None:
            mask = input_mask.reshape(b, -1, ax).transpose(1, 2).reshape(b * ax, -1)

        x = self.fn(x, input_mask = mask, **kwargs)
        x = x.reshape(b, ax, -1, d).transpose(1, 2).reshape(b, t, d)
        return x

class LinearAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads = 8, bucket_size = 64, causal = False, one_kv_head = False, ff_chunks = 1, ff_glu = False, ff_dropout = 0., attn_layer_dropout = 0., attn_dropout = 0., reversible = False, blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128, psi_fn = DEFAULT_PSI, receives_context = False, attend_axially = False, pkm_layers = tuple(), pkm_num_keys = 128, linformer_settings = None, context_linformer_settings = None):
        super().__init__()
        assert not (causal and linformer_settings is not None), 'Linformer self attention layer can only be used for non-causal networks'
        assert linformer_settings is None or isinstance(linformer_settings, LinformerSettings), 'Linformer self-attention settings must be a LinformerSettings namedtuple'
        assert context_linformer_settings is None or isinstance(context_linformer_settings, LinformerContextSettings), 'Linformer contextual self-attention settings must be a LinformerSettings namedtuple'

        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
        assert all([(local_heads <= heads) for local_heads in n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

        layers = nn.ModuleList([])

        for ind, local_heads in zip(range(depth), n_local_attn_heads):
            layer_num = ind + 1
            use_pkm = layer_num in cast_tuple(pkm_layers)

            parallel_net = Chunk(ff_chunks, FeedForward(dim), along_dim = 1) if not use_pkm else PKM(dim)

            if linformer_settings is None:
                attn = SelfAttention(dim, heads, causal, one_kv_head = one_kv_head, blindspot_size = blindspot_size, n_local_attn_heads = local_heads, local_attn_window_size = local_attn_window_size, psi_fn = psi_fn, dropout = attn_layer_dropout, attn_dropout= attn_dropout)
            else:
                attn = LinformerSelfAttention(dim, max_seq_len, heads = heads, one_kv_head = one_kv_head, dropout = attn_dropout, **linformer_settings._asdict())

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, parallel_net)
            ]))

            if attend_axially:
                layers.append(nn.ModuleList([
                    PreNorm(dim, FoldAxially(local_attn_window_size, SelfAttention(dim, heads, causal, one_kv_head = one_kv_head, psi_fn = psi_fn, dropout = attn_layer_dropout, attn_dropout= attn_dropout))),
                    PreNorm(dim, Chunk(ff_chunks, FeedForward(dim, glu = ff_glu, dropout= ff_dropout), along_dim = 1))
                ]))

            if receives_context:
                if context_linformer_settings is None:
                    attn = SelfAttention(dim, heads, one_kv_head = one_kv_head, psi_fn = psi_fn, dropout = attn_layer_dropout, attn_dropout= attn_dropout, receives_context = True)
                else:
                    attn = LinformerSelfAttention(dim, heads = heads, one_kv_head = one_kv_head, dropout = attn_dropout, **context_linformer_settings._asdict())

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
        attn_route_map = {'input_mask': route_attn}
        self.layers = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        self.pad_to_multiple = lcm(
            1 if not causal else blindspot_size,
            1 if all([(h == 0) for h in n_local_attn_heads]) else local_attn_window_size
        )

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)

class LinearAttentionTransformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, causal = False, emb_dim = None, one_kv_head = False, reversible = False, ff_chunks = 1, ff_glu = False, ff_dropout = 0., attn_layer_dropout = 0., attn_dropout = 0., blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128, psi_fn = DEFAULT_PSI, return_embeddings = False, receives_context = False, pkm_layers = tuple(), pkm_num_keys = 128, attend_axially = False, linformer_settings = None, context_linformer_settings = None):
        assert (max_seq_len % local_attn_window_size) == 0, 'max sequence length must be divisible by the window size, to calculate number of kmeans cluster'
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.axial_pos_emb = AxialPositionalEmbedding(emb_dim, axial_shape=(max_seq_len // local_attn_window_size, local_attn_window_size))
        self.transformer = LinearAttentionTransformer(dim, depth, max_seq_len, heads = heads, causal = causal, one_kv_head = one_kv_head, ff_chunks = ff_chunks, ff_glu = ff_glu, ff_dropout = ff_dropout, attn_layer_dropout = attn_layer_dropout, attn_dropout = attn_dropout, reversible = reversible, blindspot_size = blindspot_size, n_local_attn_heads = n_local_attn_heads, local_attn_window_size = local_attn_window_size, psi_fn = psi_fn, receives_context = receives_context, pkm_layers = pkm_layers, pkm_num_keys = pkm_num_keys, attend_axially = attend_axially, linformer_settings = linformer_settings, context_linformer_settings = context_linformer_settings)

        if emb_dim != dim:
            self.transformer = ProjectInOut(self.transformer, emb_dim, dim, project_out = not return_embeddings)

        self.out = nn.Linear(emb_dim, num_tokens) if not return_embeddings else nn.Identity()

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.axial_pos_emb(x).type(x.type())
        x = self.transformer(x, **kwargs)
        return self.out(x)
