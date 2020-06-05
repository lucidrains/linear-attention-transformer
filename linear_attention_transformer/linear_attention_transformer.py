import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from linear_attention_transformer.reversible import ReversibleSequence, SequentialSequence

# helper functions

def default(value, d):
    return d if value is None else value

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

def scatter_mean(src, t, index, dim, eps = 1e-5):
    numer = src.scatter_add(dim, index, t)
    denom = src.scatter_add(dim, index, torch.ones_like(t))
    return numer / (denom + eps)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

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

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)

# local attention

class LocalAttention(nn.Module):
    def __init__(self, bucket_size, heads, head_dim, causal = False, look_backward = 1, look_forward = None, dropout = 0., shared_qk = False):
        super().__init__()
        self.look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and self.look_forward > 0), 'you cannot look forward if causal'

        self.bucket_size = bucket_size
        self.causal = causal
        self.look_backward = look_backward
        self.shared_qk = shared_qk

        self.heads = heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, input_mask = None):
        shape = q.shape

        merge_into_batch = lambda t: t.reshape(-1, *t.shape[-2:])
        q, k, v = map(merge_into_batch, (q, k, v))

        b, t, e, h, device, dtype = *q.shape, self.heads, q.device, q.dtype
        bucket_size, causal, look_backward, look_forward, shared_qk = self.bucket_size, self.causal, self.look_backward, self.look_forward, self.shared_qk

        buckets = t // bucket_size

        if shared_qk:
            k = F.normalize(k, 2, dim=-1).type(q.type())

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

        if shared_qk:
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

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

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mult * dim, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim * mult, dim, bias=False)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

# self attention layer

DEFAULT_PSI = lambda x: F.elu(x) + 1

def safe_div(n, d, eps = 1e-6):
    return n.div_(d + eps)

def linear_attn(q, k, v, one_kv_head = False):
    q = q.softmax(dim=-1)
    k = k.softmax(dim=-2)

    context_einsum_eq = 'bhnd,bhne->bhde' if not one_kv_head else 'bnd,bne->bde'
    context = torch.einsum(context_einsum_eq, k, v)

    attn_einsum_eq = 'bhnd,bhde->bhne' if not one_kv_head else 'bhnd,bde->bhne'
    attn = torch.einsum(attn_einsum_eq, q, context)

    return attn.reshape(*q.shape)

def causal_linear_attn(q, k, v, psi = DEFAULT_PSI, one_kv_head = False, bucket_size = None):
    b, h, n, e, dtype = *q.shape, q.dtype
    bucket_size = default(bucket_size, 64)

    q = q.softmax(dim=-1)
    k = psi(k)

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
    def __init__(self, dim, heads, causal, one_kv_head = False, psi_fn = DEFAULT_PSI, blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128):
        super().__init__()
        assert (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = dim // heads
        self.heads = heads
        self.psi_fn = psi_fn

        self.global_attn_fn = linear_attn if not causal else partial(causal_linear_attn, psi=psi_fn, bucket_size = blindspot_size)

        self.local_attn_heads = n_local_attn_heads
        self.local_attn  = LocalAttention(local_attn_window_size, n_local_attn_heads, d_heads, causal = causal)

        self.to_q = nn.Linear(dim, dim, bias = False)

        kv_heads = 1 if one_kv_head else heads
        self.one_kv_head = one_kv_head
        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)

        self.to_out = nn.Linear(dim, dim)


    def forward(self, x, input_mask = None, **kwargs):
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        b, t, e, h = *q.shape, self.heads
        merge_heads = lambda x: x.reshape(b, t, h, -1).transpose(1, 2)

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
            global_out = self.global_attn_fn(q, k, v, one_kv_head = self.one_kv_head)
            out.append(global_out)

        attn = torch.cat(out, dim=1)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.to_out(attn)

class LinearAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads = 8, bucket_size = 64, causal = False, one_kv_head = False, ff_chunks = 1, reversible = False, blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128):
        super().__init__()
        if type(n_local_attn_heads) is not tuple:
            n_local_attn_heads = tuple([n_local_attn_heads] * depth)

        assert len(n_local_attn_heads) == depth, 'local attention heads tuple must have the same length as the depth'
        assert all([(local_heads <= heads) for local_heads in n_local_attn_heads]), 'number of local attn heads must be less than the maximum number of heads'

        layers = nn.ModuleList([])

        for _, local_heads in zip(range(depth), n_local_attn_heads):
            layer = nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads, causal, one_kv_head = one_kv_head, blindspot_size = blindspot_size, n_local_attn_heads = local_heads, local_attn_window_size = local_attn_window_size)),
                Chunk(ff_chunks, PreNorm(dim, FeedForward(dim)), along_dim = 1)
            ])
            layers.append(layer)

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.layers = execute_type(layers)

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)

class LinearAttentionTransformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, causal = False, one_kv_head = False, reversible = False, ff_chunks = 1, blindspot_size = 1, n_local_attn_heads = 0, local_attn_window_size = 128):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.transformer = LinearAttentionTransformer(dim, depth, max_seq_len, heads = heads, causal = causal, one_kv_head = one_kv_head, ff_chunks = ff_chunks, reversible = reversible, blindspot_size = blindspot_size, n_local_attn_heads = n_local_attn_heads, local_attn_window_size = local_attn_window_size)
        self.out = nn.Linear(dim, num_tokens)

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type(x.type())
        x = self.transformer(x, **kwargs)
        return self.out(x)
