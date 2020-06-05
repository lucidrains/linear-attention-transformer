import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
from linear_attention_transformer.reversible import ReversibleSequence, SequentialSequence

# helper functions

def default(value, d):
    return d if value is None else value

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

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
    def __init__(self, bucket_size, causal = False, look_backward = 1, look_forward = 0, dropout = 0., shared_qk = False):
        super().__init__()
        assert not (causal and look_forward > 0), 'you cannot look forward if causal'
        self.bucket_size = bucket_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.shared_qk = shared_qk
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, input_mask = None):
        b, t, e, device, dtype = *q.shape, q.device, q.dtype
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
        mask_value = float('-inf')

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
            mask = (mq[:, None, :, :, None] * mk[:, None, :, None, :])
            mask = merge_dims(0, 1, mask.expand(-1, h, -1, -1, -1))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhje->bhie', attn, bv)
        out = out.reshape(b, t, e)
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

def causal_linear_attn(q, k, v, psi = DEFAULT_PSI, one_kv_head = False, buckets = None):
    b, h, n, e = q.shape
    buckets = default(buckets, n)

    q = q.softmax(dim=-1)
    k = psi(k)

    bucket_fn = lambda x: x.reshape(*x.shape[:-2], buckets, -1, e)
    b_q, b_k, b_v = map(bucket_fn, (q, k, v))

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim=-2)

    context_einsum_eq = 'bhund,bhune->bhude' if not one_kv_head else 'bund,bune->bude'
    context = torch.einsum(context_einsum_eq, b_k, b_v)
    context_cumsum = context.cumsum(dim=-3)

    context = safe_div(context_cumsum, b_k_cumsum.unsqueeze(-1))

    if buckets != n:
        context = F.pad(context, (0, 0, 0, 0, 1, 0), value=0.)
        context = context[:, :-1]

    attn_einsum_eq = 'bhund,bhude->bhune' if not one_kv_head else 'bhund,bude->bhune'
    attn = torch.einsum(attn_einsum_eq, b_q, context)
    return attn.reshape(*q.shape)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, causal, one_kv_head = False, psi_fn = DEFAULT_PSI):
        super().__init__()
        assert (dim % heads) == 0, 'embedding dimension must be divisible by number of heads'
        d_heads = dim // heads
        self.heads = heads
        self.psi_fn = psi_fn
        self.attn_fn = linear_attn if not causal else partial(causal_linear_attn, psi=psi_fn)

        self.to_q = nn.Linear(dim, dim, bias = False)

        kv_heads = 1 if one_kv_head else heads
        self.one_kv_head = one_kv_head
        self.kv_heads = kv_heads
        self.to_k = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_v = nn.Linear(dim, d_heads * kv_heads, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, **kwargs):
        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        b, t, e, h = *q.shape, self.heads
        merge_heads = lambda x: x.reshape(b, t, h, -1).transpose(1, 2)

        q = merge_heads(q)

        if self.kv_heads != 1:
            k, v = map(merge_heads, (k, v))

        attn = self.attn_fn(q, k, v, one_kv_head = self.one_kv_head)
        attn = attn.transpose(1, 2).reshape(b, t, -1)
        return self.to_out(attn)

class LinearAttentionTransformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads = 8, bucket_size = 64, causal = False, one_kv_head = False, ff_chunks = 1, reversible = False):
        super().__init__()
        layers = nn.ModuleList([])

        for _ in range(depth):
            layer = nn.ModuleList([
                PreNorm(dim, SelfAttention(dim, heads, causal, one_kv_head = one_kv_head)),
                Chunk(ff_chunks, PreNorm(dim, FeedForward(dim)), along_dim = 1)
            ])
            layers.append(layer)

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.layers = execute_type(layers)

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)

class LinearAttentionTransformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, causal = False, one_kv_head = False, reversible = False, ff_chunks = 1):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)
        self.transformer = LinearAttentionTransformer(dim, depth, max_seq_len, heads = heads, causal = causal, one_kv_head = one_kv_head, ff_chunks = ff_chunks, reversible = reversible)
        self.out = nn.Linear(dim, num_tokens)

    def forward(self, x, **kwargs):
        x = self.token_emb(x)
        x = x + self.pos_emb(x).type(x.type())
        x = self.transformer(x, **kwargs)
        return self.out(x)
