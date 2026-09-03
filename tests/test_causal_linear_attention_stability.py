import ast
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import einsum


def load_causal_linear_attention():
    source_path = Path(__file__).parents[1] / 'linear_attention_transformer' / 'linear_attention_transformer.py'
    tree = ast.parse(source_path.read_text())
    names = {'exists', 'default', 'split_at_index', 'causal_linear_attn'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'torch': torch, 'math': math, 'F': F, 'einsum': einsum}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(source_path), 'exec'), namespace)
    return namespace['causal_linear_attn']


def test_extreme_half_precision_keys_remain_finite_with_gradients():
    causal_linear_attn = load_causal_linear_attention()
    q = torch.randn(1, 2, 8, 16, dtype = torch.float16, requires_grad = True)
    k = torch.full_like(q, 1000., requires_grad = True)
    v = torch.randn_like(q, requires_grad = True)

    out = causal_linear_attn(q, k, v, bucket_size = 1)

    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()
    out.float().square().mean().backward()
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


def test_causal_prefix_does_not_depend_on_future_key_magnitude():
    causal_linear_attn = load_causal_linear_attention()
    torch.manual_seed(0)
    q = torch.randn(1, 1, 8, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    changed_future = k.clone()
    changed_future[:, :, -1] = 1000.

    baseline = causal_linear_attn(q, k, v, bucket_size = 1)
    changed = causal_linear_attn(q, changed_future, v, bucket_size = 1)

    # Every prefix before the final token must be invariant to changes in a
    # future key, including when that key is far outside the exponent range.
    torch.testing.assert_close(baseline[:, :, :-1], changed[:, :, :-1])


def test_moderate_keys_match_the_unstabilized_feature_map():
    causal_linear_attn = load_causal_linear_attention()
    torch.manual_seed(1)
    q = torch.randn(1, 1, 8, 4)
    k = torch.randn_like(q).clamp(-2., 2.)
    v = torch.randn_like(q)

    stabilized = causal_linear_attn(q, k, v, bucket_size = 1)

    # For ordinary logits no clipping is active, so the stabilized path should
    # agree with the original exp(k) feature map.
    q_ref = q.softmax(dim = -1) * q.shape[-1] ** -0.5
    k_ref = torch.exp(k)
    b_q = q_ref.reshape(1, 1, 8, 1, 4)
    b_k = k_ref.reshape(1, 1, 8, 1, 4)
    b_v = v.reshape(1, 1, 8, 1, 4)
    k_sum = b_k.sum(dim = -2).cumsum(dim = -2)
    context = einsum('bhund,bhune->bhude', b_k, b_v).cumsum(dim = -3)
    denom = einsum('bhud,bhund->bhun', k_sum, b_q).clamp(min = 1e-3)
    reference = einsum('bhund,bhude,bhun->bhune', b_q, context, 1. / denom).reshape_as(q)

    torch.testing.assert_close(stabilized, reference, rtol = 1e-5, atol = 1e-5)
