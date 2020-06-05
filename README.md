## Linear Attention Transformer (wip)

<img src="./linear-attention.png"/>

[![PyPI version](https://badge.fury.io/py/linear-attention-transformer.svg)](https://badge.fury.io/py/linear-attention-transformer)

A fully featured Transformer that mixes (QKᵀ)V local attention with Q(KᵀV) global attention (scales linearly with respect to sequence length) for efficient long-range language modeling.

## Install

```bash
$ pip install linear-attention-transformer
```

## Usage

Language model

```python
import torch
from linear_attention_transformer import LinearAttentionTransformerLM

model = LinearAttentionTransformerLM(
    num_tokens = 20000,
    dim = 512,
    heads = 8,
    depth = 12,
    max_seq_len = 8192,
    causal = True,            # auto-regressive or not
    blindspot_size = 64,      # this gives the global attention a blindspot of 64 tokens back in the auto-regressive case, but gives back an order of magnitude return in memory savings. should be paired with local attention of at least a window size of this setting. setting this to 1 will allow for full global attention of past
    n_local_attn_heads = 4,   # number of local attention heads, can be a tuple specifying the exact number of local attention heads at that depth
    local_attn_window = 128,  # receptive field of the local attention
    one_kv_head = True,       # use one key/value head to save on memory / compute
    reversible = True,        # use reversible nets, from Reformer paper
    ff_chunks = 2,            # feedforward chunking, from Reformer paper
).cuda()

x = torch.randint(0, 20000, (1, 8192)).cuda()
model(x) # (1, 8192, 512)
```

## Citations

```bibtex
@inproceedings{katharopoulos-et-al-2020,
  author    = {Katharopoulos, A. and Vyas, A. and Pappas, N. and Fleuret, F.},
  title     = {Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2020},
  note      = {(to appear)}
}
```

```bibtex
@article{shen2019efficient,
  author    = {Zhuoran Shen and
               Mingyuan Zhang and
               Haiyu Zhao and
               Shuai Yi and
               Hongsheng Li},
  title     = {Efficient Attention: Attention with Linear Complexities},
  journal   = {CoRR},
  volume    = {abs/1812.01243},
  year      = {2018},
  url       = {http://arxiv.org/abs/1812.01243}
}
```

```bibtex
@misc{shazeer2019fast,
  title   = {Fast Transformer Decoding: One Write-Head is All You Need},
  author  = {Noam Shazeer},
  year    = {2019},
  eprint  = {1911.02150},
  archivePrefix = {arXiv}
}
```

```bibtex
@inproceedings{kitaev2020reformer,
    title       = {Reformer: The Efficient Transformer},
    author      = {Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle   = {International Conference on Learning Representations},
    year        = {2020},
    url         = {https://openreview.net/forum?id=rkgNKkHtvB}
}
```
