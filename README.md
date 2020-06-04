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

```bash
import torch
from linear_attention_transformer import LinearAttentionTransformerLM

model = LinearAttentionTransformerLM(
    num_tokens = 20000,
    dim = 512,
    heads = 8,
    depth = 12,
    max_seq_len = 2048,
    causal = True
).cuda()

x = torch.randint(0, 20000, (1, 2048)).cuda()
model(x) # (1, 2048, 512)
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
