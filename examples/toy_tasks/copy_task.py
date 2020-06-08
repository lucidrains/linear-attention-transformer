import tqdm
import torch
import torch.optim as optim

from linear_attention_transformer import LinearAttentionTransformerLM
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
GENERATE_EVERY  = 100
NUM_TOKENS = 256 + 2
ENC_SEQ_LEN = 256
DEC_SEQ_LEN = 512

# helpers

def cycle():
    while True:
        prefix = torch.ones((BATCH_SIZE, 1)).long().cuda()
        src = torch.randint(2, NUM_TOKENS, (BATCH_SIZE, ENC_SEQ_LEN)).long().cuda()
        tgt = torch.cat((prefix, src, src), 1)
        src_mask = torch.ones(BATCH_SIZE, ENC_SEQ_LEN).bool().cuda()
        tgt_mask = torch.ones(BATCH_SIZE, tgt.shape[1]).bool().cuda()
        yield (src, tgt, src_mask, tgt_mask)

# instantiate model

enc = LinearAttentionTransformerLM(
    num_tokens = NUM_TOKENS,
    dim = 512,
    heads = 8,
    depth = 1,
    max_seq_len = ENC_SEQ_LEN,
    n_local_attn_heads = 4,
    return_embeddings = True
).cuda()

dec = LinearAttentionTransformerLM(
    num_tokens = NUM_TOKENS,
    dim = 512,
    heads = 8,
    depth = 1,
    causal = True,
    max_seq_len = DEC_SEQ_LEN,
    receives_context = True
).cuda()

dec = AutoregressiveWrapper(dec)

# optimizer

optim = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=LEARNING_RATE)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    enc.train(), dec.train()
    src, tgt, src_mask, tgt_mask = next(cycle())

    context = enc(src)
    loss = dec(tgt, context = context, return_loss = True)
    loss.backward()
    print(loss.item())

    optim.step()
    optim.zero_grad()

    if i % GENERATE_EVERY == 0:
        enc.eval(), dec.eval()
        src, _, src_mask, _ = next(cycle())
        src, src_mask = src[0:1], src_mask[0:1]
        start_tokens = (torch.ones((1, 1)) * 1).long().cuda()

        context = enc(src)
        sample = dec.generate(start_tokens, ENC_SEQ_LEN, context = context)
        incorrects = (src != sample).abs().sum()

        print(f"input:  ", src)
        print(f"predicted output:  ", sample)
        print(f"incorrects: {incorrects}")
