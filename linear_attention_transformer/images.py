import torch
from torch import nn

class ImageLinearAttention(nn.Module):
    def __init__(self, chan, chan_out = None, kernel_size = 1, padding = 0, stride = 1, key_dim = 64, value_dim = 64, heads = 8):
        super().__init__()

        self.chan = chan
        chan_out = chan if chan_out is None else chan_out

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.heads = heads
        
        conv_kwargs = {'padding': padding, 'stride': stride}
        self.to_q = nn.Conv2d(chan, key_dim * heads, kernel_size, **conv_kwargs)
        self.to_k = nn.Conv2d(chan, key_dim, kernel_size, **conv_kwargs)
        self.to_v = nn.Conv2d(chan, value_dim, kernel_size, **conv_kwargs)

        out_conv_kwargs = {'padding': padding}
        self.to_out = nn.Conv2d(value_dim * heads, chan_out, kernel_size, **out_conv_kwargs)

    def forward(self, x, context = None):
        b, c, h, w, k_dim = *x.shape, self.key_dim

        q, k, v = (self.to_q(x), self.to_k(x), self.to_v(x))

        q = q.reshape(b, self.heads, -1, h * w)
        k = k.reshape(b, -1, h * w)
        v = v.reshape(b, -1, h * w)

        if context is not None:
            context = context.reshape(b, c, 1, -1)
            ck = self.to_k(context).reshape(b, k_dim, -1)
            cv = self.to_v(context).reshape(b, k_dim, -1)
            k = torch.cat((k, ck), dim=2)
            v = torch.cat((v, cv), dim=2)

        k = k.softmax(dim=2)
        q = q.softmax(dim=2)

        context = torch.einsum('bdn,ben->bde', k, v)
        out = torch.einsum('bhdn,bde->bhen', q, context)
        out = out.reshape(b, -1, h, w)
        out = self.to_out(out)
        return out
