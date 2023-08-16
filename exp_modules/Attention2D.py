import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from thop import profile

from models.ghostnet import GhostBottleneck


class SelfAttention2D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim=64, patch_size=32):
        super(SelfAttention2D, self).__init__()
        self.extend = nn.Conv2d(in_ch, emb_dim * 3, kernel_size=1)
        self.patch_size = patch_size
        self.split = Rearrange('n (s e) (m1 p1) (m2 p2) -> s n e m1 m2 p1 p2', s=3, p1=self.patch_size, p2=self.patch_size)
        self.to_qkv = GhostBottleneck(emb_dim * 3, emb_dim * 12, emb_dim * 3)
        self.restore = Rearrange('n e m1 m2 p1 p2 -> n e (m1 p1) (m2 p2)')
        self.squeeze = nn.Conv2d(emb_dim, out_ch, kernel_size=1)

    def forward(self, x):
        # n,c,w,w -> n,e,w,w
        x = self.extend(x)
        # to q k v
        x = self.to_qkv(x)
        # n,e,w,w -> n,e,m,m,p,p
        assert x.shape[2] % self.patch_size == 0, 'input data shape cannot be divide into patch'
        x = self.split(x)
        q, k, v = x[0], x[1], x[2]
        attn = q @ k
        x = v @ attn
        x = self.restore(x)
        x = self.squeeze(x)
        return x


if __name__ == '__main__':
    data = torch.rand(1, 3, 224, 224)
    model = SelfAttention2D(3, 3, 12, 32)
    out = model(data)
    print(out.shape)

    flops, params = profile(model, inputs=(data,), verbose=False)
    print('flops:', flops / 1E6, 'M')
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))