"""Point-cloud vision-language-action policy (PC-VLA) for the tea-harvest task.

- Vision: a fused RGB-D point cloud (N x 6: xyz in the platform frame + rgb) from the rail and wrist cameras.
  Local geometry encoder in the spirit of PointNet++/DP3: farthest-point-sample K centres, group the k nearest
  neighbours, encode each group with a shared MLP + max-pool, add a learned Fourier embedding of the centre -> K tokens.
- Language: the instruction embedded by a frozen pretrained VLM text tower (SmolVLM2, the backbone of SmolVLA),
  mean-pooled hidden states, projected to one token (precomputed per instruction).
- Proprioception: the 5 servo targets, one token.
- Action head: conditional flow matching over an action chunk (H steps x 5), a transformer that attends to the
  context tokens; integrated with Euler steps at inference.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def fps(xyz, k):
    """Farthest point sampling. xyz (B, N, 3) -> indices (B, k)."""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, k, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), 1e10, device=xyz.device)
    far = torch.randint(0, N, (B,), device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(k):
        idx[:, i] = far
        c = xyz[ar, far][:, None]
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


class FourierPos(nn.Module):
    def __init__(self, dim, n_freq=10, scale=30.0):
        super().__init__()
        self.register_buffer("B", torch.randn(3, n_freq) * scale)
        self.proj = nn.Linear(2 * n_freq + 3, dim)

    def forward(self, xyz):
        z = xyz @ self.B
        return self.proj(torch.cat([xyz, torch.sin(z), torch.cos(z)], -1))


class PointTokenizer(nn.Module):
    def __init__(self, dim=256, n_groups=64, k=32):
        super().__init__()
        self.n_groups, self.k = n_groups, k
        self.mlp1 = nn.Sequential(nn.Linear(6, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 128))
        self.mlp2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, dim))
        self.pos = FourierPos(dim)

    def forward(self, pts):
        """pts (B, N, 6) -> tokens (B, G, dim), centres (B, G, 3)."""
        xyz, rgb = pts[..., :3], pts[..., 3:]
        B = pts.shape[0]
        ci = fps(xyz, self.n_groups)
        ar = torch.arange(B, device=pts.device)[:, None]
        centres = xyz[ar, ci]                                           # (B, G, 3)
        d = torch.cdist(centres, xyz)                                   # (B, G, N)
        nn_idx = d.topk(self.k, largest=False).indices                  # (B, G, k)
        g_xyz = xyz[ar[..., None], nn_idx] - centres[:, :, None]         # local coordinates
        g_rgb = rgb[ar[..., None], nn_idx]
        f = self.mlp1(torch.cat([g_xyz * 20.0, g_rgb], -1))              # (B, G, k, 128); local offsets in ~5 cm units
        f = torch.cat([f, f.max(2, keepdim=True).values.expand_as(f)], -1)
        f = self.mlp2(f).max(2).values                                  # (B, G, dim)
        return f + self.pos(centres), centres


class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, t):
        half = self.dim // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * 1000 * f[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class PCVLA(nn.Module):
    def __init__(self, lang_dim=960, dim=256, horizon=16, act_dim=5, n_layers=4, n_heads=8, n_groups=64):
        super().__init__()
        self.horizon, self.act_dim = horizon, act_dim
        self.points = PointTokenizer(dim, n_groups=n_groups)
        self.lang = nn.Sequential(nn.LayerNorm(lang_dim), nn.Linear(lang_dim, dim))
        self.state = nn.Sequential(nn.Linear(act_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.type_emb = nn.Parameter(torch.zeros(3, dim))
        enc = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, 0.0, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(enc, 2)
        self.act_in = nn.Linear(act_dim, dim)
        self.act_pos = nn.Parameter(torch.randn(horizon, dim) * .02)
        self.time = TimeEmb(dim)
        dec = nn.TransformerDecoderLayer(dim, n_heads, 4 * dim, 0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, n_layers)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, act_dim))

    def encode(self, pts, lang, state):
        p, _ = self.points(pts)
        ctx = torch.cat([self.lang(lang)[:, None] + self.type_emb[0], self.state(state)[:, None] + self.type_emb[1], p + self.type_emb[2]], 1)
        return self.context(ctx)

    def velocity(self, ctx, x_t, t):
        h = self.act_in(x_t) + self.act_pos + self.time(t)[:, None]
        return self.out(self.decoder(h, ctx))

    def loss(self, pts, lang, state, actions):
        """Flow matching: x_t = (1-t) noise + t actions, target velocity = actions - noise."""
        ctx = self.encode(pts, lang, state)
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        return F.mse_loss(self.velocity(ctx, x_t, t), actions - noise)

    @torch.no_grad()
    def sample(self, pts, lang, state, steps=10):
        ctx = self.encode(pts, lang, state)
        x = torch.randn(pts.shape[0], self.horizon, self.act_dim, device=pts.device)
        for i in range(steps):
            t = torch.full((pts.shape[0],), i / steps, device=pts.device)
            x = x + self.velocity(ctx, x, t) / steps
        return x
