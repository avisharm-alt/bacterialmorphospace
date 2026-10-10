"""Pixel-space conditional diffusion over 5-channel Cell Painting crops.

A standard ADM-style U-Net (residual blocks with scale-shift conditioning, self-attention at 16x16), v-prediction on a
cosine schedule, deterministic DDIM sampling. The condition is the 32-d gene code of jump_g2p.diffusion_cv; it is
dropped (zeroed, with a flag) for a share of training crops, so one network gives the gene-conditioned model (w = 1),
the gene-free model (w = 0) and guided samples (w > 1). This matches the project's profile-level model, which also
compares the conditioned and code-dropped branches of one network.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ImageDiffusionConfig:
    base: int = 64
    mults: tuple = (1, 2, 3, 4)
    n_res: int = 2
    attn_res: tuple = (16,)
    dropout: float = 0.1
    steps: int = 30_000
    batch: int = 128
    lr: float = 2e-4
    warmup: int = 1000
    ema: float = 0.9995
    p_drop_code: float = 0.2
    cond_noise: float = 0.05
    sample_steps: int = 50
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def alpha_sigma(t: torch.Tensor):
    """Cosine schedule in continuous time t in [0, 1]."""
    a = torch.cos((t + 0.008) / 1.008 * math.pi / 2).clamp(1e-4, 1.0)
    s = torch.sin((t + 0.008) / 1.008 * math.pi / 2).clamp(0.0, 0.9999)
    return a, s


def t_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = (t.float() * 1000)[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, edim: int, dropout: float):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(edim, 2 * cout)
        self.n2, self.drop, self.c2 = nn.GroupNorm(32, cout), nn.Dropout(dropout), nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.emb(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.c2(self.drop(F.silu(self.n2(h) * (1 + s) + b)))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, c: int, heads: int = 4):
        super().__init__()
        self.n, self.qkv, self.out, self.h = nn.GroupNorm(32, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), heads
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, e=None):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B, C, H, W)
        return x + self.out(o)


class UNet(nn.Module):
    def __init__(self, in_ch: int, c_dim: int, cfg: ImageDiffusionConfig, res: int = 128):
        super().__init__()
        b, ed = cfg.base, cfg.base * 4
        self.base = b
        self.t_mlp = nn.Sequential(nn.Linear(b, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.c_mlp = nn.Sequential(nn.Linear(c_dim + 1, ed), nn.SiLU(), nn.Linear(ed, ed))
        self.inp = nn.Conv2d(in_ch, b, 3, padding=1)
        chs, ch, r = [b], b, res
        self.down = nn.ModuleList()
        for li, m in enumerate(cfg.mults):
            for _ in range(cfg.n_res):
                blk = nn.ModuleList([ResBlock(ch, b * m, ed, cfg.dropout)])
                ch = b * m
                if r in cfg.attn_res:
                    blk.append(Attn(ch))
                self.down.append(blk); chs.append(ch)
            if li < len(cfg.mults) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)])); chs.append(ch); r //= 2
        self.mid = nn.ModuleList([ResBlock(ch, ch, ed, cfg.dropout), Attn(ch), ResBlock(ch, ch, ed, cfg.dropout)])
        self.up = nn.ModuleList()
        for li, m in reversed(list(enumerate(cfg.mults))):
            for _ in range(cfg.n_res + 1):
                blk = nn.ModuleList([ResBlock(ch + chs.pop(), b * m, ed, cfg.dropout)])
                ch = b * m
                if r in cfg.attn_res:
                    blk.append(Attn(ch))
                self.up.append(blk)
            if li > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ch, ch, 3, padding=1)]))
                r *= 2
        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(ch, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight); nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, c, keep):
        e = self.t_mlp(t_embed(t, self.base)) + self.c_mlp(torch.cat([c * keep[:, None], keep[:, None]], 1))
        h = self.inp(x)
        hs = [h]
        for blk in self.down:
            if isinstance(blk[0], ResBlock):
                h = blk[0](h, e)
                if len(blk) > 1:
                    h = blk[1](h)
            else:
                h = blk[0](h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for blk in self.up:
            if isinstance(blk[0], ResBlock):
                h = blk[0](torch.cat([h, hs.pop()], 1), e)
                if len(blk) > 1:
                    h = blk[1](h)
            else:
                h = blk[1](blk[0](h))
        return self.out(h)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random flips and 90-degree rotations (cell morphology has no preferred orientation)."""
    k = int(torch.randint(0, 4, (1,), generator=gen, device=x.device))
    x = torch.rot90(x, k, (2, 3))
    if bool(torch.randint(0, 2, (1,), generator=gen, device=x.device)):
        x = x.flip(3)
    return x


class ImageDiffusion:
    def __init__(self, in_ch: int, c_dim: int, cfg: ImageDiffusionConfig, device: str | None = None):
        self.cfg, self.c_dim = cfg, c_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(cfg.seed)
        self.net = UNet(in_ch, c_dim, cfg).to(self.device)
        self.ema = UNet(in_ch, c_dim, cfg).to(self.device)
        self.ema.load_state_dict(self.net.state_dict())
        self.ema.requires_grad_(False)
        self.losses: list[float] = []

    def fit(self, tiles: torch.Tensor, tile_code: torch.Tensor, log_every: int = 1000) -> "ImageDiffusion":
        """tiles: (N, C, H, W) uint8 on device; tile_code: (N, c_dim) float on device."""
        cfg, dev = self.cfg, self.device
        gen = torch.Generator(device=dev).manual_seed(cfg.seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
        n, run, t0 = len(tiles), 0.0, time.time()
        self.net.train()
        net = self.net
        if dev.type == "cuda":   # ~1.7x faster on H100
            torch.backends.cudnn.benchmark = True
            net = torch.compile(self.net.to(memory_format=torch.channels_last))
        for step in range(cfg.steps):
            idx = torch.randint(0, n, (cfg.batch,), device=dev, generator=gen)
            x0 = augment(tiles[idx].float() / 127.5 - 1, gen)
            c = tile_code[idx] + cfg.cond_noise * torch.randn((cfg.batch, self.c_dim), device=dev, generator=gen)
            keep = (torch.rand(cfg.batch, device=dev, generator=gen) >= cfg.p_drop_code).float()
            t = torch.rand(cfg.batch, device=dev, generator=gen)
            a, s = alpha_sigma(t)
            a, s = a[:, None, None, None], s[:, None, None, None]
            eps = torch.randn(x0.shape, device=dev, generator=gen)
            xt = a * x0 + s * eps
            v = a * eps - s * x0
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = net(xt.contiguous(memory_format=torch.channels_last), t, c, keep)
            loss = F.mse_loss(pred.float(), v)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                d = min(cfg.ema, (1 + step) / (10 + step))
                for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                    pe.lerp_(p, 1 - d)
            run += loss.item() if step % 20 == 0 else 0.0
            if step % log_every == log_every - 1:
                self.losses.append(run / (log_every / 20))
                print(f"step {step + 1}: loss {self.losses[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
                run = 0.0
        return self

    @torch.no_grad()
    def sample(self, codes: np.ndarray, w: float = 1.0, seed: int = 0, chunk: int = 256, res: int = 128,
               in_ch: int = 5) -> np.ndarray:
        """One crop per row of `codes`; returns uint8 (N, C, H, W). w = 0 gene-free, 1 conditional, > 1 guided."""
        dev, net, out = self.device, self.ema, []
        net.eval()
        gen = torch.Generator(device=dev).manual_seed(seed)
        ts = torch.linspace(1.0, 0.0, self.cfg.sample_steps + 1, device=dev)
        for s0 in range(0, len(codes), chunk):
            c = torch.as_tensor(codes[s0:s0 + chunk], dtype=torch.float32, device=dev)
            n = len(c)
            x = torch.randn((n, in_ch, res, res), device=dev, generator=gen)
            ones, zeros = torch.ones(n, device=dev), torch.zeros(n, device=dev)
            for i in range(self.cfg.sample_steps):
                t, tn = ts[i].repeat(n), ts[i + 1]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    if w == 0:
                        v = net(x, t, c, zeros).float()
                    elif w == 1:
                        v = net(x, t, c, ones).float()
                    else:
                        v1, v0 = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([c, c]),
                                     torch.cat([ones, zeros])).float().chunk(2)
                        v = v0 + w * (v1 - v0)
                a, s = alpha_sigma(t[:1])
                x0 = (a * x - s * v).clamp(-1, 1)
                eps = (x - a * x0) / s
                an, sn = alpha_sigma(tn[None])
                x = an * x0 + sn * eps if i + 1 < self.cfg.sample_steps else x0
            out.append(((x + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy())
        return np.concatenate(out)
