"""A small conditional denoising diffusion model over wing-shape coordinates (step 3 of the wing plan).

The model works on whitened shape PCs (not pixels) and is conditioned on a vector c = [genotype code, sex]. The genotype
part is dropped (set to zero, with a flag) for a share of training rows, so one network gives both the genotype-conditioned
model and a genotype-free one (classifier-free guidance; guidance w = 0 is genotype-free, w = 1 plain conditional).

Everything here is device-agnostic torch; `generative.py` runs it on CPU locally and `modal_app.py` on a Modal GPU.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class DiffusionConfig:
    width: int = 256
    depth: int = 4
    steps: int = 20_000            # optimiser steps
    batch: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ema: float = 0.999
    T: int = 1000                  # diffusion timesteps (cosine schedule)
    sample_steps: int = 250        # strided ancestral (DDIM eta=1) sampling steps
    p_drop_genotype: float = 0.15  # classifier-free-guidance dropout of the genotype code
    cond_noise: float = 0.3        # Gaussian noise added to genotype codes in training (regularises a ~130-line input)
    clip_x0: float = 10.0          # clamp predicted clean data (whitened units) during sampling
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0]).clamp(1e-8, 1.0)
    return ab[1:].float()  # alpha_bar at t = 1..T


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(width, 2 * width)
        self.ff = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, h, e):
        scale, shift = self.film(e).chunk(2, dim=1)
        return h + self.ff(self.norm(h) * (1 + scale) + shift)


class Denoiser(nn.Module):
    """MLP epsilon-predictor with FiLM conditioning on (timestep, condition)."""

    def __init__(self, x_dim: int, c_dim: int, width: int = 256, depth: int = 4):
        super().__init__()
        self.inp = nn.Linear(x_dim, width)
        self.t_emb = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.c_emb = nn.Sequential(nn.Linear(c_dim + 1, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, x_dim))
        self.width = width

    def forward(self, x, t, c, keep):
        """keep: 1 where the genotype code is used, 0 where it is dropped (genotype-free)."""
        e = self.t_emb(timestep_embedding(t, self.width)) + self.c_emb(torch.cat([c, keep[:, None]], dim=1))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, e)
        return self.out(h)


class ConditionalDiffusion:
    """Fit on (x, genotype code, sex); sample x given (genotype code, sex) with guidance w."""

    def __init__(self, x_dim: int, g_dim: int, cfg: DiffusionConfig, device: str | None = None):
        self.cfg, self.g_dim = cfg, g_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(cfg.seed)
        self.net = Denoiser(x_dim, g_dim + 1, cfg.width, cfg.depth).to(self.device)
        self.ema = Denoiser(x_dim, g_dim + 1, cfg.width, cfg.depth).to(self.device)
        self.ema.load_state_dict(self.net.state_dict())
        self.ema.requires_grad_(False)
        self.ab = cosine_alpha_bar(cfg.T).to(self.device)
        self.losses: list[float] = []

    def _cond(self, g, sex):
        return torch.cat([g, sex[:, None]], dim=1)

    def fit(self, X: np.ndarray, G: np.ndarray, sex: np.ndarray) -> "ConditionalDiffusion":
        cfg, dev = self.cfg, self.device
        X = torch.as_tensor(X, dtype=torch.float32, device=dev)
        G = torch.as_tensor(G, dtype=torch.float32, device=dev)
        S = torch.as_tensor(sex, dtype=torch.float32, device=dev)
        gen = torch.Generator(device=dev).manual_seed(cfg.seed)
        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.05)
        n = len(X)
        self.net.train()
        running = 0.0
        for step in range(cfg.steps):
            idx = torch.randint(0, n, (min(cfg.batch, n),), device=dev, generator=gen)
            x0, g, s = X[idx], G[idx], S[idx]
            if cfg.cond_noise > 0:
                g = g + cfg.cond_noise * torch.randn(g.shape, device=dev, generator=gen)
            keep = (torch.rand(len(idx), device=dev, generator=gen) >= cfg.p_drop_genotype).float()
            g = g * keep[:, None]
            t = torch.randint(0, cfg.T, (len(idx),), device=dev, generator=gen)
            ab = self.ab[t][:, None]
            eps = torch.randn(x0.shape, device=dev, generator=gen)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            loss = ((self.net(xt, t, self._cond(g, s), keep) - eps) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                    pe.lerp_(p, 1 - cfg.ema)
            running += loss.item() if step % 50 == 0 else 0.0
            if step % 1000 == 999:
                self.losses.append(running / 20)
                running = 0.0
        return self

    @torch.no_grad()
    def sample(self, G: np.ndarray, sex: np.ndarray, w: float = 1.0, seed: int = 0, chunk: int = 65_536) -> np.ndarray:
        """One sample per row of (G, sex). w = 0: genotype-free; 1: conditional; > 1: guided."""
        out = []
        for s0 in range(0, len(G), chunk):
            out.append(self._sample(G[s0:s0 + chunk], sex[s0:s0 + chunk], w, seed + s0))
        return np.concatenate(out) if out else np.empty((0, self.net.out[1].out_features))

    def _sample(self, G, sex, w, seed):
        dev, cfg, net = self.device, self.cfg, self.ema
        net.eval()
        n = len(G)
        g = torch.as_tensor(G, dtype=torch.float32, device=dev)
        c = self._cond(g, torch.as_tensor(sex, dtype=torch.float32, device=dev))
        c0 = self._cond(torch.zeros_like(g), c[:, -1])
        ones, zeros = torch.ones(n, device=dev), torch.zeros(n, device=dev)
        gen = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn((n, net.out[1].out_features), device=dev, generator=gen)
        ts = torch.linspace(cfg.T - 1, 0, cfg.sample_steps, device=dev).round().long()
        for i, t in enumerate(ts):
            tt = t.repeat(n)
            if w == 0:
                eps = net(x, tt, c0, zeros)
            elif w == 1:
                eps = net(x, tt, c, ones)
            else:
                e1, e0 = net(x, tt, c, ones), net(x, tt, c0, zeros)
                eps = e0 + w * (e1 - e0)
            ab = self.ab[t]
            ab_prev = self.ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-cfg.clip_x0, cfg.clip_x0)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()  # keep eps consistent with the clamped x0
            if i + 1 == len(ts):
                x = x0
                break
            # DDIM with eta = 1 (ancestral): stochastic, matches DDPM marginals on the strided schedule
            sigma = ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_prev.sqrt() * x0 + dir_xt + sigma * torch.randn(x.shape, device=dev, generator=gen)
        return x.cpu().numpy()
