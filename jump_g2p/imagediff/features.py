"""Image features for scoring generated Cell Painting crops against real ones.

Two feature sets, both computed identically on real and generated crops:
- dino: DINOv2 ViT-S/14 class token of each channel separately (channel replicated to RGB, resized to 224), concatenated
  over the 5 channels (5 x 384). A generic, pretrained image representation, as in per-channel FID variants.
- cp: a small CellProfiler-like set per channel (intensity mean, sd, 10/50/90/99th percentiles, bright-area fraction,
  edge strength) plus nucleus count and mean nucleus size from the DNA channel (C01). Interpretable morphology numbers.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DNA = 0  # C01 is Hoechst (nuclei) in source_13


def load_dino(device):
    import timm
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    return m.eval().to(device)


@torch.no_grad()
def dino_features(model, tiles: np.ndarray, device, chunk: int = 256) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    out = []
    for s in range(0, len(tiles), chunk):
        x = torch.as_tensor(tiles[s:s + chunk], device=device).float() / 255
        B, C = x.shape[:2]
        x = x.reshape(B * C, 1, *x.shape[2:]).expand(-1, 3, -1, -1)
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = model((x - mean) / std).float()
        out.append(f.reshape(B, C * f.shape[1]).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def cp_features(tiles: np.ndarray, device, chunk: int = 1024) -> np.ndarray:
    out = []
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32)[None, None]
    for s in range(0, len(tiles), chunk):
        x = torch.as_tensor(tiles[s:s + chunk], device=device).float() / 255
        B, C, H, W = x.shape
        flat = x.reshape(B, C, -1)
        q = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9, 0.99], device=device), dim=2).permute(1, 2, 0)
        edge = F.conv2d(x.reshape(B * C, 1, H, W), lap, padding=1).abs().reshape(B, C, -1).mean(2)
        f = [flat.mean(2, keepdim=True), flat.std(2, keepdim=True), q, (flat > 0.25).float().mean(2, keepdim=True),
             edge[..., None]]
        f = torch.cat(f, 2).reshape(B, -1)
        # nuclei: local maxima of the blurred DNA channel above a fixed threshold; size = DNA-positive area per nucleus
        d = F.avg_pool2d(x[:, DNA:DNA + 1], 5, 1, 2)
        peak = (d == F.max_pool2d(d, 9, 1, 4)) & (d > 0.15)
        n = peak.flatten(1).sum(1).float()
        area = (x[:, DNA] > 0.15).flatten(1).float().sum(1)
        f = torch.cat([f, n[:, None], (area / n.clamp(min=1))[:, None]], 1)
        out.append(f.cpu().numpy())
    return np.concatenate(out)


def cp_names(n_ch: int = 5) -> list[str]:
    per = ["mean", "sd", "p10", "p50", "p90", "p99", "bright_frac", "edge"]
    return [f"C0{c + 1}_{p}" for c in range(n_ch) for p in per] + ["n_nuclei", "nucleus_area"]
