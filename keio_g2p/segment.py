"""Segment Keio phase-contrast images into single-cell outlines.

Cells are dark on a bright, fairly even background and sparse, so a classical pipeline is enough: flatten the
background, threshold, label, then keep isolated objects with a rod-like size and convexity. Each kept cell becomes a
closed contour resampled to N_PTS points, centred, rotated so its long axis is horizontal, and started at the point on
the +x pole, so outlines are comparable across cells. Pairs still joined at a septum stay as one object (they are part
of the strain's shape distribution).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

N_PTS = 64
MIN_AREA, MAX_AREA = 150, 6000      # pixels; WT cells are ~400–900 px in these images
MIN_SOLIDITY = 0.85                 # drops crossed or touching non-sibling cells
EDGE = 5                            # drop objects within this many px of the border


def resample(contour: np.ndarray, n: int = N_PTS) -> np.ndarray:
    c = np.vstack([contour, contour[:1]])
    seg = np.sqrt((np.diff(c, axis=0) ** 2).sum(1))
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1)


def canonical(xy: np.ndarray) -> np.ndarray:
    """Centre, rotate long axis to x, make counter-clockwise, start at the +x pole, and flip so the larger half
    is on +x (removes the arbitrary left/right and up/down choices)."""
    xy = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    xy = xy @ vt.T
    if np.mean(xy[:, 0] ** 3) < 0:
        xy[:, 0] *= -1
    if np.mean(xy[:, 1] ** 3) < 0:
        xy[:, 1] *= -1
    area2 = np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    if area2 < 0:
        xy = xy[::-1]
    i = int(np.argmax(xy[:, 0] - 0.01 * np.abs(xy[:, 1])))
    return np.roll(xy, -i, axis=0)


def segment(img: np.ndarray) -> list[np.ndarray]:
    im = img.astype(np.float32)
    bg = ndi.uniform_filter(ndi.minimum_filter(ndi.maximum_filter(im, 31), 31), 61)  # closing ≈ background
    flat = bg - im                                    # cells become bright
    flat = filters.gaussian(flat, 1.0, preserve_range=True)
    thr = filters.threshold_otsu(flat)
    mask = flat > thr
    mask = morphology.remove_small_objects(ndi.binary_fill_holes(morphology.binary_opening(mask, morphology.disk(1))), MIN_AREA)
    lab = measure.label(mask)
    out = []
    H, W = mask.shape
    for r in measure.regionprops(lab):
        y0, x0, y1, x1 = r.bbox
        if y0 < EDGE or x0 < EDGE or y1 > H - EDGE or x1 > W - EDGE:
            continue
        if not (MIN_AREA <= r.area <= MAX_AREA) or r.solidity < MIN_SOLIDITY:
            continue
        sub = np.pad(r.image, 2).astype(float)
        cs = measure.find_contours(sub, 0.5)
        if not cs:
            continue
        c = max(cs, key=len)[:, ::-1]                 # (x, y)
        out.append(canonical(resample(c)))
    return out
