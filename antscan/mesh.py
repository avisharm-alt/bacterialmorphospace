"""Turn one Antscan whole-body STL into a compact, pose-robust shape record.

The Biomedisa STLs are binary, 100-300 MB, in voxel units. We never keep them: each is streamed to a temp file,
reduced to a few thousand surface points plus global descriptors, and deleted.

Museum ants are posed differently (legs and antennae at any angle), so a raw point cloud mostly encodes pose.
The descriptors here are chosen to be robust to that: size, principal-axis proportions, and distributions of
point-to-point distances, distances from the centroid and triangle angles (Osada et al. 2002 shape distributions).
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import connected_components

log = logging.getLogger(__name__)

BASE_URL = "https://biomedisa.info"
_STL_DTYPE = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])

N_SURFACE = 8192  # area-weighted surface samples kept per specimen
N_BINS = 32
GRID = 96  # voxels along the longest body axis for the solid / core volumes
CORE_RADIUS = 3  # opening radius in grid voxels; strips legs, antennae and setae, keeps head, mesosoma, gaster
N_PROFILE = 48  # stations along the long axis for the core cross-section profile


def read_binary_stl(path: str) -> np.ndarray:
    """Triangles as a (n, 3, 3) float64 array."""
    with open(path, "rb") as f:
        f.seek(80)
        n = int(np.frombuffer(f.read(4), "<u4")[0])
        tri = np.fromfile(f, dtype=_STL_DTYPE, count=n)
    if len(tri) != n:
        raise ValueError(f"{path}: header says {n} triangles, file has {len(tri)}")
    return tri["v"].astype(np.float64)


def largest_component(tri: np.ndarray, min_frac: float = 0.0) -> tuple[np.ndarray, float]:
    """Keep the connected piece with the most surface area; also return that piece's share of all area.

    Segmentation debris (dust, label fragments, detached tarsi) shows up as small separate components.
    """
    v = tri.reshape(-1, 3)
    _, inv = np.unique(np.round(v, 3), axis=0, return_inverse=True)
    inv = inv.ravel()
    faces = inv.reshape(-1, 3)
    nv = int(inv.max()) + 1
    rows = np.concatenate([faces[:, 0], faces[:, 1]])
    cols = np.concatenate([faces[:, 1], faces[:, 2]])
    g = sparse.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(nv, nv))
    _, lab = connected_components(g, directed=False)
    face_lab = lab[faces[:, 0]]
    area = triangle_areas(tri)
    comp_area = np.bincount(face_lab, weights=area)
    best = int(np.argmax(comp_area))
    keep = face_lab == best
    return tri[keep], float(comp_area[best] / comp_area.sum())


def triangle_areas(tri: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def signed_volume(tri: np.ndarray) -> float:
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def sample_surface(tri: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    area = triangle_areas(tri)
    idx = rng.choice(len(tri), size=n, p=area / area.sum())
    u, w = rng.random((2, n))
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])


def pca_frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre, rotation (rows = axes, longest first) and per-axis SD, with axis signs fixed by skewness."""
    c = pts.mean(0)
    x = pts - c
    evals, evecs = np.linalg.eigh(np.cov(x.T))
    order = np.argsort(evals)[::-1]
    R = evecs[:, order].T
    proj = x @ R.T
    skew = (proj**3).mean(0)
    R = R * np.where(skew < 0, -1.0, 1.0)[:, None]
    if np.linalg.det(R) < 0:  # keep a proper rotation; flip the least informative axis
        R[2] *= -1
    return c, R, np.sqrt(evals[order])


def _hist(x: np.ndarray, lo: float, hi: float, bins: int = N_BINS) -> np.ndarray:
    h, _ = np.histogram(np.clip(x, lo, hi - 1e-9), bins=bins, range=(lo, hi))
    return h / max(h.sum(), 1)


def descriptors(pts: np.ndarray, area: float, volume: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Pose-robust global shape descriptors. Sizes are in mm; distributions are scale-free."""
    c, R, sd = pca_frame(pts)
    al = (pts - c) @ R.T
    rms = float(np.sqrt((al**2).sum(1).mean()))
    extent = np.percentile(al, 99, axis=0) - np.percentile(al, 1, axis=0)
    i, j, k = rng.integers(0, len(pts), size=(3, 20000))
    d2 = np.linalg.norm(pts[i] - pts[j], axis=1) / rms
    r = np.linalg.norm(al, axis=1) / rms
    a = pts[i] - pts[k]
    b = pts[j] - pts[k]
    cosang = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    a3 = np.arccos(np.clip(cosang, -1, 1))
    # per-axis profile: where along the long axis the surface sits (head / mesosoma / gaster bulges)
    prof = _hist(al[:, 0] / rms, -3.0, 3.0, 24)
    sphericity = (np.pi ** (1 / 3)) * (6 * abs(volume)) ** (2 / 3) / area if area > 0 else np.nan
    return {
        "log_size": np.log(np.array([rms, abs(volume) ** (1 / 3) + 1e-12, np.sqrt(area)] + list(extent))),
        "axis_ratio": np.log(np.array([sd[1] / sd[0], sd[2] / sd[0], sd[2] / sd[1]])),
        "sphericity": np.array([sphericity]),
        "d2": _hist(d2, 0, 4.0),
        "radial": _hist(r, 0, 3.0),
        "a3": _hist(a3, 0, np.pi),
        "long_profile": prof,
    }


def solid_grid(pts_aligned: np.ndarray, grid: int = GRID) -> tuple[np.ndarray, float, np.ndarray]:
    """Voxelise a dense, aligned surface sample into a filled solid. Returns (occupancy, voxel size mm, origin)."""
    lo = pts_aligned.min(0)
    span = pts_aligned.max(0) - lo
    h = float(span.max()) / (grid - 4)
    origin = lo - 2 * h
    shape = np.ceil(span / h).astype(int) + 5
    idx = np.floor((pts_aligned - origin) / h).astype(int)
    occ = np.zeros(shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    occ = ndimage.binary_closing(occ, iterations=1)
    occ = ndimage.binary_fill_holes(occ)
    return occ, h, origin


def body_core(occ: np.ndarray, radius: int = CORE_RADIUS) -> np.ndarray:
    """Morphological opening with a ball: thin appendages vanish, the thick body tagmata stay."""
    r = radius
    zz, yy, xx = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
    ball = (xx**2 + yy**2 + zz**2) <= r * r
    core = ndimage.binary_opening(occ, structure=ball)
    lab, n = ndimage.label(core)
    if n > 1:  # drop specks; keep pieces holding at least 2% of the core
        sizes = ndimage.sum(core, lab, range(1, n + 1))
        core = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.02 * sizes.sum()))
    return core


def core_descriptors(core: np.ndarray, h: float) -> dict[str, np.ndarray]:
    """Descriptors of the appendage-free body: size, proportions and the cross-section profile along its axis."""
    if core.sum() < 50:
        return {k: np.full(n, np.nan) for k, n in
                [("core_log_size", 4), ("core_axis_ratio", 2), ("core_profile", N_PROFILE), ("core_width", N_PROFILE)]}
    xyz = np.argwhere(core) * h
    c, R, sd = pca_frame(xyz)
    al = (xyz - c) @ R.T
    L = float(al[:, 0].max() - al[:, 0].min())
    vol = core.sum() * h**3
    t = (al[:, 0] - al[:, 0].min()) / max(L, 1e-9)
    st = np.minimum((t * N_PROFILE).astype(int), N_PROFILE - 1)
    # cross-section area per station, as a share of total volume: the head / petiole / gaster rhythm
    prof = np.bincount(st, minlength=N_PROFILE).astype(float)
    prof /= prof.sum()
    width = np.zeros(N_PROFILE)
    for s_ in range(N_PROFILE):
        m = st == s_
        if m.sum() > 2:
            width[s_] = np.sqrt(al[m, 1].var() + al[m, 2].var()) / L
    return {
        "core_log_size": np.log([L, vol ** (1 / 3), sd[1], sd[2]]),
        "core_axis_ratio": np.log([sd[1] / sd[0], sd[2] / sd[0]]),
        "core_profile": prof,
        "core_width": width,
    }


THICK_RES = 30  # voxels per cube-root of body volume for the thickness grid (pose does not change it)
GRANULE_RADII = (1, 2, 3, 4, 6, 8, 11)


def solid_grid_at(pts_aligned: np.ndarray, h: float) -> np.ndarray:
    lo = pts_aligned.min(0) - 2 * h
    shape = np.ceil((pts_aligned.max(0) - lo) / h).astype(int) + 3
    idx = np.floor((pts_aligned - lo) / h).astype(int)
    occ = np.zeros(shape, dtype=bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1))


def thickness_descriptors(pts_aligned: np.ndarray, volume: float) -> dict[str, np.ndarray]:
    """Pose-invariant thickness: how the body volume is distributed over local thicknesses.

    The grid step is tied to body volume, not to the bounding box, so splayed legs don't change it. Two views:
    the distribution of inscribed-ball radii (distance transform) over the solid, and a granulometry, the share
    of volume that survives morphological opening with balls of growing radius (legs vanish first, then
    the petiole and head, the gaster last).
    """
    h = abs(volume) ** (1 / 3) / THICK_RES
    occ = solid_grid_at(pts_aligned, h)
    dt = ndimage.distance_transform_edt(occ)[occ]
    hist = _hist(dt, 0, 16.0, 24)
    tot = occ.sum()
    gran = []
    for r in GRANULE_RADII:
        zz, yy, xx = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
        ball = (xx**2 + yy**2 + zz**2) <= r * r
        gran.append(ndimage.binary_opening(occ, structure=ball).sum() / max(tot, 1))
    return {"thickness": hist, "granulometry": np.array(gran),
            "thick_quantiles": np.percentile(dt, [50, 75, 90, 99]) / THICK_RES}


def downsample_grid(occ: np.ndarray, size: int = 32) -> np.ndarray:
    """Centre an occupancy volume in a size^3 cube (isotropic, longest axis fills it) by block averaging."""
    f = int(np.ceil(max(occ.shape) / size))
    pad = [(0, (-s) % f) for s in occ.shape]
    o = np.pad(occ, pad).astype(np.float32)
    o = o.reshape(o.shape[0] // f, f, o.shape[1] // f, f, o.shape[2] // f, f).mean((1, 3, 5))
    out = np.zeros((size,) * 3, np.float32)
    off = [(size - s) // 2 for s in o.shape]
    sl = tuple(slice(a, a + s) for a, s in zip(off, o.shape))
    out[sl] = o[: size, : size, : size][tuple(slice(0, s) for s in out[sl].shape)]
    return out


@dataclass
class ShapeRecord:
    points: np.ndarray  # (N_SURFACE, 3) float32, mm, PCA-aligned and centred
    desc: dict[str, np.ndarray]
    n_triangles: int
    kept_area_frac: float
    volume_mm3: float
    area_mm2: float
    solid32: np.ndarray  # (32,32,32) float32 occupancy fraction, whole body incl. appendages
    core32: np.ndarray  # (32,32,32) appendage-free body


def stl_unit_mm(voxel_um: float) -> float:
    """Length of one STL coordinate unit in mm.

    Most Antscan STLs are in voxels. The stitched 6.11 um scans of large ants are already in micrometres:
    scaling them by the voxel size makes a Paraponera worker 160 mm long instead of 26 mm.
    """
    return 1e-3 if np.isclose(voxel_um, 6.11) else voxel_um / 1000.0


def process_stl(path: str, voxel_um: float, seed: int = 0) -> ShapeRecord:
    tri = read_binary_stl(path) * stl_unit_mm(voxel_um)
    n_tri = len(tri)
    tri, frac = largest_component(tri)
    rng = np.random.default_rng(seed)
    area = float(triangle_areas(tri).sum())
    vol = signed_volume(tri)
    pts = sample_surface(tri, N_SURFACE, rng)
    desc = descriptors(pts, area, vol, rng)
    c, R, _ = pca_frame(pts)
    aligned = ((pts - c) @ R.T).astype(np.float32)
    dense = (sample_surface(tri, 400_000, rng) - c) @ R.T
    occ, h, _ = solid_grid(dense)
    core = body_core(occ)
    desc.update(core_descriptors(core, h))
    desc.update(thickness_descriptors(dense, vol))
    return ShapeRecord(aligned, desc, n_tri, frac, abs(vol), area, downsample_grid(occ), downsample_grid(core))


def download(url: str, dest: str, retries: int = 4) -> None:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            return
        except Exception as e:  # network hiccups: back off and retry
            if attempt == retries:
                raise
            log.warning("download failed (%s), retry %d", e, attempt + 1)
            time.sleep(2 ** (attempt + 1))


def fetch_and_process(stl_path: str, voxel_um: float, tmpdir: str | None = None, seed: int = 0) -> ShapeRecord:
    """Stream one STL from Biomedisa (`stl_path` like '/media/antscan/processed/5x/24-42.stl'), reduce, delete."""
    fd, tmp = tempfile.mkstemp(suffix=".stl", dir=tmpdir)
    os.close(fd)
    try:
        download(BASE_URL + stl_path, tmp)
        return process_stl(tmp, voxel_um, seed)
    finally:
        os.remove(tmp)


def flat_descriptor(desc: dict[str, np.ndarray], keys: tuple[str, ...] | None = None) -> tuple[np.ndarray, list[str]]:
    keys = keys or tuple(desc)
    vals, names = [], []
    for k in keys:
        v = np.atleast_1d(desc[k])
        vals.append(v)
        names += [f"{k}_{i}" for i in range(len(v))]
    return np.concatenate(vals), names
