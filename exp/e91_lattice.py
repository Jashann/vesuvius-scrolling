"""E91: lattice test for text vs texture on ink-probability maps. Bookhand text is a 2D quasi-lattice: lines
repeat at 4-6 mm, letters along a line at about 2-3.5 mm. Blob centroids (components above the map's 95th
percentile) are treated as points; their pair-displacement histogram g(dx, dy), normalised by a shuffled null
(random points in the same valid mask), should show a line peak on the vertical axis at 4-6 mm and a letter
peak on the horizontal axis at 1.5-4 mm. Texture/void blobs give a ring or nothing.
Usage: e91_lattice.py NAME:MAP[:PX_UM] ...   (MAP = png or npy probability map; rows must run along z)"""
import sys, os, json
import numpy as np
import imageio.v2 as imageio
from scipy import ndimage as ndi

OUT = "runs/e91"; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
BIN = 0.5; RMAX = 8.0  # mm


def load(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    a = imageio.imread(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float32) / 255


def centroids(p, valid, px_mm):
    thr = np.percentile(p[valid], 95)
    lab, n = ndi.label((p > thr) & valid)
    if n == 0:
        return np.zeros((0, 2))
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s * px_mm * px_mm >= 0.01]  # >= 0.01 mm^2
    if not keep:
        return np.zeros((0, 2))
    return np.array(ndi.center_of_mass(np.ones_like(lab), lab, keep)) * px_mm  # (y, x) in mm


def pair_hist(P):
    nb = int(RMAX / BIN); H = np.zeros((2 * nb + 1, 2 * nb + 1))
    if len(P) < 5:
        return H
    for i in range(len(P)):
        d = P - P[i]; d = d[(np.abs(d) < RMAX).all(1) & (np.arange(len(P)) != i)]
        iy = np.round(d[:, 0] / BIN).astype(int) + nb; ix = np.round(d[:, 1] / BIN).astype(int) + nb
        np.add.at(H, (iy, ix), 1)
    return H


def score(p, valid, px_mm, name):
    P = centroids(p, valid, px_mm)
    H = pair_hist(P)
    yy, xx = np.nonzero(valid)
    N = np.zeros_like(H)
    for _ in range(20):
        j = rng.choice(len(yy), len(P), replace=False)
        N += pair_hist(np.stack([yy[j], xx[j]], 1) * px_mm)
    N = ndi.gaussian_filter(N / 20, 1.0); Hs = ndi.gaussian_filter(H, 1.0)
    g = np.where(N > 0.5, Hs / np.maximum(N, 1e-6), np.nan)
    nb = int(RMAX / BIN); ax = (np.arange(2 * nb + 1) - nb) * BIN
    band = np.abs(ax) <= 1.0
    # line axis: vertical displacements (dy) with |dx| <= 0.75 mm; letter axis: horizontal with |dy| <= 0.75
    gv = np.nanmean(g[:, band], 1); gh = np.nanmean(g[band, :], 0)
    def peak(prof, lo, hi):
        m = (np.abs(ax) >= lo) & (np.abs(ax) <= hi); base = (np.abs(ax) >= 1.0) & (np.abs(ax) <= RMAX)
        if not np.isfinite(prof[m]).any():
            return None, None
        return round(float(np.nanmax(prof[m])), 2), round(float(ax[m][np.nanargmax(prof[m])]), 2)
    line_pk, line_at = peak(gv, 3.5, 6.5); let_pk, let_at = peak(gh, 1.5, 4.0)
    # ring baseline: azimuthal mean of g at the same radii
    r = np.hypot(*np.meshgrid(ax, ax, indexing="ij"))
    ring = np.nanmean(g[(r >= 3.5) & (r <= 6.5)])
    res = dict(name=name, n_blobs=int(len(P)), line_peak=line_pk, line_at_mm=line_at, letter_peak=let_pk, letter_at_mm=let_at,
               ring_mean_3p5_6p5=round(float(ring), 2) if np.isfinite(ring) else None,
               line_over_ring=round(float(line_pk / ring), 2) if line_pk and ring else None)
    img = np.nan_to_num(g, nan=0.0); img = np.clip(img / max(np.nanpercentile(img, 99), 1e-6), 0, 1)
    imageio.imwrite(f"{OUT}/{name}_g.png", (ndi.zoom(img, 6, order=0) * 255).astype(np.uint8))
    return res


results = []
for arg in sys.argv[1:]:
    parts = arg.split(":"); name, path = parts[0], parts[1]; px_um = float(parts[2]) if len(parts) > 2 else 9.362
    p = load(path); valid = p > 0
    if valid.mean() < 0.05:
        print(name, "empty"); continue
    r = score(p, valid, px_um / 1000, name); results.append(r); print(json.dumps(r), flush=True)
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
