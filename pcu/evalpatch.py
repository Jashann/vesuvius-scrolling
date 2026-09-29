"""Verified-patch consistency evaluation with branch-cut correction.

Wrap labels step by +1 where a sheet crosses the branch cut (theta = +-pi), by construction. A
verified patch lies on one sheet, so before scoring we move the cut, per patch, to the middle of the
largest angular gap the patch does not cover: label' = label - 1 for points with theta < theta_gap.
"""
import json, os
import numpy as np
import tifffile
from scipy import ndimage as ndi

X0, Y0 = 672, 416


def patch_points(z3, metas, root="data/patches", min_area=0.3):
    out = []
    for nme, m in metas.items():
        if not (m["bbox"][0][2] + 5 < 2 * z3 < m["bbox"][1][2] - 5 and m.get("area_cm2", 0) > min_area):
            continue
        d = f"{root}/{nme}"
        if not os.path.exists(f"{d}z.tif"):
            continue
        x, y, z = (tifffile.imread(f"{d}{a}.tif") for a in "xyz")
        pts = []; valid = (x > 0) & (z > 0)
        for sa, sb in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))), ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            za, zb = z[sa], z[sb]
            cr = valid[sa] & valid[sb] & ((za - 2 * z3) * (zb - 2 * z3) <= 0) & (za != zb)
            t = (2 * z3 - za[cr]) / (zb[cr] - za[cr])
            pts.append(np.stack([x[sa][cr] + t * (x[sb][cr] - x[sa][cr]), y[sa][cr] + t * (y[sb][cr] - y[sa][cr])], 1))
        pts = np.concatenate(pts)
        if len(pts) >= 10:
            out.append((nme, np.stack([pts[:, 0] / 2 - X0, pts[:, 1] / 2 - Y0], 1)))
    return out


def cut_corrected(values, theta):
    """values: labels (or continuous W) at patch points; theta: angle around the umbilicus."""
    ts = np.sort(theta)
    gaps = np.diff(np.r_[ts, ts[0] + 2 * np.pi])
    k = np.argmax(gaps)
    tg = ts[k] + gaps[k] / 2
    tg = (tg + np.pi) % (2 * np.pi) - np.pi
    return values - (theta < tg).astype(float) + (0 if tg > -np.pi else 0)


def score(patches, sample_fn, center, continuous=False):
    """sample_fn(xy) -> values (nan where unlabeled). Returns dict of totals."""
    T = dict(ok=0, n=0, sw=0, np=0)
    for nme, xy in patches:
        v = sample_fn(xy)
        th = np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0])
        f = np.isfinite(v)
        if f.sum() < 10:
            continue
        v = cut_corrected(v[f], th[f])
        if continuous:
            okm = np.abs(v - np.median(v)) < 0.5
            good = int(okm.sum())
        else:
            vals, cnt = np.unique(v, return_counts=True)
            good = int(cnt.max())
        T["ok"] += good; T["n"] += int(f.sum()); T["np"] += 1; T["sw"] += int(good < 0.9 * f.sum())
    return T
