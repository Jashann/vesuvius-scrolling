"""Umbilicus (scroll centre line) from the Lasagna normal field, for scrolls without a published one.

In a cross-section every sheet normal points (almost) radially away from the umbilicus, so the centre is
the point closest to all normal lines: argmin_c sum_i w_i |(I - n_i n_i^T)(c - p_i)|^2, a 2x2 linear
solve. Crushed regions have normals that do not point at the centre, so we reweight robustly (Cauchy on
the point-to-line distance, shrinking scale) and fit a smooth line through the per-slice centres."""
import numpy as np

from . import data


def slice_centre(nx, ny, stride=4, iters=8):
    """nx, ny: u8 Lasagna normal components of one slice (0 = no data). Returns (cx, cy, inlier_frac) in
    the slice's pixel frame."""
    nx = nx[::stride, ::stride].astype(np.float32); ny = ny[::stride, ::stride].astype(np.float32)
    m = (nx > 0) | (ny > 0)
    yy, xx = np.nonzero(m)
    u = (nx[m] - 128) / 127; v = (ny[m] - 128) / 127
    nrm = np.hypot(u, v)
    k = nrm > 0.3
    u, v, xx, yy = u[k] / nrm[k], v[k] / nrm[k], xx[k] * stride, yy[k] * stride
    if len(u) < 100:
        return None
    w = np.ones_like(u)
    c = np.array([xx.mean(), yy.mean()])
    scale = 0.25 * max(np.ptp(xx), np.ptp(yy))
    for _ in range(iters):
        # P = I - n n^T
        a11 = w * (1 - u * u); a12 = w * (-u * v); a22 = w * (1 - v * v)
        A = np.array([[a11.sum(), a12.sum()], [a12.sum(), a22.sum()]])
        b = np.array([(a11 * xx + a12 * yy).sum(), (a12 * xx + a22 * yy).sum()])
        c = np.linalg.solve(A, b)
        dx, dy = c[0] - xx, c[1] - yy
        d = np.abs(dx * v - dy * u)  # distance from c to the normal line through p
        w = 1 / (1 + (d / scale) ** 2)
        scale = max(scale * 0.6, 5.0)
    return float(c[0]), float(c[1]), float(np.mean(d < 3 * scale))


def slice_centre_vc(nx, ny, px=4.0, n=10000, floor=100.0, seed=0):
    """The team's objective (volume-cartographer align_and_extract_umbilicus): maximise
    sum_i cos^2(n_i, p_i - c) / max(floor, |p_i - c|), a proximity-weighted sum rather than a mean, which
    they report at 0.4-7 mm median error on 15 scrolls where the mean fails. px: volume voxels per
    Lasagna pixel (floor is in volume voxels). Grid search then pattern-search hill climb."""
    m = (nx > 0) | (ny > 0)
    yy, xx = np.nonzero(m)
    u = (nx[m].astype(np.float32) - 128) / 127; v = (ny[m].astype(np.float32) - 128) / 127
    nrm = np.hypot(u, v); k = nrm > 0.3
    if k.sum() < 100:
        return None
    rng = np.random.default_rng(seed)
    i = rng.choice(np.flatnonzero(k), min(n, int(k.sum())), replace=False)
    P = np.stack([xx[i], yy[i]], 1).astype(np.float64) * px
    N = np.stack([u[i], v[i]], 1) / nrm[i, None]

    def score(c):
        d = P - c; r = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)
        cos = (d * N).sum(1) / r
        return float((cos * cos / np.maximum(floor, r)).sum())
    H, W = nx.shape
    cands = [np.array([x, y]) * px for x in np.linspace(0, W, 33) for y in np.linspace(0, H, 33)]
    s = [score(c) for c in cands]
    c = cands[int(np.argmax(s))]; best = max(s); step = 1024.0
    while step >= 1.0:
        moved = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    q = c + step * np.array([dx, dy]); sq = score(q)
                    if sq > best:
                        best, c, moved = sq, q, True
        if not moved:
            step /= 2
    return float(c[0]), float(c[1]), best


def estimate(scroll, zs, las_level=2, las_scale=4, method="vc"):
    """Per-slice centres at full-resolution z values `zs`; returns list of dict(x, y, z, inlier)."""
    ax = data.lasagna_for(scroll, "nx", las_level); ay = data.lasagna_for(scroll, "ny", las_level)
    out = []
    for z in zs:
        zl = int(z // las_scale)
        if zl >= ax.shape[0]:
            continue
        sl = (slice(zl, zl + 1), slice(0, ax.shape[1]), slice(0, ax.shape[2]))
        a, b = data.read(ax, sl)[0], data.read(ay, sl)[0]
        if method == "vc":
            r = slice_centre_vc(a, b, px=las_scale)
            if r is not None:
                out.append(dict(x=r[0], y=r[1], z=float(z), inlier=r[2]))
            continue
        r = slice_centre(a, b)
        if r is not None:
            out.append(dict(x=r[0] * las_scale, y=r[1] * las_scale, z=float(z), inlier=r[2]))
    return out
