"""Outer shell (tifxyz) for scrolls without one, from the masked CT.

The spiral fit uses the outer shell to drop tracks outside the scroll and to pin the outermost winding
(ShellEnvelope: target radius per z and angle around the umbilicus). The published volumes are masked
(outside the scroll = 0), so per z we take the scroll mask on a coarse level, keep its largest component,
and for each angle around the umbilicus the outermost papyrus radius. Rows = z every `step` voxels,
columns = angle bins spaced ~`step` voxels along the outer boundary; coordinates in the fit frame."""
import json
import os

import numpy as np
from scipy import ndimage as ndi

from . import data


def _umb_interp(ctrl):
    cz = np.array([p["z"] for p in ctrl], float); o = np.argsort(cz)
    cx = np.array([p["x"] for p in ctrl], float)[o]; cy = np.array([p["y"] for p in ctrl], float)[o]; cz = cz[o]
    return lambda z: (np.interp(z, cz, cx), np.interp(z, cz, cy))


def outer_radius(mask, cx, cy, thetas, rmax):
    """mask: 2D bool (coarse). For each theta, largest r (coarse px) with mask True along the ray."""
    r = np.arange(0, rmax, 0.5)
    X = cx + r[None, :] * np.cos(thetas)[:, None]; Y = cy + r[None, :] * np.sin(thetas)[:, None]
    ok = (X >= 0) & (Y >= 0) & (X < mask.shape[1] - 1) & (Y < mask.shape[0] - 1)
    vals = np.zeros_like(X, bool)
    vals[ok] = mask[np.round(Y[ok]).astype(int), np.round(X[ok]).astype(int)]
    has = vals.any(1)
    last = vals.shape[1] - 1 - np.argmax(vals[:, ::-1], axis=1)
    return np.where(has, r[last], np.nan)


def _papyrus(sl, close, frac=1.0):
    """Papyrus mask of one coarse slice: Otsu on the masked (nonzero) voxels, closed, largest component."""
    from skimage.filters import threshold_otsu
    nz = sl[sl > 0]
    if nz.size < 100:
        return None
    m = sl > frac * threshold_otsu(nz)
    m = ndi.binary_closing(m, iterations=close)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (1 + np.argmax(ndi.sum(m, lab, range(1, n + 1))))
    return m


def _surface_mask(sl, close, thr=128, ct=None):
    """Mask from a surface-prediction slice: predicted papyrus surfaces (inside the CT mask when given),
    closed and filled."""
    m = sl >= thr
    if ct is not None:
        m &= ndi.binary_erosion(ct > 0, iterations=2)
    if m.sum() < 100:
        return None
    m = ndi.binary_closing(m, iterations=close)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        m = lab == (1 + np.argmax(ndi.sum(m, lab, range(1, n + 1))))
    return m


def build(vol, factor, umb_ctrl, z0, z1, step=20.0, close=3, frac=1.0, surface=False, ctvol=None, pad=0.0):
    """vol: coarse CT level (masked, 0 outside); factor: fit-frame voxels per coarse voxel.
    Returns (3, rows, cols) float32 grid in the fit frame (-1 invalid)."""
    umb = _umb_interp(umb_ctrl)
    zs = np.arange(z0, z1, step)
    def mask_at(zc):
        sl = data.read(vol, (slice(zc, zc + 1), slice(0, vol.shape[1]), slice(0, vol.shape[2])))[0]
        if surface:
            ct = None if ctvol is None else data.read(ctvol, (slice(zc, zc + 1), slice(0, ctvol.shape[1]), slice(0, ctvol.shape[2])))[0]
            return _surface_mask(sl, close, ct=ct)
        return _papyrus(sl, close, frac)

    # column count from the mid-band outline length (~`step` voxels per column)
    m0 = mask_at(int((z0 + z1) / 2 / factor))
    per = max(np.sum(m0 ^ ndi.binary_erosion(m0)), 100) * factor if m0 is not None else 20000
    ncol = int(per / step)
    thetas = np.linspace(-np.pi, np.pi, ncol, endpoint=False)
    out = np.full((3, len(zs), ncol), -1, np.float32)
    rmax = np.hypot(*vol.shape[1:])
    for i, z in enumerate(zs):
        zc = int(z / factor)
        if zc >= vol.shape[0]:
            continue
        m = mask_at(zc)
        if m is None:
            continue
        ux, uy = umb(z)
        r = outer_radius(m, ux / factor, uy / factor, thetas, rmax) + pad / factor
        ok = np.isfinite(r)
        out[0, i, ok] = (ux / factor + r[ok] * np.cos(thetas[ok])) * factor
        out[1, i, ok] = (uy / factor + r[ok] * np.sin(thetas[ok])) * factor
        out[2, i, ok] = z
    return out


def save_tifxyz(grid, path, uuid="pcu_outer_shell", step=20.0):
    import tifffile
    os.makedirs(path, exist_ok=True)
    for a, n in enumerate("xyz"):
        tifffile.imwrite(os.path.join(path, f"{n}.tif"), grid[a].astype(np.float32))
    v = grid[0] > 0
    bb = [[float(grid[a][v].min()) for a in range(3)], [float(grid[a][v].max()) for a in range(3)]]
    json.dump({"bbox": bb, "format": "tifxyz", "scale": [step, step], "type": "seg", "uuid": uuid},
              open(os.path.join(path, "meta.json"), "w"), indent=1)
