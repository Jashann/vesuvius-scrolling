"""Snap a fitted winding surface onto the papyrus sheet it belongs to.

The global spiral fit gets the winding topology from tracks and constraints, but its smooth 20-voxel surface
can sit several voxels off the sheet, or in the gap between two sheets, where the ink model sees nothing.
We sample a sheetness signal (the team's m7 surface prediction, same voxel frame as the CT we render)
along each vertex normal within +-R, take the peak offset, smooth the offset field robustly over the grid
(median then Gaussian, weighted by peak strength) and move the vertices. A few passes with shrinking R.
R stays below half the sheet spacing, so the winding assignment of the global fit is kept."""
import numpy as np
from scipy import ndimage as ndi

from . import render


def _pass(P, vm, sig, center_xy, R, med, sm, step=0.5):
    L = int(round(2 * R / step)) + 1
    st, vm2 = render.render(P, sig, factor=1, layers=L, spacing=step, center_xy=center_xy)
    vm = vm & vm2
    prof = ndi.gaussian_filter1d(st.astype(np.float32), 1.0, axis=0)
    am = np.argmax(prof, 0)
    peak = np.take_along_axis(prof, am[None], 0)[0]
    base = np.median(prof, 0)
    off = (am - (L - 1) / 2) * step
    w = np.clip(peak - base, 0, None) * vm
    # robust smoothing: weighted median via masked median filter, then weighted Gaussian
    o = np.where(w > 0, off, np.nan)
    om = ndi.generic_filter(o, np.nanmedian, size=med, mode="nearest")
    om = np.where(np.isnan(om), 0, om)
    wd = ndi.gaussian_filter(w, sm) + 1e-6
    os_ = ndi.gaussian_filter(om * w, sm) / wd
    n = render.normals(P, center_xy)
    Pn = P + n * os_[None]
    Pn[:, ~vm] = -1
    return Pn, vm, dict(R=R, off_med=float(np.median(off[vm])), moved_med=float(np.median(np.abs(os_[vm]))),
                         peak_med=float(np.median(peak[vm])))


def refine(grid, sig, factor=4, center_xy=None, radii=(7, 4, 2), med=7, sm=2.0):
    """grid: (3, h, w) in the `sig` voxel frame. Returns (refined grid (3, h*factor, w*factor), stats)."""
    P, vm = render.densify(grid, factor)
    P = P.astype(np.float64)
    P[:, ~vm] = -1
    stats = []
    for R in radii:
        P, vm, s = _pass(P, vm, sig, center_xy, R, med, sm)
        stats.append(s)
    return P.astype(np.float32), stats
