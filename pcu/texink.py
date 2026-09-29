"""Model-free ink cue: texture (gradient) energy on the layers at and just outside the fitted surface,
normalized by its local median (removes scan-level contrast differences between scrolls).
On Paris 4 at 9.6 um, gradient energy at layers 10-13 separates ink from bare papyrus with d' 0.36-0.47
per pixel, vs 0.18 for any linear combination of intensities (E50)."""
import numpy as np
from scipy import ndimage as ndi


def texture_energy(sv, layers=(10, 11, 12, 13), sigma=6, norm=60):
    """sv: (L, H, W) surface volume, layer 0 = inner face, centre = fitted surface. Returns (H, W) float map."""
    e = 0
    for L in layers:
        l = sv[L].astype(np.float32)
        gx = ndi.sobel(l, 1); gy = ndi.sobel(l, 0)
        e = e + ndi.gaussian_filter(gx * gx + gy * gy, sigma)
    valid = sv[sv.shape[0] // 2] > 0
    e = np.log1p(e)
    med = ndi.median_filter(np.where(valid, e, np.nan).astype(np.float32), size=9) if False else ndi.gaussian_filter(np.where(valid, e, 0), norm) / (ndi.gaussian_filter(valid.astype(np.float32), norm) + 1e-6)
    out = np.where(valid, e - med, 0)
    return out
