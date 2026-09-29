"""Whole-slice winding field: Laplace-oriented phase, Fried split, 2D unwrapping."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage as ndi
from skimage.restoration import unwrap_phase

from . import data, phase as ph


def umbilicus_xy(z_full, scroll="PHercParis4"):
    """Umbilicus (x, y) in full-res voxels at full-res slice z (linear interpolation)."""
    cp = data.umbilicus_for(scroll)["control_points"]
    zs = np.array([p["z"] for p in cp], float)
    o = np.argsort(zs)
    xs = np.array([p["x"] for p in cp], float)[o]; ys = np.array([p["y"] for p in cp], float)[o]
    return float(np.interp(z_full, zs[o], xs)), float(np.interp(z_full, zs[o], ys))


def scroll_mask(c_u8):
    m = ndi.binary_closing(c_u8 > 0, iterations=4)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    return m


def laplace_radial(mask, center, down=4, iters=4000):
    """Harmonic 'radial' coordinate: 0 at the umbilicus, 1 on the scroll boundary."""
    m = mask[::down, ::down]
    H, W = m.shape
    cy, cx = center[1] / down, center[0] / down
    yy, xx = np.mgrid[0:H, 0:W]
    src = np.hypot(yy - cy, xx - cx) <= 2.5
    inside = m & ~src
    idx = -np.ones(m.shape, int); idx[inside] = np.arange(inside.sum())
    n = inside.sum()
    rows, cols, vals = [], [], []
    b = np.zeros(n)
    iy, ix = np.nonzero(inside)
    diag = np.zeros(n)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = iy + dy, ix + dx
        ok = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        diag += 1
        nb_in = np.zeros(n, bool); nb_in[ok] = inside[ny[ok], nx[ok]]
        nb_src = np.zeros(n, bool); nb_src[ok] = src[ny[ok], nx[ok]]
        k = np.nonzero(nb_in)[0]
        rows.append(k); cols.append(idx[ny[k], nx[k]]); vals.append(-np.ones(len(k)))
        # boundary (outside mask) = 1, source = 0
        b += (~nb_in & ~nb_src).astype(float)
    A = sp.csr_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(n)]), np.concatenate(cols + [np.arange(n)]))), shape=(n, n))
    x0 = np.clip(np.hypot(iy - cy, ix - cx) / max(H, W) * 2, 0, 1)
    x, _ = spla.cg(A, b, x0=x0, maxiter=iters, rtol=1e-8)
    R = np.ones(m.shape); R[src] = 0; R[inside] = x
    R = ndi.zoom(R, down, order=1)[: mask.shape[0], : mask.shape[1]]
    pad = [(0, mask.shape[0] - R.shape[0]), (0, mask.shape[1] - R.shape[1])]
    return np.pad(R, pad, mode="edge")


def agreement_quality(psi, gm_l3, mask, k=0.808, tol=0.35, sigma=2.0):
    """Quality from agreement of local fringe frequency with Lasagna wrap density.

    psi: wrapped phase (radians) at L3; gm_l3: decoded grad_mag (per full-res voxel) resampled to L3.
    Returns q in [0,1]: 1 where |f_phase - f_density| / f_density is small.
    """
    gy = ph.wrap(np.diff(psi, axis=0, append=psi[-1:, :]))
    gx = ph.wrap(np.diff(psi, axis=1, append=psi[:, -1:]))
    fph = ndi.gaussian_filter(np.hypot(gx, gy) / (2 * np.pi), sigma)
    fden = ndi.gaussian_filter(gm_l3 * 8.0 * k, sigma)
    rel = np.abs(fph - fden) / (fden + 1e-3)
    q = np.exp(-(rel / tol) ** 2)
    return np.where(mask, q, 0.0)


def winding_field(c_u8, center, s=+1, sigma=1.5, unwrap=True):
    """Return dict with phi, psi, U (unwrapped, in turns) and W = U - s*theta/2pi."""
    mask = scroll_mask(c_u8)
    c = c_u8.astype(float) / 255 * 2 - 1
    e, rx, ry = ph.riesz2d(c * mask)
    phi, mx, my, coh = ph.phase_with_linefield(e, rx, ry, sigma=sigma)
    R = laplace_radial(mask, center)
    gy, gx = np.gradient(ndi.gaussian_filter(R, 3))
    sgn = np.sign(mx * gx + my * gy); sgn[sgn == 0] = 1
    phi = phi * sgn
    H, W = c.shape
    yy, xx = np.mgrid[0:H, 0:W]
    th = np.arctan2(yy - center[1], xx - center[0])
    psi = ph.wrap(phi + s * th)
    out = dict(mask=mask, phi=phi, psi=psi, theta=th, R=R, amp=np.sqrt(e**2 + rx**2 + ry**2), mx=mx * sgn, my=my * sgn)
    if unwrap:
        ma = np.ma.masked_array(psi, mask=~mask)
        Uw = unwrap_phase(ma)
        U = np.asarray(Uw.filled(np.nan)) / (2 * np.pi)
        out["U"] = U
        out["W"] = U - s * th / (2 * np.pi)
    return out
