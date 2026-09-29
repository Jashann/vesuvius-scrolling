"""Fit-free winding surfaces from the per-slice PCU phase field (CPU only).

Per slice, the unwrapped winding field W = U - theta/2pi (U = unwrapped psi = phi + theta, in turns) is
constant along a sheet and steps by one per sheet crossing, so the centre of winding k is the contour W = k.
We extract that contour per slice, parametrize it by the angle around the umbilicus, and stack slices into a
(z, angle) grid: a tifxyz-style surface for winding k over any z range, no GPU fit needed. Unwrap errors make
some slices jump a sheet; downstream we only need local (few cm) correctness to search for text."""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

from . import data, slice2d, mcfcut


def slice_W(scroll, z, las_level=1, las_px=2.0):
    """Unwrapped winding field of one slice. Returns (W (h, w) in turns, NaN outside), X0, Y0, scale) with full
    resolution coords = (col + X0) * scale."""
    cos = data.lasagna_for(scroll, "cos", las_level)
    zc = int(z // las_px)
    full = data.read(cos, (slice(zc, zc + 1), slice(0, cos.shape[1]), slice(0, cos.shape[2])))[0]
    m = ndi.binary_dilation(full > 0, iterations=20); ys, xs = np.nonzero(m)
    Y0, Y1, X0, X1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = full[Y0:Y1, X0:X1]
    ux, uy = slice2d.umbilicus_xy(z, scroll)
    cen = (ux / las_px - X0, uy / las_px - Y0)
    F = slice2d.winding_field(c, cen, s=+1, unwrap=False)
    q = np.where(F["mask"], F["amp"] / (np.percentile(F["amp"][F["mask"]], 90) + 1e-9), 0).clip(0, 1) ** 2
    U, cut, nres, _ = mcfcut.unwrap_mcf(F["psi"], F["mask"], q, geo=True)
    W = U - F["theta"] / (2 * np.pi)
    W[~F["mask"]] = np.nan
    return W, X0, Y0, las_px, cen


def winding_contour(W, k, cen, n_theta):
    """Contour W = k (longest piece) sampled at n_theta angle bins around cen. Returns (n_theta, 2) x, y in the
    W frame, NaN where the contour does not reach that angle."""
    valid = ndi.binary_erosion(np.isfinite(W), iterations=3)
    cs = measure.find_contours(np.nan_to_num(W, nan=0.0), k, mask=valid)
    out = np.full((n_theta, 2), np.nan)
    if not cs:
        return out
    cs = [c for c in cs if len(c) > 20]
    if not cs:
        return out
    c = max(cs, key=len)  # (n, 2) row, col
    th = np.arctan2(c[:, 0] - cen[1], c[:, 1] - cen[0])
    bins = ((th + np.pi) / (2 * np.pi) * n_theta).astype(int) % n_theta
    r = np.hypot(c[:, 0] - cen[1], c[:, 1] - cen[0])
    # per bin keep the point closest to the median radius of the contour (resolves folds crudely)
    rm = np.median(r)
    for b in np.unique(bins):
        sel = np.nonzero(bins == b)[0]
        j = sel[np.argmin(abs(r[sel] - rm))]
        out[b] = (c[j, 1], c[j, 0])
    return out


def build(scroll, windings, z0, z1, dz=20, n_theta=None, log=print):
    """Return {k: grid (3, n_z, n_theta)} in full-resolution voxel coords (x, y, z); -1 invalid."""
    zs = list(range(z0, z1, dz))
    grids = {}
    for iz, z in enumerate(zs):
        W, X0, Y0, s, cen = slice_W(scroll, z)
        for k in windings:
            if k not in grids:
                # angle bins ~ 20 full-res voxels along the arc at this winding's radius (first slice estimate)
                pts = winding_contour(W, k, cen, 720)
                r = np.nanmedian(np.hypot(pts[:, 0] - cen[0], pts[:, 1] - cen[1])) * s if np.isfinite(pts).any() else 500
                nt = n_theta or max(64, int(2 * np.pi * r / 20))
                grids[k] = np.full((3, len(zs), nt), -1, np.float32)
            nt = grids[k].shape[2]
            pts = winding_contour(W, k, cen, nt)
            ok = np.isfinite(pts[:, 0])
            grids[k][0, iz, ok] = (pts[ok, 0] + X0) * s
            grids[k][1, iz, ok] = (pts[ok, 1] + Y0) * s
            grids[k][2, iz, ok] = z
        log(f"slice z={z} done ({iz + 1}/{len(zs)}); W range {np.nanmin(W):.1f}..{np.nanmax(W):.1f}")
    return grids


def grow(seed, surf, center_xy, n_steps, dz=20.0, R=4, strip=4, log=print):
    """Grow a surface through z from seed rows (3, r, n) (the last row is the growth front; positive dz grows up).
    Each step copies the front row shifted by dz, appends it to the last `strip` rows and snaps the strip to the
    nearest sheet with pcu.refine (+-R voxels), keeping only the new row. Returns (3, r + n_steps, n)."""
    from . import refine
    rows = [seed[:, i] for i in range(seed.shape[1])]
    for s in range(n_steps):
        new = rows[-1].copy()
        ok = new[0] > 0
        new[2, ok] += dz
        g = np.stack(rows[-(strip - 1):] + [new], 1)
        g2, st = refine.refine(g, surf, factor=1, center_xy=center_xy, radii=(R, 2))
        nr = g2[:, -1]
        nr[:, ~ok] = -1
        rows.append(nr.astype(np.float32))
        if s % 10 == 0:
            log(f"grow step {s}: z~{np.median(nr[2][nr[0] > 0]):.0f} moved {st[0]['moved_med']:.2f} peak {st[0]['peak_med']:.0f}")
    return np.stack(rows, 1)
