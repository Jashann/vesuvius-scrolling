"""2.5D winding field: per-slice branch-cut unwrapping, then exact L1 reconciliation of all
cut-bounded regions across a stack of slices using in-plane and between-slice (z) edges."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linprog

from .phase import wrap


def reconcile(Us, psis, labs, masks, w_inplane, w_z=1.0):
    """Us, psis, labs, masks: lists (one per slice) of 2D arrays (U in turns, psi radians,
    region labels from the cut-avoiding integration, masks). w_inplane: list of 2D quality maps.
    Returns corrected list of U."""
    Z = len(Us)
    H, W = Us[0].shape
    # global region ids
    offs = [0]
    glab = []
    for z in range(Z):
        l = labs[z].copy()
        l[~masks[z] | ~np.isfinite(Us[z])] = -1
        u, inv = np.unique(l[l >= 0], return_inverse=True)
        g = -np.ones_like(l)
        g[l >= 0] = inv + offs[-1]
        glab.append(g)
        offs.append(offs[-1] + len(u))
    n = offs[-1]
    A_, B_, R_, W_ = [], [], [], []

    def add_edges(la, lb, Ua, Ub, pa, pb, w):
        ok = (la >= 0) & (lb >= 0) & (la != lb)
        if not ok.any():
            return
        g = wrap(pb[ok] - pa[ok]) / (2 * np.pi)
        r = np.round((Ub[ok] - Ua[ok]) - g)
        A_.append(la[ok]); B_.append(lb[ok]); R_.append(r); W_.append(w[ok] if np.ndim(w) else np.full(ok.sum(), w))

    for z in range(Z):
        g, U, p, q = glab[z], Us[z], psis[z], w_inplane[z]
        add_edges(g[:, :-1], g[:, 1:], U[:, :-1], U[:, 1:], p[:, :-1], p[:, 1:], np.minimum(q[:, :-1], q[:, 1:]))
        add_edges(g[:-1, :], g[1:, :], U[:-1, :], U[1:, :], p[:-1, :], p[1:, :], np.minimum(q[:-1, :], q[1:, :]))
        if z + 1 < Z:
            g2 = glab[z + 1]
            add_edges(g, g2, U, Us[z + 1], p, psis[z + 1], w_z * np.minimum(q, w_inplane[z + 1]))
    A = np.concatenate(A_); B = np.concatenate(B_); R = np.concatenate(R_); Wt = np.concatenate(W_)
    sw = A > B
    a = np.where(sw, B, A); b = np.where(sw, A, B); r = np.where(sw, -R, R)
    key = np.stack([a, b, r.astype(np.int64)], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    Wagg = np.bincount(inv.ravel(), weights=Wt)
    m = len(uk)
    ia, ib, rr = uk[:, 0], uk[:, 1], uk[:, 2].astype(float)
    rows = np.arange(m)
    Aub = sp.vstack([
        sp.coo_matrix((np.r_[-np.ones(m), np.ones(m), -np.ones(m)], (np.r_[rows, rows, rows], np.r_[n + rows, ib, ia])), shape=(m, n + m)),
        sp.coo_matrix((np.r_[-np.ones(m), -np.ones(m), np.ones(m)], (np.r_[rows, rows, rows], np.r_[n + rows, ib, ia])), shape=(m, n + m)),
    ]).tocsr()
    bub = np.r_[-rr, rr]
    c = np.r_[np.zeros(n), Wagg]
    Gr = sp.coo_matrix((np.ones(m), (ia, ib)), shape=(n, n))
    nc, rl = connected_components(Gr, directed=False)
    sizes = np.zeros(n)
    for z in range(Z):
        g = glab[z]
        sizes += np.bincount(g[g >= 0], minlength=n)
    bounds = [(None, None)] * n + [(0, None)] * m
    for k in range(nc):
        mem = np.nonzero(rl == k)[0]
        bounds[mem[np.argmax(sizes[mem])]] = (0, 0)
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method="highs")
    o = np.round(res.x[:n]).astype(np.int64)
    out = []
    for z in range(Z):
        g = glab[z]
        U = Us[z].copy()
        U[g >= 0] += o[g[g >= 0]]
        # cut pixels: re-derive from the nearest reconciled pixel and the local wrapped phase
        hole = (g < 0) & masks[z]
        if hole.any():
            from scipy import ndimage as ndi
            _, (iy, ix) = ndi.distance_transform_edt(g < 0, return_indices=True)
            base = U[iy, ix]
            U = np.where(hole, base + wrap(psis[z] - 2 * np.pi * base) / (2 * np.pi), U)
        out.append(U)
    E0 = float(np.sum(Wt * np.abs(R)))
    return out, dict(regions=n, pairs=m, E0=E0, E1=float(res.fun), status=res.status, moved=int(np.count_nonzero(o)))
