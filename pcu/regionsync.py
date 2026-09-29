"""Region-level exact L1 integer synchronization (SNAPHU-style reconcile, Costantini's LP).

Start from any unwrapping U0 (turns). Pixels joined by edges with zero integer mismatch form
regions; the only freedom left is one integer offset per region. Minimize
    sum_e w_e |r_e + o_B - o_A|
over integer offsets; the LP has a network matrix, so its optimum is integral.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linprog

from .phase import wrap


def residual_edges(U, psi, mask):
    H, W = U.shape
    idx = np.arange(H * W).reshape(H, W)
    out = []
    for axis in (1, 0):
        if axis == 1:
            a, b = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
        else:
            a, b = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
        g = wrap(psi[b] - psi[a]) / (2 * np.pi)
        r = np.round((U[b] - U[a]) - g)
        ok = mask[a] & mask[b]
        out.append((idx[a][ok], idx[b][ok], r[ok], ok))
    return out


def sync(U0, psi, mask, wh, wv, min_region=1):
    H, W = U0.shape
    E = residual_edges(U0, psi, mask)
    ws = [wh[E[0][3]], wv[E[1][3]]]
    I = np.concatenate([e[0] for e in E]); J = np.concatenate([e[1] for e in E])
    R = np.concatenate([e[2] for e in E]); Wt = np.concatenate(ws)
    same = R == 0
    G = sp.coo_matrix((np.ones(same.sum()), (I[same], J[same])), shape=(H * W, H * W))
    ncomp, lab = connected_components(G, directed=False)
    inter = ~same
    A, B, r, w = lab[I[inter]], lab[J[inter]], R[inter], Wt[inter]
    # canonical orientation A<B (flip sign of r when swapping)
    sw = A > B
    A2 = np.where(sw, B, A); B2 = np.where(sw, A, B); r2 = np.where(sw, -r, r)
    key = np.stack([A2, B2, r2.astype(np.int64)], 1)
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    Wagg = np.bincount(inv.ravel(), weights=w)
    keep = Wagg > 0
    uk, Wagg = uk[keep], Wagg[keep]
    regs = np.unique(np.concatenate([uk[:, 0], uk[:, 1]]))
    rid = -np.ones(ncomp, np.int64); rid[regs] = np.arange(len(regs))
    n, m = len(regs), len(uk)
    a, b, rr = rid[uk[:, 0]], rid[uk[:, 1]], uk[:, 2].astype(float)
    # variables x = [o (n), t (m)]; minimize sum W t
    # t - (o_b - o_a) >= r   ->  -t + o_b - o_a <= -r
    # t + (o_b - o_a) >= -r  ->  -t - o_b + o_a <= r
    rows = np.arange(m)
    Aub = sp.vstack([
        sp.coo_matrix((np.concatenate([-np.ones(m), np.ones(m), -np.ones(m)]),
                       (np.concatenate([rows, rows, rows]), np.concatenate([n + rows, b, a]))), shape=(m, n + m)),
        sp.coo_matrix((np.concatenate([-np.ones(m), -np.ones(m), np.ones(m)]),
                       (np.concatenate([rows, rows, rows]), np.concatenate([n + rows, b, a]))), shape=(m, n + m)),
    ]).tocsr()
    bub = np.concatenate([-rr, rr])
    c = np.concatenate([np.zeros(n), Wagg])
    # anchor the largest region of each connected component of the region graph
    Gr = sp.coo_matrix((np.ones(m), (a, b)), shape=(n, n))
    nc, rl = connected_components(Gr, directed=False)
    sizes = np.bincount(lab, minlength=ncomp)[regs]
    bounds = [(None, None)] * n + [(0, None)] * m
    for k in range(nc):
        members = np.nonzero(rl == k)[0]
        anchor = members[np.argmax(sizes[members])]
        bounds[anchor] = (0, 0)
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method="highs")
    o = np.round(res.x[:n]).astype(np.int64)
    off = np.zeros(ncomp, np.int64); off[regs] = o
    U1 = U0 + off[lab].reshape(H, W)
    E0 = float(np.sum(Wt * np.abs(R)))
    return U1, dict(regions=ncomp, graph_regions=n, pairs=m, E0=E0, E1=float(res.fun), status=res.status)
