"""Crest-graph synchronization: nodes = unbranched sheet-crest pieces, edges = certified +1 steps.

Wrap labels are integers per crest piece, so a sheet cannot change wrap along its own length.
Solved exactly as an L1 integer synchronization (network-matrix LP, integral optimum).
Branch cut: the half-line theta = pi from the umbilicus; crest pieces are split there and a step
that crosses it gets its expected difference corrected by one.
"""
import numpy as np
import scipy.sparse as sp
from scipy import ndimage as ndi
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components
from skimage.morphology import skeletonize

from . import constraints


def crest_pieces(phi, mask, theta, amp, min_len=12, residues=None, res_r=2):
    crest = (np.cos(phi) > 0.6) & mask & (amp > np.percentile(amp[mask], 25))
    if residues is not None:
        crest &= ~ndi.binary_dilation(residues, iterations=res_r)
    sk = skeletonize(crest)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junction = sk & (nb >= 3)
    sk2 = sk & ~ndi.binary_dilation(junction, iterations=1)
    # split at the branch cut (theta near +-pi, left of the umbilicus)
    sk2 &= ~(np.abs(np.abs(theta) - np.pi) < 0.01)
    lab, n = ndi.label(sk2, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_len
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    return lab


def recto_pieces(rec, mask, theta, thr=0.35, min_len=12):
    """Unbranched pieces of the recto-surface skeleton (one surface per sheet)."""
    sk = skeletonize((rec > thr) & mask)
    nb = ndi.convolve(sk.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junction = sk & (nb >= 3)
    sk2 = sk & ~ndi.binary_dilation(junction, iterations=1)
    sk2 &= ~(np.abs(np.abs(theta) - np.pi) < 0.01)
    lab, n = ndi.label(sk2, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel()); keep = sizes >= min_len; keep[0] = False
    return np.where(keep[lab], lab, 0)


def build_edges(lab, phi, mx, my, R, g3, Wf, mask, theta, step=6, snap=1.5, kmax=1, rmap=None, rf=2, gold_w=None):
    """Walk certified +1 steps from sampled crest points; edge between the pieces they land on."""
    H, W = lab.shape
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    ys, xs = np.nonzero(lab)
    sel = np.arange(0, len(ys), step)
    E = {}
    def piece_at(q):
        qi, qj = int(round(q[0])), int(round(q[1]))
        if not (0 <= qi < H and 0 <= qj < W) or dist[qi, qj] > snap:
            return 0, qi, qj
        return lab[iy[qi, qj], ix[qi, qj]], qi, qj

    for k in sel:
        p = np.array([ys[k], xs[k]], float)
        A = lab[ys[k], xs[k]]
        ta = theta[ys[k], xs[k]]
        cur, pts = p, [p]
        for kk in range(1, kmax + 1):
            q = constraints._next_crest(phi, mx, my, cur, 1.0)
            if q is None or not mask[int(q[0]), int(q[1])]:
                break
            pts.append(q); cur = q
            ok = constraints.certify(p, q, phi, g3, Wf) if kk == 1 else constraints.certify_k(p, q, kk, phi, g3, Wf)
            if not ok:
                if kk == 1:
                    break
                continue
            Bn, qi, qj = piece_at(q)
            if Bn == 0 or Bn == A:
                continue
            wgt = 1.0
            if rmap is not None and kk == 1:
                g_ok = constraints.recto_count(rmap, pts[-2], q, rf) == 1
                if gold_w is None and not g_ok:
                    continue
                wgt = gold_w if (gold_w is not None and g_ok) else 1.0
            tb = theta[qi, qj]
            d = kk
            if abs(tb - ta) > np.pi:  # crossed theta = +-pi
                d += 1 if tb < ta else -1
            E[(A, Bn, d)] = E.get((A, Bn, d), 0) + wgt
    return E


def solve(E, n_nodes, anchor_sizes):
    keys = list(E.keys())
    m = len(keys)
    a = np.array([k[0] for k in keys]); b = np.array([k[1] for k in keys]); d = np.array([k[2] for k in keys], float)
    w = np.array([E[k] for k in keys], float)
    nodes = np.unique(np.r_[a, b]); idx = -np.ones(n_nodes + 1, int); idx[nodes] = np.arange(len(nodes))
    ia, ib = idx[a], idx[b]; n = len(nodes)
    rows = np.arange(m)
    Aub = sp.vstack([
        sp.coo_matrix((np.r_[-np.ones(m), np.ones(m), -np.ones(m)], (np.r_[rows, rows, rows], np.r_[n + rows, ib, ia])), shape=(m, n + m)),
        sp.coo_matrix((np.r_[-np.ones(m), -np.ones(m), np.ones(m)], (np.r_[rows, rows, rows], np.r_[n + rows, ib, ia])), shape=(m, n + m)),
    ]).tocsr()
    bub = np.r_[d, -d]
    c = np.r_[np.zeros(n), w]
    Gr = sp.coo_matrix((np.ones(m), (ia, ib)), shape=(n, n))
    nc, rl = connected_components(Gr, directed=False)
    bounds = [(None, None)] * n + [(0, None)] * m
    for k in range(nc):
        mem = np.nonzero(rl == k)[0]
        bounds[mem[np.argmax(anchor_sizes[nodes[mem]])]] = (0, 0)
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method="highs")
    x = np.round(res.x[:n]).astype(int)
    viol = np.abs(x[ib] - x[ia] - d) > 0
    bad_w = np.bincount(ia, weights=w * viol, minlength=n) + np.bincount(ib, weights=w * viol, minlength=n)
    all_w = np.bincount(ia, weights=w, minlength=n) + np.bincount(ib, weights=w, minlength=n)
    node_bad_frac = np.full(n_nodes + 1, np.nan); node_bad_frac[nodes] = bad_w / np.maximum(all_w, 1e-9)
    label = np.full(n_nodes + 1, np.nan); label[nodes] = x
    comp = np.full(n_nodes + 1, -1); comp[nodes] = rl
    resid = np.abs(x[ib] - x[ia] - d)
    return label, comp, dict(nodes=n, edges=m, components=nc, violated=int(np.sum(w[resid > 0])), total=int(w.sum()), node_bad_frac=node_bad_frac)


def solve_robust(E, n_nodes, anchor_sizes, drop_frac=0.3, rounds=2):
    """Solve, drop crest pieces whose incident edge weight is mostly violated (likely merged two
    sheets), and re-solve."""
    E = dict(E)
    dropped = set()
    for _ in range(rounds):
        label, comp, info = solve(E, n_nodes, anchor_sizes)
        bad = {int(i) for i in np.nonzero(np.nan_to_num(info["node_bad_frac"]) > drop_frac)[0]}
        if not bad:
            break
        dropped |= bad
        E = {k: v for k, v in E.items() if k[0] not in dropped and k[1] not in dropped}
    label, comp, info = solve(E, n_nodes, anchor_sizes)
    info["dropped"] = len(dropped)
    return label, comp, info
