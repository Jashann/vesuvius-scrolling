"""Exact weighted-L1 phase unwrapping by graph cuts (Bioucas-Dias & Valadao 2007, PUMA),
2D grid version, warm-started. Units: turns (phase / 2pi)."""
import numpy as np
import maxflow

from .phase import wrap


def _edge_residuals(U, psi):
    """Integer mismatch r = (U_j - U_i) - wrapped(psi_j - psi_i)/2pi for right and down edges."""
    gh = wrap(psi[:, 1:] - psi[:, :-1]) / (2 * np.pi)
    gv = wrap(psi[1:, :] - psi[:-1, :]) / (2 * np.pi)
    rh = np.round((U[:, 1:] - U[:, :-1]) - gh)
    rv = np.round((U[1:, :] - U[:-1, :]) - gv)
    return rh, rv


def energy(U, psi, wh, wv):
    rh, rv = _edge_residuals(U, psi)
    return float(np.sum(wh * np.abs(rh)) + np.sum(wv * np.abs(rv)))


def _move(U, psi, wh, wv, step):
    """One binary move U <- U + step*delta, delta in {0,1}; returns new U and energy drop."""
    H, W = U.shape
    rh, rv = _edge_residuals(U, psi)
    g = maxflow.Graph[float]()
    ids = g.add_grid_nodes((H, W))
    unary1 = np.zeros((H, W))  # cost coefficient on delta_i = 1
    unary0 = np.zeros((H, W))  # cost when delta_i = 0 (for negative coefficients)
    for r, w, axis in ((rh, wh, 1), (rv, wv, 0)):
        # pair (i, j): E(di, dj) = w |r + step (dj - di)|
        E00 = w * np.abs(r); E11 = E00
        E01 = w * np.abs(r + step); E10 = w * np.abs(r - step)
        a_i = E10 - E00          # coefficient on di
        a_j = E11 - E10          # coefficient on dj
        c = E01 + E10 - E00 - E11  # on (1 - di) dj, >= 0
        if axis == 1:
            sl_i, sl_j = (slice(None), slice(None, -1)), (slice(None), slice(1, None))
            structure = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
            cw = np.zeros((H, W)); cw[:, :-1] = c
        else:
            sl_i, sl_j = (slice(None, -1), slice(None)), (slice(1, None), slice(None))
            structure = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
            cw = np.zeros((H, W)); cw[:-1, :] = c
        for sl, a in ((sl_i, a_i), (sl_j, a_j)):
            unary1[sl] += np.maximum(a, 0)
            unary0[sl] += np.maximum(-a, 0)
        g.add_grid_edges(ids, weights=cw, structure=structure, symmetric=False)
    g.add_grid_tedges(ids, unary1, unary0)
    g.maxflow()
    delta = g.get_grid_segments(ids).astype(float)  # True = sink = delta 1
    Un = U + step * delta
    return Un


def puma(psi, U0, wh, wv, max_iter=50, verbose=False):
    U = U0.copy()
    E = energy(U, psi, wh, wv)
    for it in range(max_iter):
        improved = False
        for step in (+1.0, -1.0):
            Un = _move(U, psi, wh, wv, step)
            En = energy(Un, psi, wh, wv)
            if En < E - 1e-9:
                U, E, improved = Un, En, True
        if verbose:
            print(f"  puma iter {it}: E={E:.1f}")
        if not improved:
            break
    return U, E
