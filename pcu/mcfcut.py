"""Branch-cut unwrapping with optimally matched residues (min-cost flow), cut-avoiding
integration, then exact region-level L1 reconciliation (regionsync)."""
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, breadth_first_order
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage.draw import line as draw_line

from .phase import wrap, residues


def line_cost(cost, p, q):
    rr, cc = draw_line(int(p[0]), int(p[1]), int(q[0]), int(q[1]))
    return float(cost[rr, cc].sum()), rr, cc


def match_residues(q, mask, cost, k=8, rmax=60, scale=100):
    """Min-cost flow pairing of +/- residues (and to the boundary 'earth')."""
    ys, xs = np.nonzero(q)
    ch = q[ys, xs].astype(int)
    pts = np.stack([ys, xs], 1).astype(float)
    n = len(pts)
    # distance to boundary (outside mask) for earth edges
    dist_out = ndi.distance_transform_edt(mask)
    _, (by, bx) = ndi.distance_transform_edt(mask, return_indices=True)
    G = nx.DiGraph()
    G.add_node("earth", demand=int(ch.sum()))  # earth absorbs the net charge
    for i in range(n):
        G.add_node(i, demand=-int(ch[i]))  # + residue supplies 1 unit (demand -1)
    tree = cKDTree(pts)
    segs = {}
    for i in range(n):
        dd, jj = tree.query(pts[i], k=min(k + 1, n), distance_upper_bound=rmax)
        for d, j in zip(np.atleast_1d(dd), np.atleast_1d(jj)):
            if j == i or j >= n or not np.isfinite(d):
                continue
            if ch[i] == ch[j]:
                continue
            c, rr, cc = line_cost(cost, pts[i], pts[j])
            w = int(c * scale) + 1
            G.add_edge(i, j, weight=w); G.add_edge(j, i, weight=w)
            segs[(min(i, j), max(i, j))] = (rr, cc)
        e = (int(by[ys[i], xs[i]]), int(bx[ys[i], xs[i]]))
        c, rr, cc = line_cost(cost, pts[i], e)
        w = int(c * scale) + 1
        G.add_edge(i, "earth", weight=w); G.add_edge("earth", i, weight=w)
        segs[(i, "earth")] = (rr, cc)
    flow = nx.min_cost_flow(G)
    cut = np.zeros(q.shape, bool)
    for u, d in flow.items():
        for v, f in d.items():
            if f <= 0:
                continue
            key = (u, "earth") if v == "earth" else ((v, "earth") if u == "earth" else (min(u, v), max(u, v)))
            rr, cc = segs[key]
            cut[rr, cc] = True
    return cut


def match_residues_geo(q, mask, cost, R=80, kmax=12, scale=100, earth_penalty=1.0):
    """Like match_residues but with geodesic (least-cost path) distances and paths.

    For each residue, a least-cost-path search in a (2R+1)^2 window gives costs to other residues
    and to the scroll boundary; min-cost flow picks the pairing; cuts follow the least-cost paths.
    """
    from skimage.graph import MCP_Geometric
    H, W = q.shape
    ys, xs = np.nonzero(q)
    ch = q[ys, xs].astype(int)
    n = len(ys)
    pts = np.stack([ys, xs], 1)
    tree = cKDTree(pts.astype(float))
    G = nx.DiGraph()
    G.add_node("earth", demand=int(ch.sum()))
    for i in range(n):
        G.add_node(i, demand=-int(ch[i]))
    paths = {}
    outside = ~mask
    _, (BY, BX) = ndi.distance_transform_edt(mask, return_indices=True)
    for i in range(n):
        y0, y1 = max(ys[i] - R, 0), min(ys[i] + R + 1, H)
        x0, x1 = max(xs[i] - R, 0), min(xs[i] + R + 1, W)
        C = cost[y0:y1, x0:x1].astype(float)
        m = MCP_Geometric(C)
        cum, _ = m.find_costs([(ys[i] - y0, xs[i] - x0)])
        nb = tree.query_ball_point(pts[i].astype(float), R)
        cand = [j for j in nb if j != i and ch[j] != ch[i]]
        cand.sort(key=lambda j: cum[ys[j] - y0, xs[j] - x0])
        for j in cand[:kmax]:
            cij = cum[ys[j] - y0, xs[j] - x0]
            if not np.isfinite(cij):
                continue
            w = int(cij * scale) + 1
            key = (min(i, j), max(i, j))
            if key not in paths:
                p = m.traceback((ys[j] - y0, xs[j] - x0))
                paths[key] = (np.array([a for a, _ in p]) + y0, np.array([b for _, b in p]) + x0)
            if not G.has_edge(i, j) or G[i][j]["weight"] > w:
                G.add_edge(i, j, weight=w); G.add_edge(j, i, weight=w)
        ob = outside[y0:y1, x0:x1]
        if ob.any():
            cc = np.where(ob, cum, np.inf)
            k = np.unravel_index(np.argmin(cc), cc.shape)
            p = m.traceback(k)
            paths[(i, "earth")] = (np.array([a for a, _ in p]) + y0, np.array([b for _, b in p]) + x0)
            w = int(cc[k] * scale * earth_penalty) + 1
        else:
            e = (int(BY[ys[i], xs[i]]), int(BX[ys[i], xs[i]]))
            c, rr, cc2 = line_cost(cost, pts[i], e)
            paths[(i, "earth")] = (rr, cc2)
            w = int(c * scale * earth_penalty * 4) + 1
        G.add_edge(i, "earth", weight=w); G.add_edge("earth", i, weight=w)
    flow = nx.min_cost_flow(G)
    cut = np.zeros(q.shape, bool)
    for u, d in flow.items():
        for v, f in d.items():
            if f <= 0:
                continue
            key = (u, "earth") if v == "earth" else ((v, "earth") if u == "earth" else (min(u, v), max(u, v)))
            rr, cc = paths[key]
            cut[rr, cc] = True
    return cut


def integrate(psi, allowed):
    """Integrate wrapped gradients over pixels in `allowed` (4-connected), per component.
    One BFS from a virtual root linked to one seed per component; values by pointer jumping."""
    H, W = psi.shape
    N = H * W
    idx = np.arange(N).reshape(H, W)
    I, J, D = [], [], []
    for a, b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                 ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        ok = allowed[a] & allowed[b]
        I.append(idx[a][ok]); J.append(idx[b][ok]); D.append(wrap(psi[b] - psi[a])[ok] / (2 * np.pi))
    I = np.concatenate(I); J = np.concatenate(J); D = np.concatenate(D)
    G = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(N, N))
    ncomp, lab = connected_components(G, directed=False)
    flat = np.nonzero(allowed.ravel())[0]
    _, first = np.unique(lab[flat], return_index=True)
    seeds = flat[first]
    root = N
    Iv = np.concatenate([I, J, np.full(len(seeds), root)]); Jv = np.concatenate([J, I, seeds])
    Dv = np.concatenate([D, -D, psi.ravel()[seeds] / (2 * np.pi)])
    A = sp.coo_matrix((np.ones(len(Iv)), (Iv, Jv)), shape=(N + 1, N + 1)).tocsr()
    order, pred = breadth_first_order(A, root, directed=True, return_predecessors=True)
    Dm = {}
    Dmat = sp.coo_matrix((Dv, (Iv, Jv)), shape=(N + 1, N + 1)).tocsr()
    nodes = order[1:]
    par = np.full(N + 1, root, np.int64); par[nodes] = pred[nodes]
    acc = np.zeros(N + 1)
    acc[nodes] = np.asarray(Dmat[par[nodes], nodes]).ravel()
    active = nodes
    for _ in range(64):
        p = par[active]
        notroot = p != root
        if not notroot.any():
            break
        a2 = active[notroot]; p2 = p[notroot]
        acc[a2] = acc[a2] + acc[p2]
        par[a2] = par[p2]
        active = a2
    U = np.full(N, np.nan)
    U[nodes] = acc[nodes]
    return U.reshape(H, W), lab.reshape(H, W)


def unwrap_mcf(psi, mask, quality, k=8, rmax=60, geo=False):
    q = residues(psi)
    qq = np.zeros(psi.shape, np.int8); qq[:-1, :-1] = q
    qq[~mask] = 0
    cost = np.where(mask, quality, 0.0) + 0.01
    cut = match_residues_geo(qq, mask, cost) if geo else match_residues(qq, mask, cost, k=k, rmax=rmax)
    cut = ndi.binary_dilation(cut)  # make cuts 4-connected barriers
    allowed = mask & ~cut
    U, lab = integrate(psi, allowed)
    # fill cut pixels from nearest integrated neighbour, consistent with local wrapped phase
    fillme = mask & ~np.isfinite(U)
    if fillme.any():
        _, (iy, ix) = ndi.distance_transform_edt(~np.isfinite(U) | ~mask, return_indices=True)
        base = U[iy, ix]
        U = np.where(fillme, base + wrap(psi - 2 * np.pi * base) / (2 * np.pi), U)
    lab = lab.copy(); lab[cut | ~mask] = -1
    return U, cut, int(np.count_nonzero(qq)), lab
