"""Isometric flattening of a tifxyz-style grid (3, h, w; invalid cells have x <= 0).

The fitted spiral grids are uniform in winding angle, not in arc length, so rendered text is stretched
by up to 2x along the winding. We triangulate the valid quads, flatten with ARAP (as-rigid-as-possible,
local/global; Liu et al. 2008) starting from a per-row arc-length layout, then resample the surface on a
uniform UV grid, which gives a new grid whose steps are (near) equal 3D lengths in both directions."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy import ndimage as ndi


def triangulate(valid):
    h, w = valid.shape
    idx = -np.ones((h, w), int)
    idx[valid] = np.arange(valid.sum())
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    r, c = np.nonzero(q)
    a, b, cc, d = idx[r, c], idx[r, c + 1], idx[r + 1, c], idx[r + 1, c + 1]
    return idx, np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, cc], 1)])


def _local_frames(X, F):
    """Isometric 2D coordinates of each 3D triangle: (T, 3, 2)."""
    p0, p1, p2 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    e1 = p1 - p0; e2 = p2 - p0
    l1 = np.linalg.norm(e1, axis=1)
    ex = e1 / l1[:, None]
    n = np.cross(e1, e2); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ey = np.cross(n, ex)
    q = np.zeros((len(F), 3, 2))
    q[:, 1, 0] = l1
    q[:, 2, 0] = np.sum(e2 * ex, 1); q[:, 2, 1] = np.sum(e2 * ey, 1)
    return q


def _cot_weights(q):
    """Cotangent of the angle opposite each edge (i, j) for edges (0,1), (1,2), (2,0)."""
    cots = []
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        u = q[:, i] - q[:, k]; v = q[:, j] - q[:, k]
        cr = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        cots.append(np.sum(u * v, 1) / (cr + 1e-12))
    return np.stack(cots, 1)  # (T, 3)


def arap(X, F, U0, iters=30, zweight=0.0):
    """X: (n, 3) vertices, F: (T, 3) triangles, U0: (n, 2) initial UV. Returns UV (n, 2).
    zweight > 0 softly pulls v toward the vertex z (text lines run along constant z, so this keeps
    them straight over long strips instead of letting small wrinkles bend the whole row)."""
    n = len(X)
    q = _local_frames(X, F)
    ct = np.clip(_cot_weights(q), 1e-3, 1e3)
    E = ((0, 1), (1, 2), (2, 0))
    rows, cols, vals = [], [], []
    for e, (i, j) in enumerate(E):
        a, b, w = F[:, i], F[:, j], ct[:, e]
        rows += [a, b, a, b]; cols += [b, a, a, b]; vals += [-w, -w, w, w]
    L = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    L = L + sparse.eye(n) * 1e-8
    # pin vertex 0 softly to remove the translation null space
    L = L.tolil(); L[0, 0] += 1e3; L = L.tocsc()
    solve = factorized(L)
    deg = np.asarray(abs(L).sum(1)).ravel() / 2
    zc = X[:, 2] - X[0, 2] + U0[0, 1]
    solve_v = factorized((L + sparse.diags(zweight * deg)).tocsc()) if zweight > 0 else solve
    U = U0.copy()
    for _ in range(iters):
        # local: best rotation per triangle
        S = np.zeros((len(F), 2, 2))
        for e, (i, j) in enumerate(E):
            du = U[F[:, i]] - U[F[:, j]]; dq = q[:, i] - q[:, j]
            S += ct[:, e, None, None] * du[:, :, None] * dq[:, None, :]
        Uu, _, Vt = np.linalg.svd(S)
        R = Uu @ Vt
        bad = np.linalg.det(R) < 0
        if bad.any():
            Uu[bad, :, 1] *= -1; R[bad] = Uu[bad] @ Vt[bad]
        # global
        B = np.zeros((n, 2))
        for e, (i, j) in enumerate(E):
            rq = np.einsum("tab,tb->ta", R, q[:, i] - q[:, j]) * ct[:, e, None]
            np.add.at(B, F[:, i], rq); np.add.at(B, F[:, j], -rq)
        B[0] += 1e3 * U0[0]
        U = np.stack([solve(B[:, 0]), solve_v(B[:, 1] + zweight * deg * zc)], 1)
    return U


def initial_uv(G, valid):
    """Per-row cumulative arc length along columns (u) and mean row spacing along rows (v)."""
    h, w = valid.shape
    P = np.moveaxis(G, 0, -1)
    du = np.linalg.norm(np.diff(P, axis=1), axis=2)
    du[~(valid[:, 1:] & valid[:, :-1])] = np.nan
    med = np.nanmedian(du)
    du = np.where(np.isnan(du), med, du)
    u = np.concatenate([np.zeros((h, 1)), np.cumsum(du, 1)], 1)
    dv = np.linalg.norm(np.diff(P, axis=0), axis=2)
    dv = np.nanmedian(np.where(valid[1:] & valid[:-1], dv, np.nan))
    v = np.repeat(np.arange(h)[:, None] * dv, w, 1)
    return np.stack([u, v], -1)


def distortion(X, F, U):
    """Per-triangle singular values of the 3D->2D map: returns (s1, s2) arrays (isometry: both 1)."""
    q = _local_frames(X, F)
    D3 = np.stack([q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]], 2)       # (T, 2, 2)
    D2 = np.stack([U[F[:, 1]] - U[F[:, 0]], U[F[:, 2]] - U[F[:, 0]]], 2)
    J = D2 @ np.linalg.pinv(D3)
    s = np.linalg.svd(J, compute_uv=False)
    return s[:, 0], s[:, 1]


def rasterize(U, F, X, H, W, step, ustep):
    """Barycentric resampling of per-vertex values X (n, 3) on the uniform grid (H, W); tolerant of the
    few overlapping triangles a refined surface can produce (last writer wins)."""
    out = np.full((3, H, W), -1, np.float32)
    uv = U / np.array([ustep, step])
    a, b, c = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    lo = np.ceil(np.minimum(np.minimum(a, b), c)).astype(int)
    hi = np.floor(np.maximum(np.maximum(a, b), c)).astype(int)
    den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    ok = np.abs(den) > 1e-12
    span = np.clip(hi - lo, -1, 8)
    for dy in range(9):
        for dx in range(9):
            m = ok & (span[:, 0] >= dx) & (span[:, 1] >= dy)
            if not m.any():
                continue
            px = lo[m, 0] + dx; py = lo[m, 1] + dy
            am, bm, cm, dm = a[m], b[m], c[m], den[m]
            l1 = ((bm[:, 1] - cm[:, 1]) * (px - cm[:, 0]) + (cm[:, 0] - bm[:, 0]) * (py - cm[:, 1])) / dm
            l2 = ((cm[:, 1] - am[:, 1]) * (px - cm[:, 0]) + (am[:, 0] - cm[:, 0]) * (py - cm[:, 1])) / dm
            l3 = 1 - l1 - l2
            ins = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
            fm = F[m][ins]
            val = (l1[ins, None] * X[fm[:, 0]] + l2[ins, None] * X[fm[:, 1]] + l3[ins, None] * X[fm[:, 2]])
            out[:, py[ins], px[ins]] = val.T
    return out


def _mesh(G):
    valid = G[0] > 0
    idx, F = triangulate(valid)
    X = np.moveaxis(G, 0, -1)[valid].astype(float)
    ar = np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1)
    return valid, F[ar > 1e-3 * np.median(ar)], X  # drop degenerate triangles (collapsed cells)


def _uv(G, iters, zweight):
    valid, F, X = _mesh(G)
    U0 = initial_uv(G, valid)[valid]
    if zweight > 0:
        U0 = np.c_[U0[:, 0], X[:, 2] - X[:, 2].min()]
    U = arap(X, F, U0, iters, zweight)
    # rotate so z increases along v (least-squares direction of z in the UV plane)
    A = np.linalg.lstsq(np.c_[U, np.ones(len(U))], X[:, 2], rcond=None)[0][:2]
    ang = np.arctan2(A[0], A[1])
    c, s = np.cos(ang), np.sin(ang)
    return valid, np.c_[c * U[:, 0] - s * U[:, 1], s * U[:, 0] + c * U[:, 1]]


def flatten(G, step=20.0, iters=30, zweight=0.0, ustep=None, sub=1):
    """G: (3, h, w). Returns (new_grid (3, H, W), stats). New grid spacing = `step` 3D units per cell.
    sub > 1: solve ARAP on every sub-th vertex and interpolate the UVs to the full grid (dense refined
    grids are too large for a direct sparse factorization, and the refinement changes the metric little)."""
    if sub > 1:
        vc, Uc = _uv(G[:, ::sub, ::sub], iters, zweight)
        UV = np.zeros(vc.shape + (2,)); UV[vc] = Uc
        _, (iy, ix) = ndi.distance_transform_edt(~vc, return_indices=True)
        UV = UV[iy, ix]
        valid, F, X = _mesh(G)
        r, cc = np.nonzero(valid)
        U = np.stack([ndi.map_coordinates(UV[..., a], [r / sub, cc / sub], order=1, mode="nearest") for a in range(2)], 1)
    else:
        valid, U = _uv(G, iters, zweight)
        _, F, X = _mesh(G)
    s1, s2 = distortion(X, F, U)
    U -= U.min(0)
    us = ustep or step
    H = int(np.ceil(U[:, 1].max() / step)) + 1; W = int(np.ceil(U[:, 0].max() / us)) + 1
    out = rasterize(U, F, X, H, W, step, us)
    e1 = U[F[:, 1]] - U[F[:, 0]]; e2 = U[F[:, 2]] - U[F[:, 0]]
    sa = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    folds = int(min((sa <= 0).sum(), (sa >= 0).sum()))
    return out, dict(s1_p5=float(np.percentile(s1, 5)), s1_p95=float(np.percentile(s1, 95)),
                     s2_p5=float(np.percentile(s2, 5)), s2_p95=float(np.percentile(s2, 95)),
                     area_ratio=float(np.median(s1 * s2)), folds=folds, ntri=len(F),
                     z_resid=float(np.std(U[:, 1] - (X[:, 2] - X[:, 2].min()) - np.median(U[:, 1] - (X[:, 2] - X[:, 2].min())))))
