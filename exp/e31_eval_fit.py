"""E31: score spiral-fit outputs (per-winding tifxyz grids exported from Kaggle) on human ladders.

Each ladder point gets the winding whose fitted surface passes closest (point-to-surface distance on
a locally 6x-upsampled grid), then slips per wrap are counted along each ladder. Also reports the
absolute ladder's offset consistency.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from pcu import data

Z0, Z1 = int(os.environ.get("Z0", 8400)), int(os.environ.get("Z1", 9400))
run = sys.argv[1]
lad = []
for fname in ("abs_winding.json", "relative_windings.json"):
    for col in data.fetch_json(f"{data.P4_SPIRAL}/{fname}")["collections"].values():
        P = sorted([p for p in col["points"].values() if p.get("wind_a") is not None], key=lambda p: p["wind_a"])
        if len(P) < 2: continue
        zz = np.array([p["p"][2] for p in P])
        if np.ptp(zz) > 2 or not (Z0 + 20 <= zz.mean() < Z1 - 20): continue
        lad.append((fname[:3], np.array([p["p"] for p in P]), np.array([p["wind_a"] for p in P], float)))
print("ladders", len(lad), "pairs", sum(len(w) - 1 for _, _, w in lad))
for f in sorted(glob.glob(f"runs/kaggle/{run}/grids_*.npz")):
    name = f.split("grids_")[1][:-4]
    G = np.load(f)
    V, Wv, keys = [], [], []
    for k in G.files:
        g = G[k]; ok = (g[0] > 0) & (g[2] > Z0 - 60) & (g[2] < Z1 + 60)
        ii, jj = np.nonzero(ok)
        V.append(np.stack([g[0][ok], g[1][ok], g[2][ok]], 1)); Wv.append(np.stack([np.full(len(ii), int(k[1:])), ii, jj], 1))
    V = np.concatenate(V); Wv = np.concatenate(Wv); tree = cKDTree(V)

    def winding_at(p):
        idx = tree.query_ball_point(p, 40)
        best = (np.inf, None)
        for w in np.unique(Wv[idx, 0]):
            sel = [i for i in idx if Wv[i, 0] == w]
            i0 = sel[int(np.argmin(np.linalg.norm(V[sel] - p, axis=1)))]
            g = G[f"w{w:03d}"]; r, c = Wv[i0, 1], Wv[i0, 2]
            r0, r1, c0, c1 = max(r - 2, 0), min(r + 3, g.shape[1]), max(c - 2, 0), min(c + 3, g.shape[2])
            sub = g[:, r0:r1, c0:c1]
            if (sub[0] <= 0).any():
                d = np.min(np.linalg.norm(V[sel] - p, axis=1))
            else:
                up = np.stack([ndi.zoom(sub[a], 6, order=1) for a in range(3)])
                d = np.min(np.linalg.norm(up.reshape(3, -1).T - p, axis=1))
            if d < best[0]:
                best = (d, w)
        return best

    slips = tot = far = 0; offs = []
    for kind, pts, w in lad:
        res = [winding_at(p) for p in pts]
        v = np.array([b[1] if b[1] is not None and b[0] < 10 else np.nan for b in res], float)
        far += int(np.sum(~np.isfinite(v)))
        dv = np.diff(v); dw = np.diff(w); ok = np.isfinite(dv)
        if not ok.any(): continue
        sg = np.sign(np.nanmedian(dv)) or 1
        slips += int(np.sum(sg * dv[ok] != dw[ok])); tot += int(ok.sum())
        if kind == "abs":
            f_ = np.isfinite(v); offs += list(v[f_] - w[f_])
    msg = f"{name}: slips {slips}/{tot} = {slips/max(tot,1):.4f} per wrap; unassigned points {far}"
    if offs:
        vals, cnt = np.unique(offs, return_counts=True); msg += f"; abs ladder on modal offset {cnt.max()}/{len(offs)}"
    print(msg, flush=True)
