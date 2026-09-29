"""E31b: re-score every spiral fit on the held-out ladders with ladder-level uncertainty.

Scores each fit grid (runs/kaggle/<run>/grids_<name>.npz) exactly as exp/e31_eval_fit.py (human ladders) or
exp/e43_eval_gold.py (PCU gold ladders) does, but keeps the per-ladder slip counts, then reports
  - slips per wrap with a 95% ladder-level bootstrap CI (ladders resampled, pairs within a ladder kept together),
  - a paired bootstrap CI of the difference against a reference fit,
  - a selection-free estimate for a weight sweep: choose the weight on a random half of the ladders, report
    the chosen fit on the other half (200 splits).
Usage:
  python exp/e31b_bootstrap.py human A2=pcu-fit-auto B2=pcu-fit-b B=pcu-fit-b B3w10=pcu-fit-b3 ... --ref A2 --sweep B4w4,B3w10,B4w20,B3w40 --out runs/e31b_p4.json
  LADDERS=runs/gold0826_heldout8.json Z0=8500 Z1=9500 python exp/e31b_bootstrap.py gold A0=pcu-0826 ... --ref A0 --out runs/e31b_0826.json"""
import sys, os, json, glob, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from pcu import data

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["human", "gold"]); ap.add_argument("fits", nargs="+")
ap.add_argument("--ref", default=None); ap.add_argument("--sweep", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--nmax", type=int, default=3000); ap.add_argument("--nboot", type=int, default=2000)
a = ap.parse_args()
Z0, Z1 = int(os.environ.get("Z0", 8400)), int(os.environ.get("Z1", 9400))

lad = []
if a.kind == "human":
    for fname in ("abs_winding.json", "relative_windings.json"):
        for col in data.fetch_json(f"{data.P4_SPIRAL}/{fname}")["collections"].values():
            P = sorted([p for p in col["points"].values() if p.get("wind_a") is not None], key=lambda p: p["wind_a"])
            if len(P) < 2: continue
            zz = np.array([p["p"][2] for p in P])
            if np.ptp(zz) > 2 or not (Z0 + 20 <= zz.mean() < Z1 - 20): continue
            lad.append((np.array([p["p"] for p in P]), np.array([p["wind_a"] for p in P], float)))
else:
    for col in json.load(open(os.environ["LADDERS"]))["collections"].values():
        P = sorted([p for p in col["points"].values() if p.get("wind_a") is not None], key=lambda p: p["wind_a"])
        if len(P) < 2: continue
        zz = np.array([p["p"][2] for p in P])
        if np.ptp(zz) > 2 or not (Z0 + 20 <= zz.mean() < Z1 - 20): continue
        lad.append((np.array([p["p"] for p in P]), np.array([p["wind_a"] for p in P], float)))
    rng0 = np.random.default_rng(0)
    if len(lad) > a.nmax:
        lad = [lad[i] for i in sorted(rng0.choice(len(lad), a.nmax, replace=False))]
print("ladders", len(lad), "pairs", sum(len(w) - 1 for _, w in lad), flush=True)


def score(f):
    G = np.load(f); V, Wv = [], []
    for k in G.files:
        g = G[k]; ok = (g[0] > 0) & (g[2] > Z0 - 60) & (g[2] < Z1 + 60)
        ii, jj = np.nonzero(ok)
        V.append(np.stack([g[0][ok], g[1][ok], g[2][ok]], 1)); Wv.append(np.stack([np.full(len(ii), int(k[1:])), ii, jj], 1))
    V = np.concatenate(V); Wv = np.concatenate(Wv); tree = cKDTree(V)

    def winding_at(p):
        idx = tree.query_ball_point(p, 40); best = (np.inf, None)
        for w in np.unique(Wv[idx, 0]):
            sel = [i for i in idx if Wv[i, 0] == w]
            i0 = sel[int(np.argmin(np.linalg.norm(V[sel] - p, axis=1)))]
            g = G[f"w{w:03d}"]; r, c = Wv[i0, 1], Wv[i0, 2]
            r0, r1, c0, c1 = max(r - 2, 0), min(r + 3, g.shape[1]), max(c - 2, 0), min(c + 3, g.shape[2])
            sub = g[:, r0:r1, c0:c1]
            if (sub[0] <= 0).any():
                d = np.min(np.linalg.norm(V[sel] - p, axis=1))
            else:
                up = np.stack([ndi.zoom(sub[q], 6, order=1) for q in range(3)])
                d = np.min(np.linalg.norm(up.reshape(3, -1).T - p, axis=1))
            if d < best[0]: best = (d, w)
        return best

    S, T, far = [], [], 0
    for pts, w in lad:
        res = [winding_at(p) for p in pts]
        v = np.array([b[1] if b[1] is not None and b[0] < 10 else np.nan for b in res], float)
        far += int(np.sum(~np.isfinite(v)))
        dv = np.diff(v); dw = np.diff(w); ok = np.isfinite(dv)
        if not ok.any():
            S.append(0); T.append(0); continue
        sg = np.sign(np.nanmedian(dv)) or 1
        S.append(int(np.sum(sg * dv[ok] != dw[ok]))); T.append(int(ok.sum()))
    return np.array(S), np.array(T), far


res = {}
for spec in a.fits:
    name, run = spec.split("=")
    S, T, far = score(f"runs/kaggle/{run}/grids_{name}.npz")
    res[name] = dict(slips=int(S.sum()), pairs=int(T.sum()), rate=float(S.sum() / max(T.sum(), 1)), unassigned=far, S=S.tolist(), T=T.tolist())
    print(f"{name}: {S.sum()}/{T.sum()} = {S.sum()/max(T.sum(),1):.4f} per wrap; unassigned {far}", flush=True)

rng = np.random.default_rng(1); n = len(lad); out = {}
B = rng.integers(0, n, (a.nboot, n))
def rate(name, idx):
    S = np.array(res[name]["S"])[idx]; T = np.array(res[name]["T"])[idx]; return S.sum() / max(T.sum(), 1)
for name in res:
    r = np.array([rate(name, b) for b in B])
    out[name] = dict(slips=res[name]["slips"], pairs=res[name]["pairs"], rate=res[name]["rate"], unassigned=res[name]["unassigned"],
                     ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))])
    if a.ref and name != a.ref:
        d = np.array([rate(name, b) - rate(a.ref, b) for b in B])
        out[name]["diff_vs_ref"] = dict(ref=a.ref, mean=float(res[name]["rate"] - res[a.ref]["rate"]),
                                        ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_worse=float((d >= 0).mean()))
    print(name, {k: v for k, v in out[name].items() if k != "diff_vs_ref"}, out[name].get("diff_vs_ref"), flush=True)
if a.sweep:
    sw = a.sweep.split(","); picks = []; held = []
    for s in range(200):
        perm = rng.permutation(n); A_, B_ = perm[: n // 2], perm[n // 2:]
        best = min(sw, key=lambda k: rate(k, A_)); picks.append(best); held.append(rate(best, B_))
        if a.ref: held[-1] = (held[-1], rate(a.ref, B_))
    hv = np.array([h[0] if isinstance(h, tuple) else h for h in held]); rv = np.array([h[1] for h in held]) if a.ref else None
    out["selection_free"] = dict(sweep=sw, picked={k: picks.count(k) for k in sw}, held_out_rate_mean=float(hv.mean()),
                                 held_out_rate_ci=[float(np.percentile(hv, 2.5)), float(np.percentile(hv, 97.5))],
                                 ref_rate_on_same_halves=float(rv.mean()) if rv is not None else None)
    print("selection-free:", out["selection_free"], flush=True)
out["ladders"] = n; out["pairs"] = int(sum(len(w) - 1 for _, w in lad)); out["kind"] = a.kind; out["z"] = [Z0, Z1]
json.dump(out, open(a.out, "w"), indent=1)
