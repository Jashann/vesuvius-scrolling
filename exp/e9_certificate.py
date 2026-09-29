"""E9: multi-evidence certificate for adjacent-wrap constraints, measured on all human ladder pairs.

Evidence per pair (a, b): local phase count along the segment, Lasagna wrap-density integral,
whole-slice MCF field difference, distance to the nearest residue. Report precision vs coverage
for combinations; precision is what matters for constraints fed to a global fit.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import ndimage as ndi
from pcu import data, slice2d, mcfcut, phase as ph

Y0, Y1, X0, X1 = 416, 3296, 672, 4000
cos3 = data.lasagna("cos", 3); gm4 = data.lasagna("grad_mag", 4)
lad = []
for f in ("abs_winding.json", "relative_windings.json"):
    for cid, col in data.fetch_json(f"{data.P4_SPIRAL}/{f}")["collections"].items():
        P = [(p["p"], p["wind_a"]) for p in col["points"].values() if p["wind_a"] is not None]
        if len(P) < 2: continue
        zz = np.array([p[0][2] for p in P])
        if np.ptp(zz) > 2: continue
        P = sorted(P, key=lambda t: t[1])
        lad.append(dict(name=f[:3] + cid, z3=int(round(zz.mean() / 2)), xy=np.array([[p[0][0] / 2 - X0, p[0][1] / 2 - Y0] for p in P]), w=np.array([p[1] for p in P], float)))
zs = sorted(set(l["z3"] for l in lad))
rows = []
t0 = time.time()
for zi, z3 in enumerate(zs):
    c = data.read(cos3, (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0]
    ux, uy = slice2d.umbilicus_xy(z3 * 8)
    F = slice2d.winding_field(c, (ux / 8 - X0, uy / 8 - Y0), s=+1, unwrap=False)
    mask, psi, amp, th, phi = F["mask"], F["psi"], F["amp"], F["theta"], F["phi"]
    q = np.where(mask, amp / (np.percentile(amp[mask], 90) + 1e-9), 0).clip(0, 1) ** 2
    U, cut, nres, lab = mcfcut.unwrap_mcf(psi, mask, q)
    Wf = np.nan_to_num(U - th / (2 * np.pi))
    res = np.zeros(c.shape, bool); res[:-1, :-1] = ph.residues(psi) != 0
    dres = ndi.distance_transform_edt(~res)
    g = data.read(gm4, (slice(z3 // 2, z3 // 2 + 1), slice(Y0 // 2, Y1 // 2 + 1), slice(X0 // 2, X1 // 2 + 1)))[0].astype(float) / 4000.0
    for L in [l for l in lad if l["z3"] == z3]:
        for i in range(len(L["w"]) - 1):
            a, b = L["xy"][i], L["xy"][i + 1]
            dw = abs(L["w"][i + 1] - L["w"][i])
            n = max(int(np.linalg.norm(b - a) * 4), 8); t = np.linspace(0, 1, n)
            ys, xs = a[1] + t * (b[1] - a[1]), a[0] + t * (b[0] - a[0])
            loc = abs(np.sum(ph.wrap(np.diff(ndi.map_coordinates(phi, [ys, xs], order=0)))) / (2 * np.pi))
            den = float(np.sum(ndi.map_coordinates(g, [(ys + Y0) / 2 - Y0 // 2, (xs + X0) / 2 - X0 // 2], order=1)) * np.linalg.norm(b - a) / n * 8 * 0.808)
            wa, wb = ndi.map_coordinates(Wf, [[a[1], b[1]], [a[0], b[0]]], order=0)
            glob = abs(np.round(wb) - np.round(wa))
            rows.append(dict(z3=z3, dw=dw, loc=float(loc), den=den, glob=float(glob), dres=float(ndi.map_coordinates(dres, [ys, xs], order=1).min()),
                             L=float(np.linalg.norm(b - a))))
    print(f"[{zi+1}/{len(zs)}] pairs {len(rows)} ({time.time()-t0:.0f}s)", flush=True)
json.dump(rows, open("runs/e9_rows.json", "w"))
dw = np.array([r["dw"] for r in rows]); loc = np.array([r["loc"] for r in rows]); den = np.array([r["den"] for r in rows])
glob = np.array([r["glob"] for r in rows]); dres = np.array([r["dres"] for r in rows])
checks = {
    "local": np.round(loc) == dw,
    "density": np.round(den) == dw,
    "global": glob == dw,
}
for k, v in checks.items():
    print(f"{k:8s} accuracy {v.mean():.4f}")
base = np.ones(len(dw), bool)
combos = {
    "local=den": np.round(loc) == np.round(den),
    "local=den=glob": (np.round(loc) == np.round(den)) & (np.round(loc) == glob),
    "local=den=glob & tight": (np.round(loc) == np.round(den)) & (np.round(loc) == glob) & (np.abs(loc - np.round(loc)) < 0.25) & (np.abs(den - np.round(den)) < 0.3),
    "local=den=glob & tight & dres>2": (np.round(loc) == np.round(den)) & (np.round(loc) == glob) & (np.abs(loc - np.round(loc)) < 0.25) & (np.abs(den - np.round(den)) < 0.3) & (dres > 2),
}
for k, m in combos.items():
    pred = np.round(loc)
    print(f"{k:34s} coverage {m.mean():.3f}  precision {np.mean(pred[m] == dw[m]):.4f}  errors {np.sum(pred[m] != dw[m])}")
