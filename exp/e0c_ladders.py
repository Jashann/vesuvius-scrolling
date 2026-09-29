"""E0c: does the local fringe phase count wraps correctly between annotated ladder points?

Ground truth: relative_windings.json ladders (single-z point chains with consecutive wind_a).
Signal: Lasagna cos at pyramid level 3 (19.2 um). Annotation coords are level 2 (9.6 um).
Metrics per consecutive pair: |phase cycles| vs |delta winding|; baseline: cos peak count.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import ndimage as ndi
from pcu import data, phase as ph

B = data.P4_SPIRAL
cos3 = data.lasagna("cos", 3)
gm4 = data.lasagna("grad_mag", 4)
rel = data.fetch_json(f"{B}/relative_windings.json")["collections"]

SCALE = 2.0  # level2 -> level3
rows = []
for cid, col in rel.items():
    pts = [(np.array(p["p"]), p["wind_a"]) for p in col["points"].values() if p["wind_a"] is not None]
    if len(pts) < 2:
        continue
    P = np.array([p for p, _ in pts]) / SCALE
    Wd = np.array([w for _, w in pts], float)
    if np.ptp(P[:, 2]) > 2:
        continue  # only single-slice ladders
    order = np.argsort(Wd); P, Wd = P[order], Wd[order]
    z = int(round(P[:, 2].mean()))
    x0, x1 = int(P[:, 0].min()) - 64, int(P[:, 0].max()) + 64
    y0, y1 = int(P[:, 1].min()) - 64, int(P[:, 1].max()) + 64
    img = data.read(cos3, (slice(z, z + 1), slice(y0, y1), slice(x0, x1)))[0].astype(float)
    if (img > 0).mean() < 0.5:
        continue
    c = img / 255 * 2 - 1
    win = np.outer(np.hanning(c.shape[0]), np.hanning(c.shape[1])) ** 0.25
    e, rx, ry = ph.riesz2d(c * win)
    phi, mx, my, coh = ph.phase_with_linefield(e, rx, ry, sigma=1.5)
    amp = np.sqrt(e**2 + rx**2 + ry**2)
    q, fold = ph.residues_linefield(phi, mx, my)
    defect = np.zeros_like(c, bool); defect[:-1, :-1] = (q != 0) | fold
    dist_def = ndi.distance_transform_edt(~defect)
    g4 = data.read(gm4, (slice(z // 2, z // 2 + 1), slice(y0 // 2, y1 // 2 + 2), slice(x0 // 2, x1 // 2 + 2)))[0].astype(float) / 4000.0
    for i in range(len(P) - 1):
        a = P[i, :2] - [x0, y0]; b = P[i + 1, :2] - [x0, y0]
        L = np.linalg.norm(b - a)
        n = max(int(L * 4), 8)
        t = np.linspace(0, 1, n)
        xs = a[0] + t * (b[0] - a[0]); ys = a[1] + t * (b[1] - a[1])
        co = [ys, xs]
        pp = ndi.map_coordinates(phi, co, order=0)
        mxx = ndi.map_coordinates(mx, co, order=0); myy = ndi.map_coordinates(my, co, order=0)
        cc = ndi.map_coordinates(c, co, order=1)
        # propagate orientation along the path, then integrate wrapped phase
        tot, cur_p, cur_x, cur_y = 0.0, pp[0], mxx[0], myy[0]
        for k in range(1, n):
            s = 1.0 if cur_x * mxx[k] + cur_y * myy[k] >= 0 else -1.0
            pk, xk, yk = pp[k] * s, mxx[k] * s, myy[k] * s
            tot += ph.wrap(pk - cur_p)
            cur_p, cur_x, cur_y = pk, xk, yk
        cyc = abs(tot) / (2 * np.pi)
        # baseline: number of cos maxima strictly inside the segment, plus one
        inner = cc[2:-2]
        pk_n = int(np.sum((inner[1:-1] > inner[:-2]) & (inner[1:-1] >= inner[2:]) & (inner[1:-1] > 0.0))) + 1
        dd = ndi.map_coordinates(dist_def, co, order=1)
        gco = [(ys + y0) / 2 - y0 // 2, (xs + x0) / 2 - x0 // 2]
        gint = float(np.sum(ndi.map_coordinates(g4, gco, order=1)) * (L / n) * 8.0)
        aa = ndi.map_coordinates(amp, co, order=1)
        rows.append(dict(col=cid, z=z, dw=Wd[i + 1] - Wd[i], L=L, cyc=cyc, peaks=pk_n,
                         dmin=float(dd.min()), amin=float(aa.min() / (np.median(amp) + 1e-9)), frac=float(abs(cyc - round(cyc))), gint=gint))

dw = np.array([r["dw"] for r in rows]); cyc = np.array([r["cyc"] for r in rows]); pk = np.array([r["peaks"] for r in rows])
L = np.array([r["L"] for r in rows])
print(f"pairs {len(rows)} from {len(set(r['col'] for r in rows))} ladders; delta-winding values {np.unique(dw, return_counts=True)}")
print(f"spacing per wrap (level-3 px): median {np.median(L/np.maximum(dw,1)):.2f}, 10th pct {np.percentile(L/np.maximum(dw,1),10):.2f}")
print(f"phase count exact: {np.mean(np.round(cyc) == dw):.3f}   peaks exact: {np.mean(pk == dw):.3f}")
for lo, hi in [(0, 4), (4, 6), (6, 9), (9, 99)]:
    m = (L / np.maximum(dw, 1) >= lo) & (L / np.maximum(dw, 1) < hi)
    if m.sum():
        print(f"  spacing {lo}-{hi}px: n={m.sum():4d} phase exact {np.mean(np.round(cyc[m]) == dw[m]):.3f} peaks exact {np.mean(pk[m] == dw[m]):.3f}")
json.dump(rows, open("runs/e0c_rows.json", "w"))
dmin = np.array([r["dmin"] for r in rows]); amin = np.array([r["amin"] for r in rows]); frac = np.array([r["frac"] for r in rows])
ok = np.round(cyc) == dw
print("certificate: min distance from the segment to any residue/fold")
for d in [0, 1, 2, 3, 4, 6, 8]:
    m = dmin > d
    print(f"  dmin>{d}: coverage {m.mean():.3f}  accuracy {ok[m].mean():.4f}  errors {np.sum(~ok[m])}")
for fr in [0.1, 0.2, 0.3]:
    m = (dmin > 2) & (frac < fr)
    print(f"  dmin>2 & |cyc-round|<{fr}: coverage {m.mean():.3f} accuracy {ok[m].mean():.4f} errors {np.sum(~ok[m])}")

gint = np.array([r["gint"] for r in rows])
k = 1.0 / np.median(gint[ok])
gc = gint * k
print(f"grad_mag integral scale k={k:.3f}; density-count exact {np.mean(np.round(gc) == dw):.3f}")
agree = np.round(gc) == np.round(cyc)
print(f"phase & density agree: coverage {agree.mean():.3f} accuracy {ok[agree].mean():.4f} errors {np.sum(~ok[agree])}")
for d in [2, 4]:
    m = agree & (dmin > d)
    print(f"  agree & dmin>{d}: coverage {m.mean():.3f} accuracy {ok[m].mean():.4f} errors {np.sum(~ok[m])}")
for tol in [0.25, 0.35]:
    m = agree & (np.abs(gc - np.round(gc)) < tol) & (frac < tol)
    print(f"  agree & both within {tol} of integer: coverage {m.mean():.3f} accuracy {ok[m].mean():.4f} errors {np.sum(~ok[m])}")
bad = ~ok & agree
print("remaining agreed errors (cyc, gc, L):", [(round(a,2), round(b,2), round(c,1)) for a,b,c in zip(cyc[bad], gc[bad], L[bad])])
