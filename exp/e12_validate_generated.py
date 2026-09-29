"""E12: validate automatically generated certified ladder steps against human ladders.

For each annotated slice, generate certified ladders with a dense seed grid, then for every generated
step (a -> b, claimed +1 wrap) find human ladder points lying within `tol` px of a and of b on the same
slice. If both endpoints match human points, the human winding difference must be exactly 1.
Also report the same check for UNcertified steps (steps that failed the certificate) as a control.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from pcu import data, slice2d, mcfcut, constraints

Y0, Y1, X0, X1 = 416, 3296, 672, 4000
tol = float(os.environ.get("TOL", 2.5))
cos3 = data.lasagna("cos", 3); gm4 = data.lasagna("grad_mag", 4)
REC = "PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr"
RL = int(os.environ.get("RECTO_LEVEL", 1))  # 1 = 4.8 um recto map (1,047/1,047), 2 = 9.6 um (942/942, the released files)
SC = 8 // (2 ** RL)                          # L3 px -> recto-map px
rec1 = data.open_array(f"{REC}/{RL}")


def recto_count(r1, a, b, oy, ox, shift=1.5, thr=0.35):
    """Recto bands crossed on the segment a->b (L3 coords), both ends shifted `shift` L3 px outward;
    counted on the recto map r1 (level RL) whose origin is (oy, ox) in its px."""
    d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
    a2, b2 = (a + shift * d) * SC, (b + shift * d) * SC
    n = max(int(np.linalg.norm(b2 - a2) * 2), 8); t = np.linspace(0, 1, n)
    prof = ndi.map_coordinates(r1, [a2[0] + t * (b2[0] - a2[0]) - oy, a2[1] + t * (b2[1] - a2[1]) - ox], order=1)
    above = prof > thr
    return int(np.sum(np.diff(above.astype(int)) == 1)) + int(above[0])
lad = []
for f in ("abs_winding.json", "relative_windings.json"):
    for cid, col in data.fetch_json(f"{data.P4_SPIRAL}/{f}")["collections"].items():
        P = [(p["p"], p["wind_a"]) for p in col["points"].values() if p["wind_a"] is not None]
        if len(P) < 2: continue
        zz = np.array([p[0][2] for p in P])
        if np.ptp(zz) > 2: continue
        lad.append(dict(z3=int(round(zz.mean() / 2)), yx=np.array([[p[0][1] / 2 - Y0, p[0][0] / 2 - X0] for p in P]), w=np.array([p[1] for p in P], float), id=f + cid))
zs = sorted(set(l["z3"] for l in lad))
stats = dict(cert=[0, 0], uncert=[0, 0], cert_recto=[0, 0], cert_norecto=[0, 0])
t0 = time.time()
for zi, z3 in enumerate(zs):
    c = data.read(cos3, (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0]
    ux, uy = slice2d.umbilicus_xy(z3 * 8)
    F = slice2d.winding_field(c, (ux / 8 - X0, uy / 8 - Y0), s=+1, unwrap=False)
    q = np.where(F["mask"], F["amp"] / (np.percentile(F["amp"][F["mask"]], 90) + 1e-9), 0).clip(0, 1) ** 2
    U, cut, nres, lab = mcfcut.unwrap_mcf(F["psi"], F["mask"], q)
    Wf = np.nan_to_num(U - F["theta"] / (2 * np.pi))
    g = data.read(gm4, (slice(z3 // 2, z3 // 2 + 1), slice(Y0 // 2, Y1 // 2 + 1), slice(X0 // 2, X1 // 2 + 1)))[0].astype(float) / 4000.0
    g3 = ndi.zoom(g, 2, order=1)[: c.shape[0], : c.shape[1]]
    L = [l for l in lad if l["z3"] == z3]
    allp = np.concatenate([l["yx"] for l in L])
    oy, ox = int((allp[:, 0].min() + Y0 - 40) * SC), int((allp[:, 1].min() + X0 - 40) * SC)
    ey, ex = int((allp[:, 0].max() + Y0 + 40) * SC), int((allp[:, 1].max() + X0 + 40) * SC)
    r1 = ndi.gaussian_filter(data.read(rec1, (slice(SC * z3, SC * z3 + 1), slice(oy, ey), slice(ox, ex)))[0].astype(float) / 255, 1.0)
    oy -= Y0 * SC; ox -= X0 * SC
    P = np.concatenate([l["yx"] for l in L]); Wh = np.concatenate([l["w"] for l in L]); Lid = np.concatenate([[k] * len(l["w"]) for k, l in enumerate(L)])
    tree = cKDTree(P)
    # seed densely near human ladders so generated steps overlap them
    for k, l in enumerate(L):
        for p0 in l["yx"]:
            p = np.array(p0, float)
            mloc = np.array([F["mx"][int(p[0]), int(p[1])], F["my"][int(p[0]), int(p[1])]])
            for direction in (1.0, -1.0):
                cur = p
                for _ in range(3):
                    nxt = constraints._next_crest(F["phi"], F["mx"] * direction, F["my"] * direction, cur, 1.0)
                    if nxt is None:
                        break
                    ok, info = constraints.certify(cur, nxt, F["phi"], g3, Wf)
                    ia = tree.query_ball_point(cur, tol); ib = tree.query_ball_point(nxt, tol)
                    pairs = [(i, j) for i in ia for j in ib if Lid[i] == Lid[j] and i != j]
                    if pairs:
                        i, j = pairs[0]
                        correct = abs(Wh[j] - Wh[i]) == 1
                        key = "cert" if ok else "uncert"
                        stats[key][0] += int(correct); stats[key][1] += 1
                        if ok:
                            rk = "cert_recto" if recto_count(r1, cur, nxt, oy, ox) == 1 else "cert_norecto"
                            stats[rk][0] += int(correct); stats[rk][1] += 1
                    if not ok:
                        break
                    cur = nxt
    print(f"[{zi+1}/{len(zs)}] " + "  ".join(f"{k} {a}/{b}" for k, (a, b) in stats.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
for k, (a, b) in stats.items():
    print(f"{k}: {a}/{b} = {a/max(b,1):.4f} of generated steps matching human ladders are correct")
