"""E19: 3D crest graph. Per-slice crest pieces + certified +1 edges, plus same-wrap edges linking
overlapping crest pieces of neighbouring slices. One exact robust L1 solve for all slices.
Evaluate ladders + verified patches on the centre slice (and patches on every slice)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, tifffile
from scipy import ndimage as ndi
from pcu import data, slice2d, mcfcut, crestgraph, phase as ph

Y0, Y1, X0, X1 = 416, 3296, 672, 4000
zc = int(sys.argv[1]) if len(sys.argv) > 1 else 7847
K = int(os.environ.get("K", 4)); DZ = int(os.environ.get("DZ", 2)); WZ = float(os.environ.get("WZ", 0.25))
cos3 = data.lasagna("cos", 3); gm4 = data.lasagna("grad_mag", 4)
t0 = time.time()
labs, Es, offs = {}, {}, {}
off = 0
for k in range(-K, K + 1):
    z3 = zc + k * DZ
    c = data.read(cos3, (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0]
    ux, uy = slice2d.umbilicus_xy(z3 * 8)
    F = slice2d.winding_field(c, (ux / 8 - X0, uy / 8 - Y0), s=+1, unwrap=False)
    q = np.where(F["mask"], F["amp"] / (np.percentile(F["amp"][F["mask"]], 90) + 1e-9), 0).clip(0, 1) ** 2
    U, cut, nres, _ = mcfcut.unwrap_mcf(F["psi"], F["mask"], q, geo=True)
    Wf = np.nan_to_num(U - F["theta"] / (2 * np.pi))
    g = data.read(gm4, (slice(z3 // 2, z3 // 2 + 1), slice(Y0 // 2, Y1 // 2 + 1), slice(X0 // 2, X1 // 2 + 1)))[0].astype(float) / 4000.0
    g3 = ndi.zoom(g, 2, order=1)[: c.shape[0], : c.shape[1]]
    resmap = np.zeros(c.shape, bool); resmap[:-1, :-1] = ph.residues(F["psi"]) != 0
    lab = crestgraph.crest_pieces(F["phi"], F["mask"], F["theta"], F["amp"], residues=resmap)
    E = crestgraph.build_edges(lab, F["phi"], F["mx"], F["my"], F["R"], g3, Wf, F["mask"], F["theta"])
    lab = lab.astype(np.int64); lab[lab > 0] += off
    labs[k] = lab; offs[k] = off
    for (a, b, d), w in E.items():
        Es[(a + off, b + off, d)] = Es.get((a + off, b + off, d), 0) + w
    off = int(lab.max())
    print(f"slice {z3}: pieces {lab.max() - offs[k]}, edges {len(E)} ({time.time()-t0:.0f}s)", flush=True)
    del F, U, Wf, g3
# same-wrap edges between neighbouring slices: mutual best overlap, fixed weight
nz = 0
for k in range(-K, K):
    A = labs[k]; B = ndi.grey_dilation(labs[k + 1], size=3)
    m = (A > 0) & (B > 0)
    pairs, cnt = np.unique(np.stack([A[m], B[m]], 1), axis=0, return_counts=True)
    sa = np.bincount(A.ravel()); sb = np.bincount(labs[k + 1].ravel())
    bestA, bestB = {}, {}
    for (a, b), c_ in zip(pairs, cnt):
        if c_ > bestA.get(a, (0, 0))[1]: bestA[a] = (b, c_)
        if c_ > bestB.get(b, (0, 0))[1]: bestB[b] = (a, c_)
    for a, (b, c_) in bestA.items():
        if bestB.get(b, (None,))[0] == a and c_ >= 0.3 * min(sa[a], sb[b]) and c_ >= 5:
            Es[(int(a), int(b), 0)] = Es.get((int(a), int(b), 0), 0) + WZ
            nz += 1
print("z edges", nz, "total edges", len(Es), flush=True)
sizes = np.zeros(off + 1)
for lab in labs.values():
    sizes += np.bincount(lab.ravel(), minlength=off + 1)[: off + 1]
label, comp, info = crestgraph.solve_robust(Es, off, sizes)
info.pop("node_bad_frac", None)
print("solve", info, f"({time.time()-t0:.0f}s)", flush=True)
big = np.argmax(np.bincount(comp[comp >= 0]))
inbig = comp == big

metas = json.load(open("data/patches/meta_many.json"))


def labels_at(lab, xy):
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    yy = np.clip(np.round(xy[:, 1]).astype(int), 0, lab.shape[0] - 1); xx = np.clip(np.round(xy[:, 0]).astype(int), 0, lab.shape[1] - 1)
    pc = lab[iy[yy, xx], ix[yy, xx]]
    ok = (dist[yy, xx] <= 3) & inbig[pc]
    return np.where(ok, label[pc], np.nan)


tot = {"pp_ok": 0, "pp_n": 0, "sw": 0, "np": 0}
for k in (0,) if os.environ.get("ALLSLICES", "0") == "0" else range(-K, K + 1):
    z3 = zc + k * DZ; lab = labs[k]
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    for nme, m in metas.items():
        if not (m["bbox"][0][2] + 5 < 2 * z3 < m["bbox"][1][2] - 5 and m.get("area_cm2", 0) > 0.3): continue
        d_ = f"data/patches/{nme}"
        if not os.path.exists(f"{d_}z.tif"): continue
        x, y, z = (tifffile.imread(f"{d_}{a}.tif") for a in "xyz")
        pts = []; valid = (x > 0) & (z > 0)
        for sa, sb in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))), ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            za, zb = z[sa], z[sb]; cr = valid[sa] & valid[sb] & ((za - 2 * z3) * (zb - 2 * z3) <= 0) & (za != zb)
            t = (2 * z3 - za[cr]) / (zb[cr] - za[cr])
            pts.append(np.stack([x[sa][cr] + t * (x[sb][cr] - x[sa][cr]), y[sa][cr] + t * (y[sb][cr] - y[sa][cr])], 1))
        pts = np.concatenate(pts)
        if len(pts) < 10: continue
        xy = np.stack([pts[:, 0] / 2 - X0, pts[:, 1] / 2 - Y0], 1)
        yy = np.clip(np.round(xy[:, 1]).astype(int), 0, lab.shape[0] - 1); xx = np.clip(np.round(xy[:, 0]).astype(int), 0, lab.shape[1] - 1)
        pc = lab[iy[yy, xx], ix[yy, xx]]
        v = np.where((dist[yy, xx] <= 3) & inbig[pc], label[pc], np.nan); v = v[np.isfinite(v)]
        if len(v) < 10: continue
        vals, cnt = np.unique(v, return_counts=True)
        tot["pp_ok"] += cnt.max(); tot["pp_n"] += len(v); tot["np"] += 1; tot["sw"] += int(cnt.max() < 0.9 * len(v))
print(f"patches: points on modal label {tot['pp_ok']}/{tot['pp_n']} = {tot['pp_ok']/max(tot['pp_n'],1):.3f}; switched {tot['sw']}/{tot['np']}")
# ladders on centre slice
lab = labs[0]; sl = n_ = 0
for f in ("abs_winding.json", "relative_windings.json"):
    for cid, col in data.fetch_json(f"{data.P4_SPIRAL}/{f}")["collections"].items():
        P = [(p["p"], p["wind_a"]) for p in col["points"].values() if p["wind_a"] is not None]
        if len(P) < 2: continue
        zz = np.array([p[0][2] for p in P])
        if np.ptp(zz) > 2 or int(round(zz.mean() / 2)) != zc: continue
        P = sorted(P, key=lambda t: t[1])
        xy = np.array([[p[0][0] / 2 - X0, p[0][1] / 2 - Y0] for p in P]); w = np.array([p[1] for p in P], float)
        v = labels_at(lab, xy); d = np.diff(v); dw = np.diff(w); good = np.isfinite(d)
        sl += int(np.sum(d[good] != dw[good])); n_ += int(good.sum())
print(f"ladders centre slice: slips {sl}/{n_}")
