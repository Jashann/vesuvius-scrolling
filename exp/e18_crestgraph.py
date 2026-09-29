"""E18: crest-graph synchronization on a slice; evaluate on ladders and verified patches."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, tifffile
from scipy import ndimage as ndi
from pcu import data, slice2d, mcfcut, crestgraph

Y0, Y1, X0, X1 = 416, 3296, 672, 4000
z3 = int(sys.argv[1]) if len(sys.argv) > 1 else 7847
t0 = time.time()
c = data.read(data.lasagna("cos", 3), (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0]
ux, uy = slice2d.umbilicus_xy(z3 * 8)
F = slice2d.winding_field(c, (ux / 8 - X0, uy / 8 - Y0), s=+1, unwrap=False)
q = np.where(F["mask"], F["amp"] / (np.percentile(F["amp"][F["mask"]], 90) + 1e-9), 0).clip(0, 1) ** 2
U, cut, nres, _ = mcfcut.unwrap_mcf(F["psi"], F["mask"], q, geo=True)
Wf = np.nan_to_num(U - F["theta"] / (2 * np.pi))
g = data.read(data.lasagna("grad_mag", 4), (slice(z3 // 2, z3 // 2 + 1), slice(Y0 // 2, Y1 // 2 + 1), slice(X0 // 2, X1 // 2 + 1)))[0].astype(float) / 4000.0
g3 = ndi.zoom(g, 2, order=1)[: c.shape[0], : c.shape[1]]
from pcu import phase as ph
resmap = np.zeros(c.shape, bool); resmap[:-1, :-1] = ph.residues(F["psi"]) != 0
if os.environ.get("PIECES", "cos") in ("recto", "union"):
    REC = "PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr"
    rec = data.read(data.open_array(f"{REC}/3"), (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0].astype(float) / 255
    lab = crestgraph.recto_pieces(ndi.gaussian_filter(rec, 0.5), F["mask"], F["theta"])
    if os.environ.get("PIECES") == "union":
        labc = crestgraph.crest_pieces(F["phi"], F["mask"], F["theta"], F["amp"], residues=resmap)
        far = ~ndi.binary_dilation(lab > 0, iterations=3)
        labc = np.where(far, labc, 0)
        sz = np.bincount(labc.ravel()); keep = sz >= 12; keep[0] = False
        labc = np.where(keep[labc], labc, 0)
        u, inv = np.unique(labc[labc > 0], return_inverse=True)
        lab2 = lab.copy().astype(np.int64); lab2[labc > 0] = lab.max() + 1 + inv
        lab = lab2
else:
    lab = crestgraph.crest_pieces(F["phi"], F["mask"], F["theta"], F["amp"], residues=resmap if os.environ.get("RESCUT", "1") == "1" else None)
npieces = lab.max()
rmap = None
if os.environ.get("GOLD", "0") != "0":
    REC_ = "PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr"
    rmap = ndi.gaussian_filter(data.read(data.open_array(f"{REC_}/2"), (slice(2 * z3, 2 * z3 + 1), slice(2 * Y0, 2 * Y1), slice(2 * X0, 2 * X1)))[0].astype(float) / 255, 0.5)
gw = None if os.environ.get("GOLD", "0") in ("0", "only") else float(os.environ["GOLD"])
E = crestgraph.build_edges(lab, F["phi"], F["mx"], F["my"], F["R"], g3, Wf, F["mask"], F["theta"], snap=float(os.environ.get("SNAP", 1.5)), kmax=int(os.environ.get("KMAX", 1)), rmap=rmap, gold_w=gw)
sizes = np.bincount(lab.ravel(), minlength=npieces + 1).astype(float)
label, comp, info = (crestgraph.solve_robust if os.environ.get("ROBUST", "1") == "1" else crestgraph.solve)(E, npieces, sizes)
info.pop("node_bad_frac", None)
print("pieces", npieces, info, f"{time.time()-t0:.0f}s", flush=True)
big = np.argmax(np.bincount(comp[comp >= 0])) if (comp >= 0).any() else -1
inbig = comp == big
print("largest component covers", int(sizes[np.nonzero(inbig)[0]].sum()), "of", int(sizes[1:].sum()), "crest px")

# evaluation: map points to the nearest crest piece (within 3 px) in the largest component
dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
def labels_at(xy):
    yy, xx = np.round(xy[:, 1]).astype(int), np.round(xy[:, 0]).astype(int)
    yy = np.clip(yy, 0, lab.shape[0] - 1); xx = np.clip(xx, 0, lab.shape[1] - 1)
    pc = lab[iy[yy, xx], ix[yy, xx]]
    ok = (dist[yy, xx] <= 3) & inbig[pc]
    return np.where(ok, label[pc], np.nan)

# ladders
tot = sl = cov = n = 0
for f in ("abs_winding.json", "relative_windings.json"):
    for cid, col in data.fetch_json(f"{data.P4_SPIRAL}/{f}")["collections"].items():
        P = [(p["p"], p["wind_a"]) for p in col["points"].values() if p["wind_a"] is not None]
        if len(P) < 2: continue
        zz = np.array([p[0][2] for p in P])
        if np.ptp(zz) > 2 or int(round(zz.mean() / 2)) != z3: continue
        P = sorted(P, key=lambda t: t[1])
        xy = np.array([[p[0][0] / 2 - X0, p[0][1] / 2 - Y0] for p in P]); w = np.array([p[1] for p in P], float)
        v = labels_at(xy); d = np.diff(v); dw = np.diff(w); good = np.isfinite(d)
        n += len(dw); cov += good.sum(); sl += int(np.sum(d[good] != dw[good])); tot += int(good.sum())
print(f"ladders: covered {cov}/{n} pairs, slips {sl}/{tot}")
# verified patches
metas = json.load(open("data/patches/meta_many.json"))
pp_ok = pp_n = sw = npch = 0
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
    v = labels_at(np.stack([pts[:, 0] / 2 - X0, pts[:, 1] / 2 - Y0], 1)); v = v[np.isfinite(v)]
    if len(v) < 10: continue
    vals, cnt = np.unique(v, return_counts=True)
    pp_ok += cnt.max(); pp_n += len(v); npch += 1; sw += int(cnt.max() < 0.9 * len(v))
print(f"patches: points on modal label {pp_ok}/{pp_n} = {pp_ok/max(pp_n,1):.3f}; switched {sw}/{npch}")
np.savez_compressed(f"runs/e18_z{z3}_{os.environ.get('PIECES','cos')}.npz", lab=lab, label=label, comp=comp)
