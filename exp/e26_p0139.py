"""E26: generalization to PHerc0139 (never used in development). Ground truth: 37 whole-wrap segments
w023..w059 (absolute winding). Metric: in each 5-degree angular sector, (our label - w) must be one
constant across all wraps (labels and w use different branch cuts, so the constant may step only at
the two cut angles). Score = fraction of segment points on their sector's modal offset."""
import sys, os, json, re, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, requests, tifffile
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage as ndi
from pcu import data, slice2d, mcfcut, crestgraph, phase as ph

SC = "PHerc0139"; S3 = data.S3
Y0, Y1, X0, X1 = 1000, 2350, 1000, 2450
zs = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1500, 2200, 2900]
# segments
lst = requests.get(f"{S3}/?list-type=2&delimiter=/&prefix={SC}/segments/").text
segs = [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", lst) if re.search(r"-w\d{3}_", p)]
root = "data/p0139_segments"; os.makedirs(root, exist_ok=True)
def fetch(p):
    m = re.search(r"-w(\d{3})_", p); w = int(m.group(1)); sid = p.rstrip("/").split("/")[-1].split("-")[0]
    base = f"{p}mesh/{sid}-on-20260102150214-2.399um.tifxyz/"
    out = []
    for a in "xyz":
        f = f"{root}/{sid}_w{w}_{a}.tif"
        if not os.path.exists(f):
            r = requests.get(f"{S3}/{base}{a}.tif", timeout=300)
            if r.status_code != 200: return None
            open(f, "wb").write(r.content)
        out.append(f)
    return w, out
with ThreadPoolExecutor(12) as ex:
    S = [s for s in ex.map(fetch, segs) if s]
print("segments", len(S), sorted(w for w, _ in S), flush=True)
grids = [(w, [tifffile.imread(f) for f in fs]) for w, fs in S]

def seg_points(zf):
    out = []
    for w, (x, y, z) in grids:
        valid = (x > 0) & (z > 0); pts = []
        for sa, sb in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))), ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
            za, zb = z[sa], z[sb]; cr = valid[sa] & valid[sb] & ((za - zf) * (zb - zf) <= 0) & (za != zb)
            t = (zf - za[cr]) / (zb[cr] - za[cr])
            pts.append(np.stack([x[sa][cr] + t * (x[sb][cr] - x[sa][cr]), y[sa][cr] + t * (y[sb][cr] - y[sa][cr])], 1))
        pts = np.concatenate(pts)
        if len(pts): out.append((w, np.stack([pts[:, 0] / 8 - X0, pts[:, 1] / 8 - Y0], 1)))
    return out

def normal_pairs(P, step=6, maxd=25, dw=1):
    """Ground-truth adjacent-wrap pairs: sampled points on wrap w and their nearest point on w+1."""
    from scipy.spatial import cKDTree
    byw = {w: p for w, p in P}
    A, B = [], []
    for w, p in P:
        if w + dw not in byw: continue
        t = cKDTree(byw[w + dw]); d, j = t.query(p[::step])
        ok = d < maxd * dw
        A.append(p[::step][ok]); B.append(byw[w + dw][j[ok]])
    return np.concatenate(A), np.concatenate(B)


def sector_slips(vals, ws, th, sign, nb=72):
    """Per angular sector: median label per wrap, then check consecutive wraps differ by sign*1."""
    b = ((th + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    ok = tot = 0
    for k in np.unique(b):
        m = b == k
        wl = {}
        for w in np.unique(ws[m]):
            mm = m & (ws == w)
            if mm.sum() >= 2: wl[int(w)] = np.median(vals[mm])
        for w in wl:
            if w + 1 in wl:
                tot += 1; ok += (wl[w + 1] - wl[w]) == sign
    return ok, tot


def sector_score(vals, ws, th, sign):
    off = vals - sign * ws
    b = ((th + np.pi) / (2 * np.pi) * 72).astype(int) % 72
    ok = 0
    for k in np.unique(b):
        m = b == k
        _, cnt = np.unique(off[m], return_counts=True); ok += cnt.max()
    return ok / len(off)

cos3 = data.lasagna_for(SC, "cos", 3); gm4 = data.lasagna_for(SC, "grad_mag", 4)
for z3 in zs:
    t0 = time.time()
    c = data.read(cos3, (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0]
    ux, uy = slice2d.umbilicus_xy(z3 * 8, SC); cen = (ux / 8 - X0, uy / 8 - Y0)
    F = slice2d.winding_field(c, cen, s=+1, unwrap=False)
    q = np.where(F["mask"], F["amp"] / (np.percentile(F["amp"][F["mask"]], 90) + 1e-9), 0).clip(0, 1) ** 2
    U, cut, nres, _ = mcfcut.unwrap_mcf(F["psi"], F["mask"], q, geo=True)
    Wf = np.nan_to_num(U - F["theta"] / (2 * np.pi))
    g = data.read(gm4, (slice(z3 // 2, z3 // 2 + 1), slice(Y0 // 2, Y1 // 2 + 1), slice(X0 // 2, X1 // 2 + 1)))[0].astype(float) / 4000.0
    g3 = ndi.zoom(g, 2, order=1)[: c.shape[0], : c.shape[1]]
    resmap = np.zeros(c.shape, bool); resmap[:-1, :-1] = ph.residues(F["psi"]) != 0
    lab = crestgraph.crest_pieces(F["phi"], F["mask"], F["theta"], F["amp"], residues=resmap)
    E = crestgraph.build_edges(lab, F["phi"], F["mx"], F["my"], F["R"], g3, Wf, F["mask"], F["theta"], kmax=3)
    sizes = np.bincount(lab.ravel(), minlength=lab.max() + 1).astype(float)
    label, comp, info = crestgraph.solve_robust(E, lab.max(), sizes); info.pop("node_bad_frac", None)
    big = np.argmax(np.bincount(comp[comp >= 0])); inbig = comp == big
    dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    P = seg_points(z3 * 8)
    ws = np.concatenate([np.full(len(p), w) for w, p in P]); xy = np.concatenate([p for _, p in P])
    th = np.arctan2(xy[:, 1] - cen[1], xy[:, 0] - cen[0])
    inside = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < c.shape[1] - 1) & (xy[:, 1] < c.shape[0] - 1)
    xy, ws, th = xy[inside], ws[inside], th[inside]
    vm = np.round(ndi.map_coordinates(Wf, [xy[:, 1], xy[:, 0]], order=1))
    yy, xx = np.round(xy[:, 1]).astype(int), np.round(xy[:, 0]).astype(int)
    pc = lab[iy[yy, xx], ix[yy, xx]]
    # local wrap spacing along the normal from the Lasagna density (L3 px per wrap)
    spacing = 1.0 / np.maximum(ndi.map_coordinates(g3, [xy[:, 1], xy[:, 0]], order=1) * 8 * 0.808, 1e-3)
    res = {}
    for snap in (1.0, 3.0):
        vc = np.where((dist[yy, xx] <= snap) & inbig[pc], label[pc], np.nan)
        for name, v in (("mcf_field", vm), (f"crest_snap{snap:g}", vc)):
            if name == "mcf_field" and snap == 3.0: continue
            for sub, msk in (("all", np.ones(len(v), bool)), ("spacing>=5px", spacing >= 5)):
                f = np.isfinite(v) & msk
                if f.sum() < 100: continue
                sign = np.sign(np.polyfit(ws[f], v[f], 1)[0]) or 1
                o, t = sector_slips(v[f], ws[f], th[f], sign)
                res[f"{name}/{sub}"] = f"adjacent-wrap correct {o}/{t} = {o/max(t,1):.3f}"
    res["median_spacing_px"] = round(float(np.median(spacing)), 2)
    A, B = normal_pairs(P)
    inb = lambda q: (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < c.shape[1] - 1) & (q[:, 1] < c.shape[0] - 1)
    k = inb(A) & inb(B); A, B = A[k], B[k]
    def lab_at(q, snap):
        yy, xx = np.round(q[:, 1]).astype(int), np.round(q[:, 0]).astype(int)
        pc = lab[iy[yy, xx], ix[yy, xx]]
        return np.where((dist[yy, xx] <= snap) & inbig[pc], label[pc], np.nan)
    for name, va, vb in (("mcf", np.round(ndi.map_coordinates(Wf, [A[:, 1], A[:, 0]], order=1)), np.round(ndi.map_coordinates(Wf, [B[:, 1], B[:, 0]], order=1))),
                         ("crest", lab_at(A, 1.5), lab_at(B, 1.5))):
        d = vb - va; f = np.isfinite(d)
        sg = np.sign(np.median(d[f])) or 1
        res[f"pairs_{name}"] = f"{np.mean(d[f] == sg):.3f} of {f.sum()} (coverage {f.mean():.2f})"
    lc = []
    for a, b in zip(A[::3], B[::3]):
        n = max(int(np.linalg.norm(b - a) * 4), 8); t = np.linspace(0, 1, n)
        ys_, xs_ = a[1] + t * (b[1] - a[1]), a[0] + t * (b[0] - a[0])
        pp = ndi.map_coordinates(F["phi"], [ys_, xs_], order=0); mxx = ndi.map_coordinates(F["mx"], [ys_, xs_], order=0); myy = ndi.map_coordinates(F["my"], [ys_, xs_], order=0)
        tot_, cp, cx_, cy_ = 0.0, pp[0], mxx[0], myy[0]
        for kk in range(1, n):
            sg_ = 1.0 if cx_ * mxx[kk] + cy_ * myy[kk] >= 0 else -1.0
            tot_ += ph.wrap(pp[kk] * sg_ - cp); cp, cx_, cy_ = pp[kk] * sg_, mxx[kk] * sg_, myy[kk] * sg_
        lc.append(abs(tot_) / (2 * np.pi))
    lc = np.array(lc); dd = np.linalg.norm(B - A, axis=1)[::3]
    from pcu import constraints as cst
    cert = []; truth_ok = []
    for a, b in zip(A[::3], B[::3]):
        pa, pb = np.array([a[1], a[0]]), np.array([b[1], b[0]])
        ok, info = cst.certify(pa, pb, F["phi"], g3, Wf)
        cert.append(ok)
    cert = np.array(cert)
    res["certificate"] = f"coverage {cert.mean():.3f}, precision {np.mean(np.round(lc[cert]) == 1) if cert.any() else float('nan'):.4f} (n {cert.sum()})"
    A2, B2 = normal_pairs(P, dw=2)
    k2 = inb(A2) & inb(B2); A2, B2 = A2[k2][::3], B2[k2][::3]
    fp = [cst.certify(np.array([a[1], a[0]]), np.array([b[1], b[0]]), F["phi"], g3, Wf)[0] for a, b in zip(A2, B2)]
    res["false_cert_on_2wrap_pairs"] = f"{np.sum(fp)}/{len(fp)}"
    dd3 = np.linalg.norm(B - A, axis=1)[::3]
    z0 = np.round(lc) == 0
    res["count0_pair_dist"] = f"median {np.median(dd3[z0]):.1f}px, <3px {np.mean(dd3[z0] < 3):.2f}; count1 median {np.median(dd3[~z0]):.1f}"
    d2 = np.linalg.norm(B2 - A2, axis=1); fpa = np.array(fp)
    res["false_cert_2wrap_dist"] = f"falsely certified: median {np.median(d2[fpa]) if fpa.any() else 0:.1f}px; all 2-wrap pairs median {np.median(d2):.1f}px"
    good1 = (dd3 >= 4) & (dd3 <= 16)
    good2 = (d2 >= 10) & (d2 <= 30)
    res["CLEAN_local_count_eq1"] = f"{np.mean(np.round(lc[good1]) == 1):.3f} (n {good1.sum()})"
    res["CLEAN_cert_coverage"] = f"{cert[good1].mean():.3f}"
    res["CLEAN_false_cert_2wrap"] = f"{fpa[good2].sum()}/{good2.sum()} = {fpa[good2].mean():.4f}"
    res["local_count_eq1"] = f"{np.mean(np.round(lc) == 1):.3f} (0: {np.mean(np.round(lc)==0):.2f}, 2+: {np.mean(np.round(lc)>=2):.2f}); pair dist median {np.median(dd):.1f}px"
    print(f"z3={z3}: wraps crossing {len(P)}, points {len(ws)}, residues {nres}, graph {info}", res, f"{time.time()-t0:.0f}s", flush=True)
