"""E72: reproduce a TRACE hotspot from independent geometry. For each other fit, find the winding whose
surface passes closest to the hotspot's 3D points, crop that fit grid to the neighbourhood, snap + flatten +
render it (survey.winding_volume), run the TRACE ensemble, and compare the probability on the hotspot's
3D points against the rest of the crop.
Usage: e72_reproduce.py SCROLL HOT_XYZ.npy GRIDS.npz [GRIDS.npz ...]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ext"))
sys.path[:0] = [f"{root}/villa-ink/ink-detection", f"{root}/villa/vesuvius/src"]
import numpy as np, torch, imageio
from scipy.spatial import cKDTree
from pcu import data, survey
from koine_machines.inference.infer import build_repo_training_model_bundle
from vesuvius.image_proc.intensity.normalization import normalize_robust

sc = sys.argv[1]; H = np.load(sys.argv[2]).T  # (n, 3) x, y, z
H = H + np.array([0, 0, float(os.environ.get("ZSHIFT", "0"))])
TAG = os.environ.get("TAG", "")
OUT = "runs/e72"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CK = {"sA": "runs/kaggle/pcu-trace-train/sA_best.pth", "sB": "runs/kaggle/pcu-trace-train/sB_best.pth",
      "v2A": "runs/modal_v2/v2A_best.pth"}
nets = {k: build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev).eval()
        for k, p in CK.items()}
S = data.SCROLLS[sc]
vol = data.open_array(S["vol"] + "/0"); surf = data.open_array(S["surf"] + "/0")
umb = data.umbilicus_for(sc)["control_points"]; cz = np.array([p["z"] for p in umb])
i = int(np.argmin(abs(cz - np.median(H[:, 2])))); cxy = (umb[i]["x"], umb[i]["y"])
hc = H.mean(0)


def predict(net, v, P=128, Z=17, stride=64):
    Hh, Wd = v.shape[1:]
    out = np.zeros((Hh, Wd), np.float32); wt = np.zeros((Hh, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    co = [(r, q) for r in range(0, max(Hh - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    c = v.shape[0] // 2; z0 = c - Z // 2
    with torch.no_grad():
        for k in range(0, len(co), 32):
            b = co[k:k + 32]
            x = np.stack([v[z0:z0 + Z, r:r + P, q:q + P] for r, q in b]).astype(np.float32)
            keep = (x[:, Z // 2] > 0).reshape(len(b), -1).mean(1) > 0.3
            if not keep.any():
                continue
            x = np.stack([normalize_robust(p) for p in x]).astype(np.float32)
            pr = torch.sigmoid(net(torch.from_numpy(x)[:, None].to(dev)).float())[:, 0].cpu().numpy()
            for (r, q), pp, kk in zip(b, pr, keep):
                if kk:
                    out[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
    return np.where(wt > 0, out / np.maximum(wt, 1e-6), 0)


def best_window(E, v, px_mm=0.009362):
    win = int(20 / px_mm); b = 0.0
    for r0 in range(0, max(E.shape[0] - win, 0) + 1, win // 4):
        for c0 in range(0, max(E.shape[1] - win, 0) + 1, win // 4):
            vv = v[r0:r0 + win, c0:c0 + win]
            if vv.size and vv.mean() > 0.6:
                b = max(b, float(np.percentile(E[r0:r0 + win, c0:c0 + win][vv], 95)))
    return round(b, 3)


res = {}
for gp in sys.argv[3:]:
    G = np.load(gp); name = os.path.basename(gp).replace("grids_", "").replace(".npz", "")
    best = None
    for w in G.files:
        g = G[w]; P3 = g.reshape(3, -1).T; ok = (P3 > 0).all(1)
        if not ok.any():
            continue
        near = np.linalg.norm(P3[ok] - hc, axis=1).min()
        if near > 400:
            continue
        d, _ = cKDTree(P3[ok]).query(H[::5])
        m = float(np.median(d))
        if best is None or m < best[0]:
            best = (m, w)
    if best is None:
        print(name, "no winding near the hotspot"); continue
    w = best[1]; g = G[w]
    # crop the fit grid to cells within 12 mm (1280 vox) of the hotspot centre
    dd = np.linalg.norm(g.transpose(1, 2, 0) - hc, axis=2)
    rr, cc = np.nonzero(dd < 1280)
    gc = g[:, rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()
    sv, fg, st = survey.winding_volume(gc, vol, surf, cxy, layers=21, px=1.0)
    valid = sv[sv.shape[0] // 2] > 0
    E = np.mean([predict(n, sv) for n in nets.values()], 0)
    # probability at the hotspot's 3D points (nearest flattened-grid cell, step 5)
    F = fg.reshape(3, -1).T; fok = (F > 0).all(1)
    tree = cKDTree(F[fok]); idx = np.flatnonzero(fok)
    d, j = tree.query(H)
    rr5, cc5 = np.unravel_index(idx[j], fg.shape[1:])
    on = d < 25
    ph = E[np.clip(rr5[on] * 5, 0, E.shape[0] - 1), np.clip(cc5[on] * 5, 0, E.shape[1] - 1)]
    r = dict(winding=w, median_dist_vox=round(best[0], 1), hot_points_on_surface=int(on.sum()), n_hot=len(H),
             p_hot_median=round(float(np.median(ph)), 3) if on.any() else None,
             p_crop_p50=round(float(np.percentile(E[valid], 50)), 3), p_crop_p95=round(float(np.percentile(E[valid], 95)), 3),
             frac_crop_above_hot=round(float((E[valid] > np.median(ph)).mean()), 3) if on.any() else None,
             best_window_p95=best_window(E, valid))
    res[name] = r
    print(name, r, flush=True)
    imageio.imwrite(f"{OUT}/{TAG}{name}_{w}_ens.png", (np.clip(E, 0, 1) * 255).astype(np.uint8))
    imageio.imwrite(f"{OUT}/{TAG}{name}_{w}_mid.jpg", sv[sv.shape[0] // 2])
    mk = np.zeros(E.shape, np.uint8)
    mk[np.clip(rr5[on] * 5, 0, E.shape[0] - 1), np.clip(cc5[on] * 5, 0, E.shape[1] - 1)] = 255
    imageio.imwrite(f"{OUT}/{TAG}{name}_{w}_hotmask.png", mk)
json.dump(res, open(f"{OUT}/{TAG}reproduce.json", "w"), indent=1)
