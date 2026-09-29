"""E73: vet a TRACE hotspot straight from a fit grid (no saved flattened grid needed). For the target winding and
its neighbours (same column range = same angular sector of the spiral fit), crop the fit grid, snap + flatten
+ render 41 layers, and score sA/sB/v2A: best 2x2 cm window p95 at depth offset 0 for all windings, plus the
depth profile (-8..+8) for the target. Usage: e73_vet.py SCROLL GRIDS.npz WINDING F0 F1 NEIGHBOURS(comma) TAG"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ext"))
sys.path[:0] = [f"{root}/villa-ink/ink-detection", f"{root}/villa/vesuvius/src"]
import numpy as np, torch, imageio
from pcu import data, survey
from koine_machines.inference.infer import build_repo_training_model_bundle
from vesuvius.image_proc.intensity.normalization import normalize_robust

sc, gpath, W0, f0, f1, nb, TAG = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7]
OUT = "runs/e73"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CK = {"sA": "runs/kaggle/pcu-trace-train/sA_best.pth", "sB": "runs/kaggle/pcu-trace-train/sB_best.pth",
      "v2A": "runs/modal_v2/v2A_best.pth"}
nets = {k: build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev).eval()
        for k, p in CK.items()}
G = np.load(gpath)
S = data.SCROLLS[sc]
vol = data.open_array(S["vol"] + "/0"); surf = data.open_array(S["surf"] + "/0")
umb = (json.load(open(os.environ["UMB_FILE"])) if os.environ.get("UMB_FILE") else data.umbilicus_for(sc))["control_points"]; cz = np.array([p["z"] for p in umb])


def predict(net, v, P=128, Z=17, stride=64):
    H, Wd = v.shape[1:]
    out = np.zeros((H, Wd), np.float32); wt = np.zeros((H, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    co = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for k in range(0, len(co), 64):
            b = co[k:k + 64]
            x = np.stack([v[:, r:r + P, q:q + P] for r, q in b]).astype(np.float32)
            keep = (x[:, Z // 2] > 0).reshape(len(b), -1).mean(1) > 0.3
            if not keep.any():
                continue
            x = np.stack([normalize_robust(p) for p in x]).astype(np.float32)
            pr = torch.sigmoid(net(torch.from_numpy(x)[:, None].to(dev)).float())[:, 0].cpu().numpy()
            for (r, q), pp, kk in zip(b, pr, keep):
                if kk:
                    out[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
    return np.where(wt > 0, out / np.maximum(wt, 1e-6), 0)


def best_window(p, v, px_mm=0.009362):
    win = int(20 / px_mm); b = 0.0
    for r0 in range(0, max(p.shape[0] - win, 0) + 1, win // 4):
        for c0 in range(0, max(p.shape[1] - win, 0) + 1, win // 4):
            vv = v[r0:r0 + win, c0:c0 + win]
            if vv.size and vv.mean() > 0.6:
                b = max(b, float(np.percentile(p[r0:r0 + win, c0:c0 + win][vv], 95)))
    return round(b, 3)


res = {}
for w in [W0] + [x for x in nb.split(",") if x]:
    g = G[w]; n = g.shape[2]
    gc = g[:, :, max(0, int((f0 - 0.04) * n)):min(n, int((f1 + 0.04) * n) + 1)].copy()
    zz = gc[2][gc[2] > 0]; i = int(np.argmin(abs(cz - np.median(zz))))
    sv, fg, st = survey.winding_volume(gc, vol, surf, (umb[i]["x"], umb[i]["y"]), layers=41, px=1.0)
    c = sv.shape[0] // 2; valid = sv[c] > 0
    r = {"shape": list(sv.shape)}
    offs = range(-8, 9, 2) if w == W0 else (0,)
    for off in offs:
        v = sv[c + off - 8:c + off + 9]
        for k, net in nets.items():
            p = predict(net, v)
            r[f"{k}_off{off}"] = best_window(p, valid)
            if off == 0:
                r[f"{k}_p95"] = round(float(np.percentile(p[valid], 95)), 3) if valid.any() else None
                if k == "sA":
                    imageio.imwrite(f"{OUT}/{TAG}_{w}_sA.png", (p * 255).astype(np.uint8))
    imageio.imwrite(f"{OUT}/{TAG}_{w}_mid.jpg", sv[c])
    if w == W0:
        np.save(f"{OUT}/{TAG}_{w}_sv41.npy", sv)
    res[w] = r
    print(TAG, w, json.dumps(r), flush=True)
json.dump(res, open(f"{OUT}/{TAG}.json", "w"), indent=1)
