"""E71: vet a TRACE hotspot. Re-render a window of a surveyed winding with 41 layers, run the detectors on
17-layer windows centred at depth offsets -8..+8, and test row periodicity. Real surface ink peaks within
+-2 layers of the sheet (E51); a neighbouring-sheet artifact peaks off-centre.
Usage: e71_candidate.py SCROLL GRID.npz COL0 COL1 TAG   (COL in full-res survey map pixels)"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ext"))
sys.path[:0] = [f"{root}/villa-ink/ink-detection", f"{root}/villa/vesuvius/src"]
import numpy as np, torch, imageio
from pcu import data, render, rowstack
from koine_machines.inference.infer import build_repo_training_model_bundle
from vesuvius.image_proc.intensity.normalization import normalize_robust

sc, gpath, c0, c1, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
OUT = "runs/e71"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CK = {"base": "data/models/ink_9um/seed42_step075000.pth", "sA": "runs/kaggle/pcu-trace-train/sA_best.pth",
      "sB": "runs/kaggle/pcu-trace-train/sB_best.pth", "v2A": "runs/modal_v2/v2A_best.pth"}
nets = {k: build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev).eval()
        for k, p in CK.items()}
fg = np.load(gpath)["grid"]
fg = fg[:, :, c0 // 5:c1 // 5 + 1]
S = data.SCROLLS[sc]
vol = data.open_array(S["vol"] + "/0")
umb = data.umbilicus_for(sc)["control_points"]; cz = np.array([p["z"] for p in umb])
zz = fg[2][fg[2] > 0]; i = int(np.argmin(abs(cz - np.median(zz))))
sv, vm = render.render(fg, vol, factor=5, layers=41, spacing=1.0, center_xy=(umb[i]["x"], umb[i]["y"]))
sv = sv[::-1].copy()
np.save(f"{OUT}/{tag}_sv41.npy", sv)
print(tag, "rendered", sv.shape, "z", int(zz.min()), int(zz.max()), flush=True)


def predict(net, v, P=128, Z=17, stride=64):
    H, Wd = v.shape[1:]
    out = np.zeros((H, Wd), np.float32); wt = np.zeros((H, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    coords = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    with torch.no_grad():
        for k in range(0, len(coords), 32):
            b = coords[k:k + 32]
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


c = sv.shape[0] // 2
valid = sv[c] > 0
res = {}
for off in range(-8, 9, 2):
    v = sv[c + off - 8:c + off + 9]
    r = {}
    for k in ("sA", "sB", "v2A", "base"):
        p = predict(nets[k], v)
        r[k] = round(float(np.percentile(p[valid], 95)), 3)
        if off == 0:
            imageio.imwrite(f"{OUT}/{tag}_{k}_off0.png", (p * 255).astype(np.uint8))
            blocks = rowstack.screen(p, valid, px_mm=0.009362, block_mm=15, min_h_mm=12)
            r[f"{k}_rows"] = [(int(a), round(float(b), 1)) for a, b in blocks]
    res[off] = r
    print(tag, "offset", off, r, flush=True)
imageio.imwrite(f"{OUT}/{tag}_mid.jpg", sv[c])
json.dump(res, open(f"{OUT}/{tag}.json", "w"), indent=1)
