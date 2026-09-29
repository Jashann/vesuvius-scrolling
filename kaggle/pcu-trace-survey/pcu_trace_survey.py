# TRACE survey of an eligible scroll: every fitted winding -> snap to sheet -> isometric flatten -> render at the
# native 9.362 um -> released ink_9um and TRACE students sA/sB in-process. Text-presence statistic: p95 of the
# probability (calibrated: known text 0.69/0.54 for sA/sB vs blank 0.34/0.22), plus best 2x2 cm window p95.
import os, sys, json, time, glob, subprocess
T0 = time.time()
W = os.environ.get("SURVEY_W", "/kaggle/working")
BUDGET = float(os.environ.get("BUDGET_H", "10.8")) * 3600
SC = os.environ.get("SC", "PHerc0826"); FIT = os.environ.get("FIT", "B1w20"); EVERY = int(os.environ.get("EVERY", "2"))


def sh(c):
    print(f"\n$ {c}  [{time.time()-T0:.0f}s]", flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True, errors="replace")
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)


sh("pip install -q 'zarr==2.18.7' 'numcodecs==0.15.1' fsspec aiohttp tifffile imagecodecs pynrrd s3fs cachetools edt pyyaml")
sh("curl -sL --retry 5 -o /tmp/ink9.pth https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth")
import numpy as np, torch, imageio
CODE = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/pcu/__init__.py", recursive=True)[0]))
sys.path[:0] = [CODE, f"{CODE}/ink", f"{CODE}/vesuvius_src"]
os.environ["PCU_CACHE"] = "/tmp/cache"
from pcu import data, survey
from koine_machines.inference.infer import build_repo_training_model_bundle
from vesuvius.image_proc.intensity.normalization import normalize_robust
CK = {"base": "/tmp/ink9.pth"}
for p in glob.glob("/kaggle/input/**/s[A-D]_best.pth", recursive=True) + glob.glob("/kaggle/input/**/v3[AB]_best.pth", recursive=True):
    CK[os.path.basename(p).split("_")[0]] = p
dev = "cuda"
nets = {k: build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev).eval()
        for k, p in CK.items()}
print("models", list(nets), flush=True)
GRIDS = os.environ.get("GRIDS") or glob.glob(f"/kaggle/input/**/grids_{FIT}.npz", recursive=True)[0]


def predict(net, sv, P=128, Z=17, stride=64):
    c = sv.shape[0] // 2; z0 = c - Z // 2
    H, Wd = sv.shape[1:]
    out = np.zeros((H, Wd), np.float32); wt = np.zeros((H, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    coords = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for k in range(0, len(coords), 64):
            batch = coords[k:k + 64]
            x = np.stack([sv[z0:z0 + Z, r:r + P, q:q + P] for r, q in batch]).astype(np.float32)
            keep = (x[:, Z // 2] > 0).reshape(len(batch), -1).mean(1) > 0.3
            if not keep.any():
                continue
            x = np.stack([normalize_robust(p) for p in x]).astype(np.float32)
            pr = torch.sigmoid(net(torch.from_numpy(x)[:, None].to(dev)).float())[:, 0].cpu().numpy()
            for (r, q), pp, kk in zip(batch, pr, keep):
                if kk:
                    out[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
    return np.where(wt > 0, out / np.maximum(wt, 1e-6), 0)


def best4_p95(p, v, px_mm=0.009362):
    win = int(20 / px_mm); best = 0.0
    for r0 in range(0, max(p.shape[0] - win, 0) + 1, win // 4):
        for c0 in range(0, max(p.shape[1] - win, 0) + 1, win // 4):
            w = v[r0:r0 + win, c0:c0 + win]
            if w.size and w.mean() > 0.6:
                best = max(best, float(np.percentile(p[r0:r0 + win, c0:c0 + win][w], 95)))
    return best


S = data.SCROLLS[SC]
VOX = float(S.get("px_um", 9.362)); PX = 9.362 / VOX  # render at the detector pixel (9.362 um) whatever the scan voxel
vol = data.open_array(S["vol"] + "/0"); surf = data.open_array(S["surf"] + "/0")
umb = (json.load(open(os.environ["UMB_FILE"])) if os.environ.get("UMB_FILE") else data.umbilicus_for(SC))["control_points"]
cz = np.array([p["z"] for p in umb])
G = np.load(GRIDS)
START = os.environ.get("START", "w065" if SC == "PHerc0826" else "w000")
names = [n for n in sorted(G.files)[5:-2:EVERY] if n >= START]
print(SC, FIT, "windings", len(names), flush=True)
results = []
for w in names:
    if time.time() - T0 > BUDGET:
        break
    t = time.time()
    try:
        g = G[w]; i = int(np.argmin(abs(cz - np.median(g[2][g[2] > 0]))))
        sv, fg, st = survey.winding_volume(g, vol, surf, (umb[i]["x"], umb[i]["y"]), layers=21, px=PX)
    except Exception as e:
        print("render failed", w, repr(e)[:300], flush=True); continue
    v = sv[sv.shape[0] // 2] > 0
    r = dict(w=w, shape=list(sv.shape), area_cm2=round(float(v.sum()) * 0.009362 ** 2 / 100, 2), render_s=round(time.time() - t))
    np.savez_compressed(f"{W}/{w}_grid.npz", grid=fg)
    imageio.imwrite(f"{W}/{w}_mid.jpg", sv[sv.shape[0] // 2][::2, ::2])
    for k, net in nets.items():
        p = predict(net, sv)
        imageio.imwrite(f"{W}/{w}_{k}.png", (p[::2, ::2] * 255).astype(np.uint8))
        r[f"{k}_p95"] = round(float(np.percentile(p[v], 95)), 3) if v.any() else None
        r[f"{k}_best4_p95"] = round(best4_p95(p, v), 3)
    r["secs"] = round(time.time() - t)
    subprocess.run("rm -rf /tmp/cache", shell=True)  # the CT chunk cache filled the disk (session killed at ~23 windings)
    results.append(r); print(r, flush=True)
    json.dump(results, open(f"{W}/trace_survey_{SC}_{FIT}.json", "w"), indent=1)
print("total", time.time() - T0)
