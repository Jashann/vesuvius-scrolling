"""E58: local (MPS) ink survey of eligible scrolls with TRACE students vs the released ink_9um.
Sources: team zarr surface volumes (PHerc1447, PHerc0800; 8.64 um -> resampled to 9.362 um) and our own
renders (runs/render/*_rev.zarr, layer 0 = inner face). Saves per-model maps and the best-4cm^2 score.
Usage: e58_trace_survey.py team | render GLOB"""
import sys, os, re, json, glob, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ext"))
sys.path[:0] = [f"{root}/villa-ink/ink-detection", f"{root}/villa/vesuvius/src"]
import numpy as np, torch, requests, zarr, imageio
from scipy import ndimage as ndi
from concurrent.futures import ThreadPoolExecutor
from koine_machines.inference.infer import build_repo_training_model_bundle
from vesuvius.image_proc.intensity.normalization import normalize_robust

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
OUT = "runs/e58"; os.makedirs(OUT, exist_ok=True)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
CK = {"base": "data/models/ink_9um/seed42_step075000.pth", "sA": "runs/kaggle/pcu-trace-train/sA_best.pth",
      "sB": "runs/kaggle/pcu-trace-train/sB_best.pth"}
nets = {k: build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev).eval()
        for k, p in CK.items()}
sess = requests.Session()


def predict(net, sv, P=128, Z=17, stride=64):
    c = sv.shape[0] // 2; z0 = c - Z // 2
    H, Wd = sv.shape[1:]
    out = np.zeros((H, Wd), np.float32); wt = np.zeros((H, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    coords = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    with torch.no_grad():
        for k in range(0, len(coords), 32):
            batch = coords[k:k + 32]
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


def best4(m, valid, px_mm=0.009362):
    win = int(20 / px_mm); st = win // 4; best = 0.0
    for r0 in range(0, max(m.shape[0] - win, 0) + 1, st):
        for c0 in range(0, max(m.shape[1] - win, 0) + 1, st):
            vv = valid[r0:r0 + win, c0:c0 + win]
            if vv.size and vv.mean() >= 0.7:
                best = max(best, float(m[r0:r0 + win, c0:c0 + win][vv].mean()))
    return best


def run(tag, sv):
    valid = sv[sv.shape[0] // 2] > 0
    r = dict(tag=tag, shape=list(sv.shape), area_cm2=round(float(valid.sum()) * 0.009362 ** 2 / 100, 2))
    imageio.imwrite(f"{OUT}/{tag}_mid.jpg", sv[sv.shape[0] // 2])
    for k, net in nets.items():
        p = predict(net, sv)
        imageio.imwrite(f"{OUT}/{tag}_{k}.png", (p * 255).astype(np.uint8))
        r[f"{k}_hi"] = round(float((p[valid] > 0.8).mean()), 4)
        r[f"{k}_best4"] = round(best4((p > 0.8).astype(np.float32), valid), 4)
    print(json.dumps(r), flush=True)
    with open(f"{OUT}/results.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")


def ls(prefix):
    t = sess.get(f"{S3}/?list-type=2&delimiter=/&prefix={prefix}&max-keys=1000", timeout=120).text
    return re.findall(r"<Prefix>([^<]*)</Prefix>", t)[1:]


def load_sv(path, keep=24):
    meta = sess.get(f"{S3}/{path}/0/.zarray", timeout=120).json()
    L0, H, Wd = meta["shape"]; cl, ch, cw = meta["chunks"]
    zs = max(0, L0 // 2 - keep // 2); L = min(keep, L0 - zs)
    arr = np.zeros((L, H, Wd), np.uint8)

    def one(ij):
        i, j = ij
        for _ in range(4):
            try:
                r = sess.get(f"{S3}/{path}/0/0/{i}/{j}", timeout=120)
                if r.status_code != 200:
                    return
                b = np.frombuffer(r.content, np.uint8).reshape(cl, ch, cw)
                h = min(ch, H - i * ch); w = min(cw, Wd - j * cw)
                arr[:, i * ch:i * ch + h, j * cw:j * cw + w] = b[zs:zs + L, :h, :w]
                return
            except Exception:
                time.sleep(2)
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(one, [(i, j) for i in range((H + ch - 1) // ch) for j in range((Wd + cw - 1) // cw)]))
    return arr


done = set()
if os.path.exists(f"{OUT}/results.jsonl"):
    done = {json.loads(l)["tag"] for l in open(f"{OUT}/results.jsonl")}
if sys.argv[1] == "team":
    for sc in ("PHerc1447", "PHerc0800"):
        for seg in ls(f"{sc}/segments/"):
            if seg.endswith("raw/"):
                continue
            tag = f"{sc}_{seg.split('/')[2][-12:]}"
            if tag in done:
                continue
            svp = [p.rstrip("/") for p in ls(f"{seg}surface-volumes/") if p.rstrip("/").endswith(".zarr")]
            if not svp:
                continue
            sv = load_sv(svp[0])
            f = 8.64 / 9.362
            run(tag, ndi.zoom(sv, (f, f, f), order=1))
else:
    for p in sorted(glob.glob(sys.argv[2])):
        tag = os.path.basename(p).replace("_rev.zarr", "").replace(".zarr", "")
        if tag in done:
            continue
        sv = zarr.open(p)[:]
        if sv.shape[0] < 17:
            continue
        if "9.6" in os.environ.get("PX", ""):  # our eligible renders are at 9.6 um: resample to 9.362
            f = 9.6 / 9.362
            sv = ndi.zoom(sv, (f, f, f), order=1)
        run(tag, sv)
