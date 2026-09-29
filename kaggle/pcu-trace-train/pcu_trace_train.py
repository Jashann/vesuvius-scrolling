# TRACE [A]: native coarse distillation. Fine-tune ink_9um (hecate is too memory-hungry to train on a T4) on the team's REAL 9.362 um surface volumes
# of PHerc0139/0814 (the eligible-scroll protocol), targets = the team's 2.4 um ink predictions on the same
# mesh canvas. Held out: PHerc0841 (labels) and PHerc0139 w043 (2.4 um ink map). Two students, one per GPU.
import os, sys, json, time, glob, subprocess, re, io
T0 = time.time()
W = "/kaggle/working"
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
HF = "https://huggingface.co/scrollprize"
D = os.environ.get("TRACE_DIR", "/tmp/trace")
LOCAL = os.environ.get("TRACE_LOCAL") == "1"
ROUGH = os.environ.get("TRACE_ROUGH") == "1"  # TRACE-R: train/eval on rough (depth-warped, rotated) surfaces


def sh(c):
    print(f"\n$ {c}  [{time.time()-T0:.0f}s]", flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True, errors="replace")
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    return r.returncode


def prepare():
    sh("pip install -q 'zarr==2.18.7' 'numcodecs==0.15.1' tifffile imagecodecs opencv-python-headless pynrrd s3fs cachetools edt pyyaml")
    os.makedirs(D, exist_ok=True)
    for url, dst in ((f"{HF}/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth", f"{D}/ink9.pth"),):
        if not os.path.exists(dst):
            sh(f"curl -sL --retry 5 -o '{dst}' {url}")
    import numpy as np, requests, tifffile, cv2
    from concurrent.futures import ThreadPoolExecutor
    man = json.load(open(os.environ.get("TRACE_MANIFEST") or glob.glob("/kaggle/input/**/trace_manifest.json", recursive=True)[0]))
    ntr = int(os.environ.get("TRACE_NTRAIN", 32))  # disk: ~1 GB per segment after layer crop
    idx = np.unique(np.linspace(0, len(man["train"]) - 1, min(ntr, len(man["train"]))).round().astype(int))
    man["train"] = [man["train"][i] for i in idx]
    if os.environ.get("TRACE_PILOT"):
        man["test"] = [e for e in man["test"] if "w043" in e["seg"]]
    sess = requests.Session()

    def get_sv(e, tag):
        """Coarse surface volume level 0 (uncompressed zarr v2 chunks) -> uint8 .npy memmap."""
        out = f"{D}/{tag}_sv.npy"
        if os.path.exists(out):
            return out
        meta = sess.get(f"{S3}/{e['sv']}/0/.zarray", timeout=120).json()
        L0, H, Wd = meta["shape"]; cl, ch, cw = meta["chunks"]
        KEEP = int(os.environ.get("TRACE_KEEP", 22))  # central layers only (disk): 17 + depth jitter (+ warp for TRACE-R)
        zs = max(0, L0 // 2 - KEEP // 2); L = min(KEEP, L0 - zs)
        arr = np.lib.format.open_memmap(out + ".part", mode="w+", dtype=np.uint8, shape=(L, H, Wd))
        jobs = [(i, j) for i in range((H + ch - 1) // ch) for j in range((Wd + cw - 1) // cw)]

        def one(ij):
            i, j = ij
            for _ in range(4):
                try:
                    r = sess.get(f"{S3}/{e['sv']}/0/0/{i}/{j}", timeout=120)
                    if r.status_code == 404:
                        return
                    b = np.frombuffer(r.content, np.uint8).reshape(cl, ch, cw)
                    h = min(ch, H - i * ch); w = min(cw, Wd - j * cw)
                    arr[:, i * ch:i * ch + h, j * cw:j * cw + w] = b[zs:zs + L, :h, :w]
                    return
                except Exception:
                    time.sleep(2)
        with ThreadPoolExecutor(32) as ex:
            list(ex.map(one, jobs))
        arr.flush(); del arr
        os.replace(out + ".part", out)
        return out

    def get_target(e, tag, shape):
        out = f"{D}/{tag}_tgt.npy"
        if os.path.exists(out):
            return out
        t = tifffile.imread(io.BytesIO(sess.get(f"{S3}/{e['ink24']}", timeout=600).content))
        t = cv2.resize(t, (shape[2], shape[1]), interpolation=cv2.INTER_AREA)
        np.save(out, t.astype(np.uint8))
        return out

    def get_labels(e, tag, shape):
        sys.path.insert(0, CODE)
        from pcu import zarr3
        out = f"{D}/{tag}_lab.npy"
        if os.path.exists(out) or not e["labels"]:
            return out if os.path.exists(out) else None
        base = f"{S3}/{e['labels'][0]}"
        lab = zarr3.read_2d(f"{base}/inklabels.zarr/0")
        m = None
        for name in ("validation", "supervision"):
            try:
                m = zarr3.read_2d(f"{base}/{name}.zarr/0"); break
            except Exception:
                pass
        lab = cv2.resize(lab, (shape[2], shape[1]), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (shape[2], shape[1]), interpolation=cv2.INTER_NEAREST) if m is not None else np.full(lab.shape, 255, np.uint8)
        np.save(out, np.stack([lab, m]))
        return out

    entries = [(e, f"tr{i:02d}") for i, e in enumerate(man["train"])] + [(e, f"te{i}") for i, e in enumerate(man["test"])]

    def prep(et):
        e, tag = et
        try:
            p = get_sv(e, tag)
            mpx = os.environ.get("TRACE_MATCH_PX")
            if mpx and "8.64um" in e["sv"] and not os.path.exists(p + ".rs"):
                from scipy import ndimage as ndi
                f = 8.64 / float(mpx)
                a = np.load(p, mmap_mode="r"); b = ndi.zoom(np.asarray(a), (1, f, f), order=1).astype(np.uint8)
                del a; np.save(p, b); open(p + ".rs", "w").write(mpx)
            shape = np.load(p, mmap_mode="r").shape
            get_target(e, tag, shape)
            if tag.startswith("te"):
                get_labels(e, tag, shape)
            return tag, shape
        except Exception as ex:
            return tag, repr(ex)[:200]
    with ThreadPoolExecutor(4) as ex:
        for tag, res in ex.map(prep, entries):
            du = subprocess.run(f"du -sm '{D}'", shell=True, capture_output=True, text=True).stdout.split()[0]
            print("prepared", tag, res, f"disk {du} MB [{time.time()-T0:.0f}s]", flush=True)
    sh(f"df -h /tmp | tail -1; du -sh '{D}'")


def load_ink9(dev):
    import torch
    sys.path[:0] = [f"{CODE}/ink", f"{CODE}/vesuvius_src"]
    from koine_machines.inference.infer import build_repo_training_model_bundle
    payload = torch.load(f"{D}/ink9.pth", map_location="cpu", weights_only=False)
    cm = build_repo_training_model_bundle(payload, f"{D}/ink9.pth")
    init = os.environ.get("TRACE_INIT")
    if init:  # warm start from an earlier TRACE student
        sd = torch.load(init, map_location="cpu", weights_only=False)["model"]
        keys = set(cm.model.state_dict().keys())
        if not (set(sd) & keys):  # our saves strip a "model." prefix that the wrapper expects back
            sd = {("model." + k if ("model." + k) in keys else k): v for k, v in sd.items()}
        missing = cm.model.load_state_dict(sd, strict=False)
        print("init from", init, "missing", len(missing.missing_keys), "unexpected", len(missing.unexpected_keys), flush=True)
    return cm.model.to(dev), payload


def smooth_field(rng, shape, amp, sig, tilt):
    """random smooth depth displacement (voxels): gaussian-filtered noise scaled to max |d| = amp, plus a plane"""
    import numpy as np
    from scipy import ndimage as ndi
    d = ndi.gaussian_filter(rng.normal(size=shape).astype(np.float32), sig, mode="wrap")
    d *= amp / (np.abs(d).max() + 1e-6)
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    a, b = rng.uniform(-tilt, tilt, 2)
    d += a * (yy - shape[0] / 2) + b * (xx - shape[1] / 2)
    return d


def warp_depth(v, d, z0, Z):
    """v: (L, H, W); sample layers z0 + k + d(u, v) (linear in depth, clamped): the view of a surface that
    deviates from the true sheet by d. Returns (Z, H, W) float32."""
    import numpy as np
    L = v.shape[0]
    zz = z0 + np.arange(Z, dtype=np.float32)[:, None, None] + d[None]
    zz = np.clip(zz, 0, L - 1.001); i0 = np.floor(zz).astype(np.int64); f = zz - i0
    v = np.asarray(v, dtype=np.float32)
    a = np.take_along_axis(v, i0, 0); b = np.take_along_axis(v, i0 + 1, 0)
    return a * (1 - f) + b * f


def norm_patches(x):
    """x: (B, Z, Y, X) float -> per-patch robust normalization as in the team's inference (normalize_robust)."""
    from vesuvius.image_proc.intensity.normalization import normalize_robust
    import numpy as np
    return np.stack([normalize_robust(p) for p in x]).astype(np.float32)


def auc(p, y):
    import numpy as np
    a = p[y == 1]; b = p[y == 0]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.concatenate([a, b]); rk = x.argsort().argsort().astype(np.float64)
    return float((rk[:len(a)].sum() - len(a) * (len(a) - 1) / 2) / (len(a) * len(b)))


def predict(model, sv, flip, dev, stride=64, P=128, Z=17):
    """2D ink probability over a whole coarse surface volume with ink_9um (native 9.362 um)."""
    import numpy as np, torch
    from contextlib import nullcontext
    c = sv.shape[0] // 2; z0 = c - Z // 2
    H, Wd = sv.shape[1:]
    out = np.zeros((H, Wd), np.float32); wt = np.zeros((H, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    coords = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    model.eval()
    with torch.no_grad(), (nullcontext() if LOCAL else torch.autocast("cuda", dtype=torch.float16)):
        for k in range(0, len(coords), 32):
            batch = coords[k:k + 32]
            x = np.stack([sv[z0:z0 + Z, r:r + P, q:q + P] for r, q in batch]).astype(np.float32)
            if flip:
                x = x[:, ::-1]
            keep = x[:, Z // 2].reshape(len(batch), -1).mean(1) > 0
            x = torch.from_numpy(norm_patches(np.ascontiguousarray(x)))[:, None].to(dev)
            pr = torch.sigmoid(model(x).float())[:, 0].cpu().numpy()
            for (r, q), pp, kk in zip(batch, pr, keep):
                if kk:
                    out[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
    return np.where(wt > 0, out / np.maximum(wt, 1e-6), 0)


def evaluate(model, dev, flip, tag):
    import numpy as np
    res = {}
    for p in sorted(glob.glob(f"{D}/te*_sv.npy")):
        t = os.path.basename(p)[:3]
        sv = np.load(p, mmap_mode="r")
        tgt = np.load(f"{D}/{t}_tgt.npy")
        crop = os.environ.get("TRACE_EVAL_CROP")
        if crop:
            a, b, c, d = map(int, crop.split(","))
            sv = sv[:, a:b, c:d]; tgt = tgt[a:b, c:d]
        pr = predict(model, sv, flip, dev)
        valid = np.asarray(sv[sv.shape[0] // 2]) > 0
        r = {}
        y = np.full(tgt.shape, -1, np.int8); y[valid & (tgt > 200)] = 1; y[valid & (tgt < 60)] = 0
        r["auc_map"] = auc(pr, y)
        if os.path.exists(f"{D}/{t}_lab.npy"):
            lab, m = np.load(f"{D}/{t}_lab.npy")
            y2 = np.full(lab.shape, -1, np.int8); sel = valid & (m > 0)
            y2[sel & (lab > 127)] = 1; y2[sel & (lab <= 127)] = 0
            r["auc_lab"] = auc(pr, y2)
        if ROUGH:  # the same held-out segment seen through a surface that deviates +-4 vox from the sheet
            rg = np.random.default_rng(7)
            dfield = smooth_field(rg, sv.shape[1:], 4.0, 40.0, 0.0)
            c0 = sv.shape[0] // 2 - 8
            svr = warp_depth(sv, dfield, c0, 17)
            prr = predict(model, svr, flip, dev)
            r["auc_map_rough"] = auc(prr, y)
            if os.path.exists(f"{D}/{t}_lab.npy"):
                r["auc_lab_rough"] = auc(prr, y2)
        res[t] = r
    print(tag, json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in res.items()}), flush=True)
    return res


def train(gpu, name, lr, steps, aug):
    import numpy as np, torch, torch.nn.functional as F
    from contextlib import nullcontext
    dev = "mps" if LOCAL else f"cuda:{gpu}"
    if not LOCAL:
        torch.cuda.set_device(gpu)
    base, payload = load_ink9(dev)
    r0 = evaluate(base, dev, False, f"{name} base fwd"); r1 = evaluate(base, dev, True, f"{name} base rev")
    m0 = np.nanmean([v["auc_map"] for v in r0.values()]); m1 = np.nanmean([v["auc_map"] for v in r1.values()])
    flip = bool(m1 > m0)
    print(name, "orientation flip =", flip, flush=True)
    model = base.requires_grad_(True).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=not LOCAL)
    P, Z = 128, 17
    svs = [(np.load(p, mmap_mode="r"), np.load(p.replace("_sv.npy", "_tgt.npy"), mmap_mode="r"))
           for p in sorted(glob.glob(f"{D}/tr*_sv.npy")) if os.path.exists(p.replace("_sv.npy", "_tgt.npy"))]
    print(name, "training segments", len(svs), flush=True)
    rng = np.random.default_rng(gpu)
    best = -1; bs = int(os.environ.get("TRACE_BS", 16)); ev = int(os.environ.get("TRACE_EVAL", 1500))
    for it in range(steps):
        xs, ys = [], []
        while len(xs) < bs:
            sv, tg = svs[rng.integers(len(svs))]
            H, Wd = sv.shape[1:]
            if ROUGH and rng.random() < 0.8:
                from scipy import ndimage as ndi
                Pw = 182  # room for a rotation, then crop the centre P x P
                if H <= Pw or Wd <= Pw:
                    continue
                r = rng.integers(0, H - Pw); q = rng.integers(0, Wd - Pw)
                blk = np.asarray(sv[:, r:r + Pw, q:q + Pw])
                if (blk[blk.shape[0] // 2] > 0).mean() < 0.8:
                    continue
                dfield = smooth_field(rng, (Pw, Pw), rng.uniform(0, 5), rng.uniform(15, 60), rng.uniform(0, 0.03))
                z0 = blk.shape[0] // 2 - Z // 2 + int(rng.integers(-2, 3))
                x = warp_depth(blk, dfield, z0, Z)
                y = np.asarray(tg[r:r + Pw, q:q + Pw], dtype=np.float32) / 255
                ang = rng.uniform(-180, 180)
                x = ndi.rotate(x, ang, axes=(1, 2), reshape=False, order=1)
                y = ndi.rotate(y, ang, reshape=False, order=1)
                o = (Pw - P) // 2
                x = x[:, o:o + P, o:o + P]; y = y[o:o + P, o:o + P]
            else:
                r = rng.integers(0, H - P); q = rng.integers(0, Wd - P)
                z0 = sv.shape[0] // 2 - Z // 2 + (int(rng.integers(-2, 3)) if aug else 0)
                if z0 < 0 or z0 + Z > sv.shape[0]:
                    continue
                x = np.asarray(sv[z0:z0 + Z, r:r + P, q:q + P], dtype=np.float32)
                y = np.asarray(tg[r:r + P, q:q + P], dtype=np.float32) / 255
            if (x[Z // 2] > 0).mean() < 0.8:
                continue
            if y.max() < 0.3 and rng.random() < 0.6:  # keep more inked patches
                continue
            if flip:
                x = x[::-1]
            if aug:
                if rng.random() < 0.5:
                    x = x[:, :, ::-1]; y = y[:, ::-1]
                if rng.random() < 0.5:
                    x = x[:, ::-1]; y = y[::-1]
                g = rng.uniform(0.8, 1.25); x = 255 * (np.clip(x, 0, 255) / 255) ** g
                if rng.random() < 0.3:
                    from scipy import ndimage as ndi
                    x = ndi.gaussian_filter(x, (0, rng.uniform(0.3, 0.9), rng.uniform(0.3, 0.9)))
                x = x + rng.normal(0, rng.uniform(0, 4), x.shape).astype(np.float32)
            xs.append(np.ascontiguousarray(x)); ys.append(np.ascontiguousarray(y))
        x = torch.from_numpy(norm_patches(np.stack(xs)))[:, None].to(dev)
        y = torch.from_numpy(np.stack(ys))[:, None].to(dev)
        with (nullcontext() if LOCAL else torch.autocast("cuda", dtype=torch.float16)):
            logit = model(x)
        loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if it % 200 == 0:
            print(name, it, round(float(loss.detach()), 4), f"[{time.time()-T0:.0f}s]", flush=True)
        if (it + 1) % ev == 0 or it + 1 == steps:
            res = evaluate(model, dev, flip, f"{name} step {it+1}")
            score = np.nanmean([v.get("auc_lab", v["auc_map"]) for v in res.values()])
            if ROUGH:  # select for both clean and rough-surface performance
                score = 0.5 * score + 0.5 * np.nanmean([v.get("auc_lab_rough", v.get("auc_map_rough", np.nan)) for v in res.values()])
            if score > best:
                best = score
                out = dict(payload); out["model"] = {k.replace("model.", "", 1): v.detach().cpu() for k, v in model.state_dict().items()}
                out.pop("ema_model", None)
                torch.save(out, f"{W}/{name}_best.pth")
                json.dump(dict(step=it + 1, score=best, res=res, flip=flip), open(f"{W}/{name}_best.json", "w"))
            model.train()
        if time.time() - T0 > 11 * 3600:
            break


if __name__ == "__main__":
    CODE = os.environ.get("TRACE_CODE") or os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/pcu/__init__.py", recursive=True)[0]))
    W = os.environ.get("TRACE_OUT", W)
    if len(sys.argv) == 1:
        prepare()
        procs = [subprocess.Popen([sys.executable, __file__, "0", "sA", "2e-5", "12000", "1"]),
                 subprocess.Popen([sys.executable, __file__, "1", "sB", "6e-5", "12000", "1"])]
        for p in procs:
            p.wait()
        print("total", time.time() - T0)
    else:
        gpu, name, lr, steps, aug = sys.argv[1:6]
        train(int(gpu), name, float(lr), int(steps), aug == "1")
