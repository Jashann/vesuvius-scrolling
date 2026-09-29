"""Benchmark: reproduce the held-out AUC table for the released ink_9um checkpoints (seed42, seed43), KLAVIS
(domenicor046/ink9um-dense) and the TRACE checkpoints on PHerc0841 (three human-labelled segments) and on PHerc0139 w043
(human labels), from public data, in one command.

    python eval_trace.py [--models base,base43,klavis,sA,sB,v3A,v3B] [--segments te1,te2,te3,w043] [--out runs/eval] [--save_maps]

Downloads (into $PCU_CACHE) the surface volumes (native resolution, central 17 layers) and their human ink labels
from the Vesuvius Challenge open-data bucket, plus the public checkpoints and ours from the GitHub release, then
prints AUC against the human labels for each model and segment. A GPU is recommended (about 3 minutes per model
and segment on a T4); on CPU expect about an hour per segment. PHerc0841 was never used for training any TRACE
checkpoint; the TRACE checkpoints were selected on these segments (see docs/TRACE.md); ink_9um is the public baseline."""
import argparse, io, json, os, sys, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [f"{ROOT}/ext/villa-ink/ink-detection", f"{ROOT}/ext/villa/vesuvius/src"]

RELEASE = "https://github.com/Jashann/vesuvius-scrolling/releases/download/v1.0"
HF = "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth"
CK = {"base": HF, "base43": "https://huggingface.co/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed43/step-075000.pth",
      "klavis": "https://huggingface.co/domenicor046/ink9um-dense/resolve/main/dense9um-all7-step075000.pth",
      "sA": f"{RELEASE}/sA_best.pth", "sB": f"{RELEASE}/sB_best.pth", "v3A": f"{RELEASE}/v3A_best.pth", "v3B": f"{RELEASE}/v3B_best.pth"}
SEGS = {  # PHerc0841 held-out segments: surface volume (9.366 um) and human labels (drawn on the 2.403 um canvas)
    "te1": ("PHerc0841/segments/20260220213127-w00", "w00"),
    "te2": ("PHerc0841/segments/20260220214732-auto_grown_20260220144552896", "ag144"),
    "te3": ("PHerc0841/segments/20260221022814-auto_grown_20260220174252405", "ag174"),
    "w043": ("PHerc0139/segments/20260112000000-w043_2026011217", "0139w043"),  # human labels on the 2.399 um canvas
}
SV0139 = "surface-volumes/9.362um-1.2m-113keV-volume-20250728140407.zarr"
LAB0139 = "ink-labels/2.399um-volume-20260102150214/20260918"
SV = "surface-volumes/9.366um-1.2m-113keV-volume-20250821151531.zarr"
LAB = "ink-labels/2.403um-volume-20260319124803/20260918"


def get(url, dst):
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        subprocess.run(f"curl -sL --retry 5 -o '{dst}.part' '{url}' && mv '{dst}.part' '{dst}'", shell=True, check=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,base43,klavis,sA,sB,v3A,v3B"); ap.add_argument("--segments", default="te1,te2,te3,w043")
    ap.add_argument("--out", default="runs/eval"); ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--save_maps", action="store_true", help="also write downsampled probability maps and labels for figures")
    a = ap.parse_args()
    import numpy as np, torch, cv2
    from pcu import data, zarr3
    from koine_machines.inference.infer import build_repo_training_model_bundle
    from vesuvius.image_proc.intensity.normalization import normalize_robust
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.out, exist_ok=True); ckdir = os.path.join(os.environ.get("PCU_CACHE", "data/cache"), "checkpoints")
    nets = {}
    for k in a.models.split(","):
        p = get(CK[k], f"{ckdir}/{k}.pth")
        nets[k] = build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev).eval()

    def predict(net, sv, P=128, Z=17):
        H, W = sv.shape[1:]; out = np.zeros((H, W), np.float32); wt = np.zeros((H, W), np.float32)
        w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
        co = [(r, q) for r in range(0, max(H - P, 0) + 1, a.stride) for q in range(0, max(W - P, 0) + 1, a.stride)]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            for i in range(0, len(co), 64):
                b = co[i:i + 64]
                x = np.stack([sv[:, r:r + P, q:q + P] for r, q in b]).astype(np.float32)
                keep = (x[:, Z // 2] > 0).reshape(len(b), -1).mean(1) > 0
                x = np.stack([normalize_robust(p) for p in x]).astype(np.float32)
                pr = torch.sigmoid(net(torch.from_numpy(x)[:, None].to(dev)).float())[:, 0].cpu().numpy()
                for (r, q), pp, kk in zip(b, pr, keep):
                    if kk:
                        out[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
        return np.where(wt > 0, out / np.maximum(wt, 1e-6), 0)

    def auc(p, y):
        s = p[y >= 0]; t = y[y >= 0]; o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        n1 = t.sum(); n0 = len(t) - n1
        return float((r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    rows = {}
    for tag in a.segments.split(","):
        seg, name = SEGS[tag]; t0 = time.time()
        svp, labp = (SV0139, LAB0139) if tag == "w043" else (SV, LAB)
        arr = data.open_array(f"{seg}/{svp}/0"); L, H, W = arr.shape; z0 = L // 2 - 8
        sv = np.asarray(arr[z0:z0 + 17])
        lab = cv2.resize(zarr3.read_2d(f"{data.S3}/{seg}/{labp}/inklabels.zarr/0"), (W, H), interpolation=cv2.INTER_AREA)
        m = None
        for nm in ("validation", "supervision"):
            try:
                m = cv2.resize(zarr3.read_2d(f"{data.S3}/{seg}/{labp}/{nm}.zarr/0"), (W, H), interpolation=cv2.INTER_NEAREST); break
            except Exception:
                pass
        if m is None:
            m = np.full((H, W), 255, np.uint8)
        valid = (sv[8] > 0) & (m > 0)
        y = np.full((H, W), -1, np.int8); y[valid & (lab > 127)] = 1; y[valid & (lab <= 127)] = 0
        rows[name] = {}
        for k, net in nets.items():
            p = predict(net, sv); rows[name][k] = round(auc(p, y), 3)
            print(f"{name:8s} {k:6s} AUC {rows[name][k]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
            if a.save_maps:
                cv2.imwrite(f"{a.out}/{name}_{k}.png", (p[::4, ::4] * 255).astype(np.uint8))
        if a.save_maps:
            cv2.imwrite(f"{a.out}/{name}_labels.png", np.where(y[::4, ::4] == 1, 255, np.where(y[::4, ::4] == 0, 64, 0)).astype(np.uint8))
            cv2.imwrite(f"{a.out}/{name}_mid.jpg", sv[8][::4, ::4])
    print("\n| model | " + " | ".join(rows) + " |"); print("|---|" + "---|" * len(rows))
    for k in nets:
        print(f"| {k} | " + " | ".join(f"{rows[n][k]:.3f}" for n in rows) + " |")
    json.dump(rows, open(f"{a.out}/trace_heldout_auc.json", "w"), indent=1)


if __name__ == "__main__":
    main()
