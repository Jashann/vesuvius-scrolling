"""TRACE inference on a surface volume (any Vesuvius Challenge surface volume, or our renders).

    python -m pcu.trace_infer <zarr_url_or_path> <checkpoint.pth> <out.png> [--reverse] [--stride 64]

Input: a zarr array (L, H, W) uint8 with rows along z and the inner (recto) face first, at about 9.362 um per pixel
(8.64 um scans: pass --resample 8.64, which resamples in-plane only, as in training; the layer spacing is left
as scanned). The central 17 layers are used. Output: an ink probability
map (uint8 PNG, 0-255) at the input resolution. Checkpoints are in ink_9um format (build_repo_training_model_bundle).
Requires the organisers' repositories on sys.path: ext/villa-ink/ink-detection (koine_machines) and
ext/villa/vesuvius/src (vesuvius.image_proc), see README."""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [f"{ROOT}/ext/villa-ink/ink-detection", f"{ROOT}/ext/villa/vesuvius/src"]


def load_model(path, device=None):
    """Load an ink_9um-format checkpoint (released ink_9um, KLAVIS, TRACE sA/sB/v3A/v3B) as an eval model."""
    import torch
    from koine_machines.inference.infer import build_repo_training_model_bundle
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return build_repo_training_model_bundle(payload, path).model.to(device).eval(), device


def predict(model, sv, device="cpu", stride=64, P=128, Z=17, batch=64):
    """sv: (L>=17, H, W) uint8/float, inner face first. Returns (H, W) float32 probabilities (0 where no papyrus)."""
    import torch
    from vesuvius.image_proc.intensity.normalization import normalize_robust
    c = sv.shape[0] // 2; z0 = c - Z // 2
    H, W = sv.shape[1:]
    out = np.zeros((H, W), np.float32); wt = np.zeros((H, W), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    coords = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(W - P, 0) + 1, stride)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        for k in range(0, len(coords), batch):
            b = coords[k:k + batch]
            x = np.stack([sv[z0:z0 + Z, r:r + P, q:q + P] for r, q in b]).astype(np.float32)
            keep = (x[:, Z // 2] > 0).reshape(len(b), -1).mean(1) > 0  # same rule as eval_trace.py
            if not keep.any():
                continue
            x = np.stack([normalize_robust(p) for p in x]).astype(np.float32)
            pr = torch.sigmoid(model(torch.from_numpy(x)[:, None].to(device)).float())[:, 0].cpu().numpy()
            for (r, q), pp, kk in zip(b, pr, keep):
                if kk:
                    out[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
    return np.where(wt > 0, out / np.maximum(wt, 1e-6), 0)


def best_window_p95(p, valid, px_mm=0.009362, win_mm=20.0):
    """the survey triage statistic: 95th percentile of p over the best 2 x 2 cm window with >60% papyrus"""
    win = int(win_mm / px_mm); best = 0.0
    for r0 in range(0, max(p.shape[0] - win, 0) + 1, max(win // 4, 1)):
        for c0 in range(0, max(p.shape[1] - win, 0) + 1, max(win // 4, 1)):
            v = valid[r0:r0 + win, c0:c0 + win]
            if v.size and v.mean() > 0.6:
                best = max(best, float(np.percentile(p[r0:r0 + win, c0:c0 + win][v], 95)))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input"); ap.add_argument("checkpoint"); ap.add_argument("out")
    ap.add_argument("--reverse", action="store_true", help="flip the layer order (use when the volume is outer face first)")
    ap.add_argument("--resample", type=float, default=None, help="voxel size in um of the input if not 9.362 (e.g. 8.64)")
    ap.add_argument("--stride", type=int, default=64)
    a = ap.parse_args()
    import imageio.v2 as imageio
    from scipy import ndimage as ndi
    if a.input.startswith("http"):
        import re
        from pcu import data
        rel = re.sub(r"^https?://[^/]+/", "", a.input)  # path inside the open-data bucket
        arr = data.open_array(rel)
    else:
        import zarr
        arr = zarr.open(a.input, mode="r")
    L = arr.shape[0]; z0 = max(0, L // 2 - 11); sv = np.asarray(arr[z0:z0 + 22])
    if a.reverse:
        sv = sv[::-1]
    if a.resample and abs(a.resample - 9.362) > 0.05:
        f = a.resample / 9.362; sv = ndi.zoom(sv, (1, f, f), order=1)  # in-plane only, as in training
    model, dev = load_model(a.checkpoint)
    p = predict(model, sv, dev, stride=a.stride)
    imageio.imwrite(a.out, (p * 255).astype(np.uint8))
    valid = sv[sv.shape[0] // 2] > 0
    print(f"wrote {a.out} {p.shape}; p95 over papyrus {np.percentile(p[valid], 95):.3f}; best 2x2 cm window p95 {best_window_p95(p, valid):.3f}")


if __name__ == "__main__":
    main()
