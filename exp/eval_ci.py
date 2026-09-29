"""Block-bootstrap confidence intervals for the held-out ink AUCs, from the maps saved by `eval_trace.py --save_maps`.

    python exp/eval_ci.py runs/eval [--block 256] [--nboot 1000] [--ref base]

For each segment (files <seg>_labels.png and <seg>_<model>.png, 4x downsampled) the valid pixels are tiled into
square blocks (default 256 px at 4x = 1,024 native px, about 1 cm); blocks are resampled with replacement and the
AUC recomputed, which respects the spatial correlation that a per-pixel bootstrap ignores. Reports each model's
AUC with a 95% interval and the paired difference against --ref on the same block resamples.
Writes <dir>/trace_heldout_auc_ci.json and prints a markdown table."""
import argparse, glob, json, os
import numpy as np


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--nboot", type=int, default=1000); ap.add_argument("--ref", default="base"); a = ap.parse_args()
    import imageio.v2 as io
    out = {}; rng = np.random.default_rng(0)
    for lab in sorted(glob.glob(f"{a.dir}/*_labels.png")):
        seg = os.path.basename(lab)[:-11]; L = io.imread(lab); L = L if L.ndim == 2 else L[..., 0]
        y = np.where(L == 255, 1, np.where(L == 64, 0, -1)).astype(np.int8)
        models = {os.path.basename(f)[len(seg) + 1:-4]: io.imread(f) for f in sorted(glob.glob(f"{a.dir}/{seg}_*.png")) if not f.endswith("_labels.png")}
        models = {k: (v if v.ndim == 2 else v[..., 0]).astype(np.float32) / 255 for k, v in models.items()}
        H, W = y.shape; B = a.block
        blocks = [(r, c) for r in range(0, H, B) for c in range(0, W, B) if (y[r:r + B, c:c + B] == 1).any() and (y[r:r + B, c:c + B] == 0).any()]
        full = {k: auc(v[y >= 0], y[y >= 0]) for k, v in models.items()}
        samp = {k: [] for k in models}
        for _ in range(a.nboot):
            pick = [blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]
            yy = np.concatenate([y[r:r + B, c:c + B].ravel() for r, c in pick]); m = yy >= 0
            for k, v in models.items():
                samp[k].append(auc(np.concatenate([v[r:r + B, c:c + B].ravel() for r, c in pick])[m], yy[m]))
        out[seg] = {}
        for k in models:
            s = np.array(samp[k]); d = {"auc": round(full[k], 3), "ci95": [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)], "blocks": len(blocks)}
            if a.ref in models and k != a.ref:
                dd = s - np.array(samp[a.ref]); d["diff_vs_ref"] = {"mean": round(full[k] - full[a.ref], 3), "ci95": [round(float(np.percentile(dd, 2.5)), 3), round(float(np.percentile(dd, 97.5)), 3)]}
            out[seg][k] = d
    json.dump(out, open(f"{a.dir}/trace_heldout_auc_ci.json", "w"), indent=1)
    segs = list(out); print("| model | " + " | ".join(segs) + " |"); print("|---|" + "---|" * len(segs))
    for k in out[segs[0]]:
        cells = []
        for s in segs:
            d = out[s].get(k); cells.append(f"{d['auc']:.3f} ({d['ci95'][0]:.3f} to {d['ci95'][1]:.3f})" if d else "")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("\npaired difference vs", a.ref)
    for k in out[segs[0]]:
        if "diff_vs_ref" in out[segs[0]][k]:
            print(f"| {k} | " + " | ".join(f"{out[s][k]['diff_vs_ref']['mean']:+.3f} ({out[s][k]['diff_vs_ref']['ci95'][0]:+.3f} to {out[s][k]['diff_vs_ref']['ci95'][1]:+.3f})" for s in segs) + " |")


if __name__ == "__main__":
    main()
