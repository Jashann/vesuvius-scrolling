"""Block-bootstrap confidence intervals for the held-out ink AUCs, from the maps saved by `eval_trace.py --save_maps`.

    python exp/eval_ci.py runs/eval [--block 256] [--nboot 2000] [--ref base]

For each segment (files <seg>_labels.png and <seg>_<model>.png, 4x downsampled uint8 maps) the labelled pixels are
tiled into square blocks (default 256 px at 4x = 1,024 native px, about 1 cm); blocks are resampled with
replacement and the AUC recomputed, which respects the spatial correlation that a per-pixel bootstrap ignores.
Because the maps are 8-bit, each block is summarised by a 256-bin histogram of positive and negative pixels, so a
bootstrap replicate is a sum of histograms and the AUC follows from the cumulative counts (exact for 8-bit maps).
Reports each model's AUC with a 95% interval and the paired difference against --ref on the same block resamples.
Writes <dir>/trace_heldout_auc_ci.json and prints markdown tables."""
import argparse, glob, json, os
import numpy as np


def auc_from_hist(hp, hn):
    """AUC with ties counted half, from histograms of positive / negative scores over the same bins."""
    cn = np.cumsum(hn) - hn  # negatives strictly below each bin
    return float((hp * (cn + 0.5 * hn)).sum() / (hp.sum() * hn.sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--nboot", type=int, default=2000); ap.add_argument("--ref", default="base"); a = ap.parse_args()
    import imageio.v2 as io
    out = {}; rng = np.random.default_rng(0)
    for lab in sorted(glob.glob(f"{a.dir}/*_labels.png")):
        seg = os.path.basename(lab)[:-11]; L = io.imread(lab); L = L if L.ndim == 2 else L[..., 0]
        y = np.where(L == 255, 1, np.where(L == 64, 0, -1)).astype(np.int8)
        models = {}
        for f in sorted(glob.glob(f"{a.dir}/{seg}_*.png")):
            k = os.path.basename(f)[len(seg) + 1:-4]
            if k == "labels": continue
            v = io.imread(f); models[k] = (v if v.ndim == 2 else v[..., 0]).astype(np.uint8)
        H, W = y.shape; B = a.block
        blocks = [(r, c) for r in range(0, H, B) for c in range(0, W, B) if (y[r:r + B, c:c + B] == 1).any() and (y[r:r + B, c:c + B] == 0).any()]
        hist = {}
        for k, v in models.items():
            hp = np.zeros((len(blocks), 256)); hn = np.zeros((len(blocks), 256))
            for i, (r, c) in enumerate(blocks):
                yy = y[r:r + B, c:c + B]; vv = v[r:r + B, c:c + B]
                hp[i] = np.bincount(vv[yy == 1], minlength=256); hn[i] = np.bincount(vv[yy == 0], minlength=256)
            hist[k] = (hp, hn)
        full = {k: auc_from_hist(hp.sum(0), hn.sum(0)) for k, (hp, hn) in hist.items()}
        samp = {k: np.empty(a.nboot) for k in models}
        for b in range(a.nboot):
            idx = rng.integers(0, len(blocks), len(blocks))
            for k, (hp, hn) in hist.items():
                samp[k][b] = auc_from_hist(hp[idx].sum(0), hn[idx].sum(0))
        out[seg] = {}
        for k in models:
            s = samp[k]; d = {"auc": round(full[k], 3), "ci95": [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)], "blocks": len(blocks)}
            if a.ref in models and k != a.ref:
                dd = s - samp[a.ref]; d["diff_vs_ref"] = {"mean": round(full[k] - full[a.ref], 3), "ci95": [round(float(np.percentile(dd, 2.5)), 3), round(float(np.percentile(dd, 97.5)), 3)]}
            out[seg][k] = d
        print(seg, "blocks", len(blocks), flush=True)
    json.dump(out, open(f"{a.dir}/trace_heldout_auc_ci.json", "w"), indent=1)
    segs = list(out); print("\n| model | " + " | ".join(segs) + " |"); print("|---|" + "---|" * len(segs))
    for k in out[segs[0]]:
        print(f"| {k} | " + " | ".join(f"{out[s][k]['auc']:.3f} ({out[s][k]['ci95'][0]:.3f} to {out[s][k]['ci95'][1]:.3f})" if k in out[s] else "" for s in segs) + " |")
    print("\npaired difference vs", a.ref)
    for k in out[segs[0]]:
        if "diff_vs_ref" in out[segs[0]][k]:
            print(f"| {k} | " + " | ".join(f"{out[s][k]['diff_vs_ref']['mean']:+.3f} ({out[s][k]['diff_vs_ref']['ci95'][0]:+.3f} to {out[s][k]['diff_vs_ref']['ci95'][1]:+.3f})" for s in segs) + " |")


if __name__ == "__main__":
    main()
