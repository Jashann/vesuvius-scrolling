"""E58b: re-score saved survey maps (runs/e58) with statistics calibrated on known text (PHerc0139 w043 at the
eligible protocol): p95 of the TRACE probability inside the valid surface (text 0.69 / 0.54 for sA / sB vs
0.34 / 0.22 on a blank PHerc1447 segment), best 2x2 cm window of p95, and the row-periodicity peak ratio."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, imageio.v3 as iio
from pcu import rowstack

rows = []
for mid in sorted(glob.glob("runs/e58/*_mid.jpg")):
    tag = os.path.basename(mid)[:-8]
    v = iio.imread(mid) > 0
    r = dict(tag=tag, area_cm2=round(float(v.sum()) * 0.009362 ** 2 / 100, 2))
    for k in ("base", "sA", "sB"):
        fn = f"runs/e58/{tag}_{k}.png"
        if not os.path.exists(fn):
            continue
        p = iio.imread(fn).astype(np.float32) / 255
        vv = v[: p.shape[0], : p.shape[1]]
        if vv.sum() < 1000:
            continue
        r[f"{k}_p95"] = round(float(np.percentile(p[vv], 95)), 3)
        win = int(20 / 0.009362); best = 0.0
        for r0 in range(0, max(p.shape[0] - win, 0) + 1, win // 4):
            for c0 in range(0, max(p.shape[1] - win, 0) + 1, win // 4):
                w = vv[r0:r0 + win, c0:c0 + win]
                if w.size and w.mean() > 0.6:
                    best = max(best, float(np.percentile(p[r0:r0 + win, c0:c0 + win][w], 95)))
        r[f"{k}_best4_p95"] = round(best, 3)
        sc = rowstack.screen(p, vv, px_mm=0.009362, min_h_mm=12)
        r[f"{k}_rows"] = round(max([s[1] for s in sc]), 1) if sc else None
    rows.append(r)
rows.sort(key=lambda r: -(r.get("sA_best4_p95", 0) + r.get("sB_best4_p95", 0)))
for r in rows:
    print(json.dumps(r))
json.dump(rows, open("runs/e58/scores.json", "w"), indent=1)
