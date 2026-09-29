"""E69: band umbilicus for fits. Dense team-objective estimates across [z0, z1) (11 slices), robust straight
line x(z), y(z) (Theil-Sen), residual spread reported; written as control points over the band +-500.
Usage: e69_umb_band.py SCROLL z0 z1 OUT.json"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy.stats import theilslopes
from pcu import umbilicus

sc, z0, z1, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pts = umbilicus.estimate(sc, np.linspace(z0, z1, 11).astype(int))
z = np.array([p["z"] for p in pts]); x = np.array([p["x"] for p in pts]); y = np.array([p["y"] for p in pts])
fx = theilslopes(x, z); fy = theilslopes(y, z)
rx = x - (fx[1] + fx[0] * z); ry = y - (fy[1] + fy[0] * z); res = np.hypot(rx, ry)
print(sc, z0, z1, "per-slice", [(int(a), int(b), int(c)) for a, b, c in zip(z, x, y)])
print(sc, "line residual median", int(np.median(res)), "max", int(res.max()), "slope px/1000z", int(1000 * np.hypot(fx[0], fy[0])))
zz = np.arange(max(0, z0 - 500), z1 + 501, 100)
json.dump({"control_points": [dict(x=round(float(fx[1] + fx[0] * q)), y=round(float(fy[1] + fy[0] * q)), z=int(q), score=100) for q in zz],
           "metadata": {"tool": "pcu e69 band umbilicus (team objective + Theil-Sen)", "band": [z0, z1],
                        "residual_median_px": float(np.median(res))}}, open(out, "w"), indent=1)
