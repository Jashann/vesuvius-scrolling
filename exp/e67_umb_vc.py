"""E67: team umbilicus objective (pcu.umbilicus.slice_centre_vc) on Lasagna normals, validated against the
published umbilicus. Usage: e67_umb_vc.py SCROLL [N_SLICES] [OUT.json]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from pcu import data, umbilicus

sc = sys.argv[1]; ns = int(sys.argv[2]) if len(sys.argv) > 2 else 9
Z = data.lasagna_for(sc, "nx", 2).shape[0] * 4
pts = umbilicus.estimate(sc, np.linspace(0.08 * Z, 0.92 * Z, ns).astype(int))
pub = data.SCROLLS.get(sc, {}).get("umb")
errs = []
for p in pts:
    r = dict(z=int(p["z"]), x=round(p["x"]), y=round(p["y"]))
    if pub:
        u = data.umbilicus_for(sc)["control_points"]; cz = np.array([q["z"] for q in u])
        q = u[int(np.argmin(abs(cz - p["z"])))]
        r["err"] = round(float(np.hypot(p["x"] - q["x"], p["y"] - q["y"]))); errs.append(r["err"])
    print(r, flush=True)
if errs:
    print(sc, "median err px", np.median(errs), "max", max(errs))
if len(sys.argv) > 3:
    json.dump({"control_points": [dict(x=round(p["x"]), y=round(p["y"]), z=int(p["z"]), score=100) for p in pts]},
              open(sys.argv[3], "w"), indent=1)
