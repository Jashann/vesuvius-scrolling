"""E66: cheap umbilicus estimators from the masked CT at a coarse pyramid level, validated on scrolls with a
published umbilicus. Per z: mask = CT > 0 closed and hole-filled; estimators: centroid, and argmax of the
distance transform (the deepest interior point). Usage: e66_umb_mask.py SCROLL [LEVEL]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import ndimage as ndi
from pcu import data

sc = sys.argv[1]; lev = int(sys.argv[2]) if len(sys.argv) > 2 else 4
S = data.SCROLLS[sc]
a = data.open_array(f"{S['vol']}/{lev}")
f = 2 ** lev
umb = data.umbilicus_for(sc)["control_points"] if S.get("umb") else None
Z = a.shape[0]
out = []
for zi in np.linspace(0.1 * Z, 0.9 * Z, 9).astype(int):
    s = np.asarray(a[zi]).astype(np.float32)
    m = ndi.binary_fill_holes(ndi.binary_closing(s > 0, iterations=3))
    lab, n = ndi.label(m)
    if n == 0:
        continue
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    cy, cx = ndi.center_of_mass(m)
    dt = ndi.distance_transform_edt(m)
    ty, tx = np.unravel_index(np.argmax(ndi.gaussian_filter(dt, 2)), dt.shape)
    r = dict(z=int(zi * f), cen=(round(cx * f), round(cy * f)), dt=(int(tx * f), int(ty * f)))
    if umb:
        cz = np.array([p["z"] for p in umb]); i = int(np.argmin(abs(cz - zi * f)))
        ux, uy = umb[i]["x"], umb[i]["y"]
        r["pub"] = (round(ux), round(uy)); r["err_cen"] = round(float(np.hypot(cx * f - ux, cy * f - uy)))
        r["err_dt"] = round(float(np.hypot(tx * f - ux, ty * f - uy)))
    out.append(r); print(r, flush=True)
if umb:
    print(sc, "median err cen", np.median([r["err_cen"] for r in out]), "dt", np.median([r["err_dt"] for r in out]))
