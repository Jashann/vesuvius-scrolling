"""E11: generate certified automatic ladders on one slice; timing and yield."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import ndimage as ndi
from pcu import data, slice2d, mcfcut, constraints
Y0, Y1, X0, X1 = 416, 3296, 672, 4000
z3 = int(sys.argv[1]) if len(sys.argv) > 1 else 7847
step = int(sys.argv[2]) if len(sys.argv) > 2 else 96
c = data.read(data.lasagna("cos", 3), (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0]
ux, uy = slice2d.umbilicus_xy(z3 * 8)
F = slice2d.winding_field(c, (ux / 8 - X0, uy / 8 - Y0), s=+1, unwrap=False)
q = np.where(F["mask"], F["amp"] / (np.percentile(F["amp"][F["mask"]], 90) + 1e-9), 0).clip(0, 1) ** 2
U, cut, nres, lab = mcfcut.unwrap_mcf(F["psi"], F["mask"], q)
Wf = np.nan_to_num(U - F["theta"] / (2 * np.pi))
g = data.read(data.lasagna("grad_mag", 4), (slice(z3 // 2, z3 // 2 + 1), slice(Y0 // 2, Y1 // 2 + 1), slice(X0 // 2, X1 // 2 + 1)))[0].astype(float) / 4000.0
g3 = ndi.zoom(g, 2, order=1)[: c.shape[0], : c.shape[1]]
t = time.time()
lads = constraints.generate(F["phi"], F["mx"], F["my"], F["R"], g3, Wf, F["mask"], F["amp"], seed_step=step)
n = [len(l) - 1 for l in lads]
print(f"z3={z3}: {len(lads)} ladders, {sum(n)} certified adjacent pairs, mean length {np.mean(n):.1f}, {time.time()-t:.0f}s")
os.makedirs("runs/constraints", exist_ok=True)
json.dump(constraints.to_vc3d_json(lads, z3 * 2, X0, Y0), open(f"runs/constraints/pcu_z2_{z3*2}.json", "w"))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(14, 14)); ax.imshow(c, cmap="gray")
for L in lads: ax.plot(L[:, 1], L[:, 0], "r.-", ms=2, lw=0.8)
plt.savefig(f"runs/e11_z{z3}.png", dpi=50, bbox_inches="tight")
