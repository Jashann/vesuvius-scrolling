"""E13: certified constraint dataset over a z-band of PHerc Paris 4 (2 slices per 64-slice block)."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import ndimage as ndi
from pcu import data, slice2d, mcfcut, constraints
Y0, Y1, X0, X1 = 416, 3296, 672, 4000
z_lo, z_hi, per = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 2
os.makedirs("runs/constraints", exist_ok=True)
cos3 = data.lasagna("cos", 3); gm4 = data.lasagna("grad_mag", 4)
REC = "PHercParis4/representations/predictions/surfaces/20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr"
rec2 = data.open_array(f"{REC}/2")
OUT = os.environ.get("OUT", "runs/constraints_tiered")
os.makedirs(OUT, exist_ok=True)
t0 = time.time(); tot = 0; totg = 0
STEP = int(os.environ.get("STEP", 0))
zlist = list(range(z_lo, z_hi, STEP)) if STEP else [b * 64 + 8 + k * (64 // per) for b in range(z_lo // 64, z_hi // 64 + 1) for k in range(per)]
for z3 in zlist:
    if True:
        out = f"{OUT}/pcu_z2_{z3*2}_silver.json"
        if os.path.exists(out) or not (z_lo <= z3 < z_hi):
            continue
        c = data.read(cos3, (slice(z3, z3 + 1), slice(Y0, Y1), slice(X0, X1)))[0]
        if (c > 0).mean() < 0.05:
            continue
        ux, uy = slice2d.umbilicus_xy(z3 * 8)
        F = slice2d.winding_field(c, (ux / 8 - X0, uy / 8 - Y0), s=+1, unwrap=False)
        q = np.where(F["mask"], F["amp"] / (np.percentile(F["amp"][F["mask"]], 90) + 1e-9), 0).clip(0, 1) ** 2
        U, cut, nres, lab = mcfcut.unwrap_mcf(F["psi"], F["mask"], q)
        Wf = np.nan_to_num(U - F["theta"] / (2 * np.pi))
        g = data.read(gm4, (slice(z3 // 2, z3 // 2 + 1), slice(Y0 // 2, Y1 // 2 + 1), slice(X0 // 2, X1 // 2 + 1)))[0].astype(float) / 4000.0
        g3 = ndi.zoom(g, 2, order=1)[: c.shape[0], : c.shape[1]]
        r2 = ndi.gaussian_filter(data.read(rec2, (slice(2 * z3, 2 * z3 + 1), slice(2 * Y0, 2 * Y1), slice(2 * X0, 2 * X1)))[0].astype(float) / 255, 0.5)
        lads = constraints.generate(F["phi"], F["mx"], F["my"], F["R"], g3, Wf, F["mask"], F["amp"], seed_step=48, rmap=r2, rf=2)
        gold = constraints.gold_runs(lads)
        json.dump(constraints.to_vc3d_json([p for p, _ in lads], z3 * 2, X0, Y0), open(out, "w"))
        json.dump(constraints.to_vc3d_json(gold, z3 * 2, X0, Y0, name_prefix="pcu_gold"), open(out.replace("_silver", "_gold"), "w"))
        n = sum(len(p) - 1 for p, _ in lads); ng = sum(len(p) - 1 for p in gold); tot += n; totg += ng
        print(f"z3={z3}: {len(lads)} ladders, {n} pairs (gold {ng}); total {tot} (gold {totg}) ({time.time()-t0:.0f}s)", flush=True)
