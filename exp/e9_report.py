"""Regenerate every certificate number in docs/PCU_REPORT.md section 2 from the committed per-pair rows, in seconds,
with no bucket access:  python exp/e9_report.py [results/e9_rows.json]

Rows come from exp/e9_certificate.py (one per human adjacent-wrap pair of PHerc. Paris 4): z3 (slice), dw (true wrap
difference), loc (local phase count), den (calibrated wrap-density integral), glob (whole-slice field difference),
dres (distance to the nearest phase residue), L (pair length in L3 pixels, 19.2 um each).
Prints: single-measurement accuracies, the certificate tiers with 95% Wilson lower bounds, precision by wrap spacing,
and the 20 random half-split calibration (density scale fitted on half the slices, certificate scored on the other half)."""
import sys, json, math
import numpy as np

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/e9_rows.json"))
z3 = np.array([r["z3"] for r in rows]); dw = np.array([r["dw"] for r in rows]); loc = np.array([r["loc"] for r in rows])
den = np.array([r["den"] for r in rows]); glob = np.array([r["glob"] for r in rows]); L = np.array([r["L"] for r in rows])
K0 = 0.808  # density calibration factor baked into e9_certificate.py


def wilson_low(k, n, z=1.96):
    if n == 0: return float("nan")
    p = k / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - r) / (1 + z * z / n)


def certificate(loc, den, glob, tl=0.25, td=0.3):
    rl, rd = np.round(loc), np.round(den)
    return (rl == rd) & (rl == glob) & (np.abs(loc - rl) < tl) & (np.abs(den - rd) < td)


pred = np.round(loc)
print(f"pairs {len(rows)} on {len(set(z3))} slices; true wrap differences: {dict(zip(*np.unique(dw, return_counts=True)))}")
print(f"local phase {np.mean(pred == dw):.4f}  density {np.mean(np.round(den) == dw):.4f}  whole-slice field {np.mean(glob == dw):.4f}")
tiers = {"local = density": np.round(loc) == np.round(den),
         "local = density = field": (np.round(loc) == np.round(den)) & (np.round(loc) == glob),
         "all agree, near-integer (0.25 / 0.30)": certificate(loc, den, glob)}
print(f"\n{'rule':40s} coverage  precision  errors  certified  wilson95")
for k, m in tiers.items():
    n = int(m.sum()); e = int(np.sum(pred[m] != dw[m]))
    print(f"{k:40s} {m.mean():8.3f}  {1 - e / n:9.4f}  {e:6d}  {n:9d}  {wilson_low(n - e, n):.4f}")
c = tiers["all agree, near-integer (0.25 / 0.30)"]
print("\nby region (level-3 slice ranges of the two spiral-fit bands):")
for lo, hi, lab in ((4200, 4700, "z2 8400-9400 (fit band 1)"), (5500, 6000, "z2 11000-12000 (fit band 2)")):
    m = (z3 >= lo) & (z3 < hi); cm = c & m
    print(f"  {lab:28s} pairs {m.sum():4d}  local {np.mean(pred[m] == dw[m]):.3f}  certified {cm.sum() / max(m.sum(), 1):4.0%}  precision {np.mean(pred[cm] == dw[cm]):.4f} ({int(np.sum(pred[cm] != dw[cm]))} errors in {cm.sum()})")
print("\nby wrap spacing (pair length; L3 pixel = 19.2 um):")
for lo, hi, lab in ((0, 5, "< 96 um"), (5, 7, "96-134 um"), (7, 9, "134-173 um"), (9, 12, "173-230 um"), (12, 1e9, "> 230 um")):
    m = (L >= lo) & (L < hi); cm = c & m
    print(f"  {lab:11s} pairs {m.sum():4d}  certified {cm.sum() / max(m.sum(), 1):5.0%}  precision {np.mean(pred[cm] == dw[cm]) if cm.any() else float('nan'):.4f}")
# held-out calibration: fit the density scale on half the slices (median of dw / raw density on that half), certify the other half
rng = np.random.default_rng(0); slices = np.array(sorted(set(z3))); raw = den / K0; res = []
for s in range(20):
    perm = rng.permutation(len(slices)); tr = set(slices[perm[: len(slices) // 2]])
    intr = np.isin(z3, list(tr)); k = np.median(dw[intr] / raw[intr]); te = ~intr
    ct = certificate(loc[te], raw[te] * k, glob[te]); n = int(ct.sum()); e = int(np.sum(pred[te][ct] != dw[te][ct]))
    res.append((k, ct.mean(), 1 - e / max(n, 1)))
ks, cov, prec = np.array(res).T
print(f"\n20 half-splits by slice: density scale k = {ks.mean():.3f} +- {ks.std():.3f}; held-out coverage {cov.mean():.3f}; "
      f"held-out precision {prec.mean():.4f} (worst {prec.min():.4f})")
