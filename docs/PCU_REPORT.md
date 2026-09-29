# Certified automatic winding constraints from Lasagna predictions (PHerc. Paris 4)

**One-line summary.** Treating Lasagna's `cos` channel as the real part of a wrapped winding phase
lets us (1) count wraps between two points from three independent measurements, (2) certify the
count only when they agree, and (3) generate relative-winding constraints automatically in the VC3D
point-collection format the spiral fit already consumes. On all 1,443 human adjacent-wrap pairs of
PHerc. Paris 4, certified counts are **99.62% correct at 70% coverage**, **100% correct in the densest
bands** (wrap spacing under 173 um), and the same precision holds for automatically generated
constraints (1,581 of 1,587 matched to human ladders) and on held-out calibration splits.

Why it matters: the team's stated bottleneck is "winding constraints that are precise and fast enough
to use widely" (Winding Constraints open problem), and the community's winding-ruler study showed that
constraints at ~93% accuracy make the spiral fit *worse* than none. Precision, not coverage, is what a
global fit needs. These constraints run at ~6 s per slice on a laptop, with no annotation.

## 1. Idea

A scroll cross-section is a fringe pattern: the "which wrap" coordinate `w` is a phase, and Lasagna
already predicts `cos(2 pi w)`. Three measurements of "how many wraps between a and b" are then available,
each failing in a different way:

| Measurement | How | Typical failure |
|---|---|---|
| **Local phase count** | Riesz transform (monogenic signal) of `cos` gives the missing quadrature; count 2 pi cycles along the segment, propagating the normal line-field orientation (a double-cover walk, no global orientation needed) | fused sheets below sampling (count 0), split sheets (count 2) |
| **Wrap-density integral** | integrate Lasagna `grad_mag` (= abs(grad frac_pos)) along the segment; one scalar calibration | drifts over long segments |
| **Whole-slice winding field** | Laplace-oriented phase, umbilicus vortex removed analytically (Fried split), unwrapped with min-cost-flow residue matching (Costantini-style branch cuts) | jumps along sheets at the Lasagna field's own dislocations |

A pair is **certified** when all three give the same integer and the first two are within 0.25 / 0.3
of that integer.

## 2. Results (PHerc. Paris 4, 2026 scan 20260411134726, Lasagna L2 predictions)

Ground truth: `relative_windings.json` and `abs_winding.json` from the public spiral dataset
(183 single-slice ladders, 63 slices, 1,443 adjacent pairs).

**Individual measurements** (adjacent pairs): local phase 95.6%, density 89.2%, whole-slice field 94.7%.

**Certificate:**

| Rule | Coverage | Precision | Errors |
|---|---|---|---|
| local = density | 88.5% | 98.27% | 23 |
| local = density = field | 85.8% | 98.99% | 13 |
| **all agree, near-integer** | **70.0%** | **99.62%** | **4** |
| held-out calibration (20 random half splits by slice) | 69.5% | 99.64% (worst 99.24%) | |

By wrap spacing (the dense regions are where constraints are needed most):

| Spacing | Pairs | Certified | Precision |
|---|---|---|---|
| < 96 um | 61 | 33% | 100% |
| 96-134 um | 230 | 70% | 100% |
| 134-173 um | 238 | 71% | 100% |
| 173-230 um | 353 | 75% | 99.6% |
| > 230 um | 620 | 71% | 99.3% |

**Generated constraints validated independently.** The generator seeds on sheet crests and steps
outward one wrap at a time, keeping a step only if it is certified. Steps whose endpoints fall within
2.5 px of human ladder points: certified 1,581/1,587 = **99.62%** correct; steps that failed the
certificate 128/148 = 86.5% (the certificate is doing the work).

**Fourth, independent vote (gold tier).** The three measurements above all derive from Lasagna, so they
can agree on the same mistake (a consistently missed sheet). A fourth vote from a different model, the
recto-surface nnU-Net, counts recto surfaces on the step (both ends shifted 1.5 L3 px along the step,
since the recto surface sits 0 to 1 px outward of the Lasagna crest). Generated steps that pass all
four votes: **942 / 942 correct** against human ladders (recto counted at 9.6 um; 1,047 / 1,047 at
4.8 um), 95% lower confidence bound about 99.6%. Every error of the three-vote tier falls in the
steps the fourth vote rejects.

**Throughput.** ~6 s per slice (L3, whole cross-section) on an Apple M5 laptop; one slice yields
~700-1,000 ladders and ~2,000-3,000 certified adjacent pairs. Over 175 slices of PHerc. Paris 4 (L3 z 3200-8800, two slices per 64-slice block, about 2 h on a laptop): **377,437 gold (four-vote) and 581,323 silver (three-vote) certified adjacent-wrap pairs**, each merged into one relative-winding document the spiral fit can load next to its own `relative_windings.json` (`pcu_relative_windings_PHercParis4_gold.json.gz`, `..._silver.json.gz`).

## 3. Whole-slice automatic winding fields (secondary result)

Automatic, annotation-free winding numbers for entire cross-sections, scored on all 182 ladders:

| Unwrapper | Slips per wrap crossed | Ladders reproduced exactly |
|---|---|---|
| reliability-guided (skimage) | 7.2% | 65.9% |
| min-cost-flow branch cuts, straight cuts | 5.5% | 73.1% |
| min-cost-flow branch cuts, least-cost-path cuts | 5.1% | (90.3% of points on ladder-modal offset) |

Honest limitation: measured against verified single-wrap patches (3D, human-checked), only ~74% of
patch points keep one wrap number in the whole-slice field. The field is reliable *across* sheets but
jumps *along* sheets at the ~6,000 intrinsic dislocations per slice in the Lasagna phase (orientation-
free count; they persist across neighbouring slices, and running Lasagna locally at 2x resolution does
not remove them). The certified constraints are local and unaffected; the whole-slice field is a
diagnostic, not a replacement for the spiral fit.

## 4. Crest-graph synchronization (sheets cannot switch along themselves)

To fix the along-sheet weakness, label sheet *crest pieces* instead of pixels. Nodes are unbranched
pieces of the sheet skeleton (split at junctions, phase residues and the branch cut); edges are only
certified +1 steps; wrap labels come from one exact L1 integer synchronization (network-matrix LP,
integral optimum), after dropping pieces whose incident certified edges are mostly violated (pieces
that silently merged two sheets). Because a label is per piece, a sheet cannot change wrap along a
piece. Scored on 8 slices:

| Method | Ladder slips | Verified-patch points on one wrap | Patches with a jump |
|---|---|---|---|
| whole-slice MCF field | 5.1% | 78.0% | 52% |
| crest graph, Lasagna crest pieces | 3.9% (9/228) | 89.6% | 27% |
| crest graph, recto-surface pieces (snap 2 px) | **2.7%** (6/222) | **91.6%** | **21%** |

(Patch numbers use a branch-cut correction: a sheet crossing the theta = pi half-line steps by one
wrap by construction, so each patch is scored after moving the cut into its largest uncovered angle.
With that correction the whole-slice field scores 78.0% and Lasagna-crest pieces 89.6% on the same
four slices.)

Long-range relative windings from the crest graph (recto pieces, certified 1-3 wrap edges), human
ladder pairs on four slices: 1 wrap apart 99.4%, up to 4 apart 95.1%, up to 9 apart 88.8%, up to 19
apart 87.4%, 20+ apart 54%. A 3D version (linking pieces across neighbouring slices) did not help
yet (85-86%).

## 5. Second scroll (PHerc0139, not used during development)

Reference: the 37 published whole-wrap segments w023-w059 on the same 2.4 um scan. Scored on
nearest-point pairs between wraps w and w+1 (and w+2 as negatives), three slices, no retuning.
- Where the reference is geometrically consistent (w/w+1 4-16 px apart at L3), Lasagna's local count is
  right on 86-91% of adjacent pairs (95.6% on Paris 4), the certificate covers 53-66%, and it certifies
  5-8% of truly two-wrap pairs as one. Estimated precision of generated constraints on this scroll:
  about 99%, below Paris 4's 99.6%. The failures are places where all three Lasagna-derived
  measurements consistently miss one sheet.
- Many apparent disagreements are in the reference itself: in 73% of adjacent pairs where Lasagna sees
  no sheet between w and w+1, the two published segments are within 3 px (median 25 um) of each other,
  closer than one papyrus thickness. At z3=2200, w041 lies within 38 um of w042 along 86% of its
  cross-section, and w045 of w046 along 54% (flagged for review).

## 6. Annotation audit

13 human relative-winding pairs where the local phase count and the density integral agree tightly with
each other (both say 2 wraps) but the label says 1 (`annotation_audit.json`, gallery
`annotation_audit.png`). In several the CT shows two distinct sheets between the clicked points (likely
skipped wraps); others are single thick sheets whose plies separated. Each is worth a quick look,
because a wrong relative-winding label pulls the whole spiral fit.

## 7. Also measured (negative or neutral results, to save others time)

- Recto-surface prediction as the phase source (L2): worse than Lasagna cos (15% vs 9% slips).
- Averaging the complex phase over +-4 slices: no residue reduction (defects are structural).
- Goldstein spectral filter: -30% residues, fewer ladder slips, no gain along sheets.
- Cut costs from density agreement or recto support: no measurable gain.
- 2.5D exact reconciliation of cut-bounded regions across 9 slices: no change on ladders.
- Full-resolution PUMA graph-cut unwrapping: correct but too slow on a laptop (7M nodes).

## 8. How to run

```
python exp/e11_generate.py <z_L3> 48        # certified ladders for one slice -> runs/constraints/*.json
python exp/e13_band.py <z_lo> <z_hi> 2       # a z-band
python exp/e9_certificate.py                 # precision/coverage on human ladders
python exp/e12_validate_generated.py         # independent validation of generated steps
```

Code: `pcu/phase.py` (Riesz phase, double-cover residues), `pcu/slice2d.py` (Laplace orientation, Fried
split), `pcu/mcfcut.py` (min-cost-flow branch cuts, cut-avoiding integration), `pcu/constraints.py`
(certificate, generator, VC3D JSON), `pcu/lasagna_run.py` (local Lasagna inference on MPS).

Data: Vesuvius Challenge open data (CC BY-NC 4.0). Lasagna model: scrollprize/lasagna (MIT).
