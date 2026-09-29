# Certified automatic winding constraints from Lasagna predictions (PHerc. Paris 4)

**One-line summary.** Treating Lasagna's `cos` channel as the real part of a wrapped winding phase
lets us (1) count wraps between two points three ways from that one prediction, (2) certify the
count only when they agree, add a fourth vote from an independent model (gold tier), and (3) generate
relative-winding constraints automatically in the VC3D point-collection format the spiral fit already
consumes. On all 1,502 human adjacent-wrap pairs of PHerc. Paris 4 (63 slices), certified counts are
**99.62% correct at 70% coverage** (4 errors; base rate of the best single measurement 95.6%; 95%
Wilson lower bound 99.0%), **100% correct in the densest bands** (wrap spacing under 173 um); the
gold tier of automatically generated steps is **942 / 942** against human ladders (lower bound 99.6%),
and the precision holds on held-out calibration splits (99.64%). In the organisers' spiral fitter the
constraints cut winding slips per wrap from 20.3% to 9.3% on Paris 4 and from 50.1% to 36.4% on the
eligible scroll PHerc0826 (section 2b).

Why it matters: the organisers' stated bottleneck is "winding constraints that are precise and fast
enough to use widely" (Winding Constraints open problem), and their note on the community
winding-ruler study is that constraints at roughly 93% accuracy make the spiral fit *worse* than none
(that figure is theirs, measured differently, and is not directly comparable with the precision here).
Precision, not coverage, is what a global fit needs. These constraints run at about 6 s per slice on a
laptop, with no annotation.

Terms: a *ladder* is a human-clicked line of points across consecutive wraps in one slice; an
*adjacent pair* is two neighbouring ladder points (one wrap apart); *coverage* is the fraction of
pairs the certificate accepts; *slips per wrap* (section 2b) is the fraction of adjacent pairs the
fitted surfaces assign to the wrong wrap difference.

## 1. Idea

A scroll cross-section is a fringe pattern: the "which wrap" coordinate `w` is a phase, and Lasagna
already predicts `cos(2 pi w)`. Three measurements of "how many wraps between a and b" can be read off
the same prediction, each failing in a different way (they are not independent evidence: all three can
agree on a sheet Lasagna missed, which is why the gold tier adds a vote from a different model):

| Measurement | How | Typical failure |
|---|---|---|
| **Local phase count** | Riesz transform (monogenic signal) of `cos` gives the missing quadrature; count 2 pi cycles along the segment, propagating the normal line-field orientation (a double-cover walk, no global orientation needed) | fused sheets below sampling (count 0), split sheets (count 2) |
| **Wrap-density integral** | integrate Lasagna `grad_mag` (= abs(grad frac_pos)) along the segment; one scalar calibration | drifts over long segments |
| **Whole-slice winding field** | Laplace-oriented phase, umbilicus vortex removed analytically (Fried split), unwrapped with min-cost-flow residue matching (Costantini-style branch cuts) | jumps along sheets at the Lasagna field's own dislocations |

A pair is **certified** when all three give the same integer and the first two are within 0.25 / 0.3
of that integer.

## 2. Results (PHerc. Paris 4, 2026 scan 20260411134726, Lasagna L2 predictions)

Ground truth: `relative_windings.json` and `abs_winding.json` from the public spiral dataset
(183 single-slice ladders, 63 slices, 1,502 adjacent pairs; `exp/e9_certificate.py`, log in
`results/e9.log`, per-pair values in `results/e9_rows.json`).

**Individual measurements** (accuracy on all 1,502 adjacent pairs): local phase 95.6%, density 89.2%,
whole-slice field 94.7%.

**Certificate:**

| Rule | Coverage | Precision | Errors | 95% Wilson lower bound |
|---|---|---|---|---|
| local = density | 88.5% | 98.27% | 23 | 97.4% |
| local = density = field | 85.8% | 98.99% | 13 | 98.3% |
| **all agree, near-integer** | **70.0%** | **99.62%** | **4** | **99.0%** |
| held-out calibration (20 random half splits by slice) | 69.5% | 99.64% (worst 99.24%) | | |

Figure `docs/fig/pcu_precision_coverage.png` traces precision against coverage as the near-integer
tolerance is relaxed from 0.05 to 0.5.

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
~700-1,000 ladders and ~2,000-3,000 certified adjacent pairs. Over 175 slices of PHerc. Paris 4 (L3 z 3200-8800, two slices per 64-slice block, about 2 h on a laptop): **377,437 gold (four-vote) certified adjacent-wrap pairs** (plus 581,323 silver, three-vote, not released), merged into one relative-winding document the spiral fit loads next to its own `relative_windings.json` (`pcu_relative_windings_PHercParis4_gold.json` on the GitHub release).

## 2b. Use in the spiral fit

Setup: the organisers' `fit_spiral.py` (villa `6bbe6e2`), PHerc. Paris 4 band z2 8400-9400, 12k steps on a Kaggle T4, *fully automatic* inputs only (tracks, Lasagna, umbilicus, outer shell; no verified patches, no human windings, no winding model), plus our gold constraints loaded as unattached relative-winding strips (`input_use_pcl_relative: true`). Scored by `exp/e31_eval_fit.py` on 32 human ladders held out of every fit (about 300 adjacent pairs per run) by point-to-surface winding assignment; slips per wrap = adjacent pairs assigned the wrong wrap difference.

| Fit | Constraints | Strip weight (radius / dt) | Slips per wrap |
|---|---|---|---|
| A2 | none | | 20.3% |
| B2 | PCU gold, default sampling | 2 / 2 (fit default) | 14.6% |
| B3w4 | PCU gold, 10x sampling | 4 / 20 | 17.3% |
| **B3w10** | PCU gold, 10x sampling | **10 / 20** | **9.3%** (30 / 324; 95% CI 6.6 to 12.9%) |
| B3w20 | PCU gold, 10x sampling | 20 / 20 | 10.3% |
| B3w40 | PCU gold, 10x sampling | 40 / 20 | 21.8% |

The constraint weight is a real lever with an optimum near 10; too high and the fit follows every constraint strip at the expense of the tracks. Fits that also used the organisers' verified patches: A 15.9% without, B 15.6% with the constraints (no change: patches already pin those wraps). With human windings (not held out) the fit reaches 3.0%. Absolute-winding ladders: A2 4/7, B2 5/5, B3w10 8/8 on the modal offset. Jobs: `kaggle/pcu-fit-auto/` (A2, B2), `kaggle/pcu-fit-b3/` (B3); the fit's own satisfaction metrics are in `results/fit/`.

**Gold-ladder metric for scrolls without human ladders** (`exp/e43_eval_gold.py`): PCU gold ladders from slices held out of the fit serve as ground truth. Calibration on Paris 4 against the human-ladder metric: A2 26.7% / B2 26.1% (harsher, and it discriminates weakly, so treat eligible-scroll numbers as relative).

**PHerc0826 (eligible scroll), band z 8500-9500**, scored on 8 gold slices never given to any fit (`data/gold/gold0826_heldout8.json`, 3,426 ladders):

| Fit | Constraints | Slips per wrap | Unassigned ladder points |
|---|---|---|---|
| A0 | none | 50.1% | 2,017 |
| B0 | gold v1, default weight | 44.9% | |
| B1w10 | gold v2, 10x sampling, weight 10 | 39.0% | |
| **B1w20** | gold v2, 10x sampling, weight 20 | **36.4%** | 1,425 |
| C1 | as B1w20 plus our outer shell | 36.5% | |

A 27% relative reduction, and the surfaces cover more sheets (fewer unassigned points). The eligible-scroll fit remains far worse than Paris 4 (9 to 20%): sheet placement, not winding labels, is the remaining problem there (the fitted surfaces sit 2 to 8 voxels off the sheet, which `pcu/refine.py` corrects after the fit).

## 3. Whole-slice automatic winding fields (secondary result)

Automatic, annotation-free winding numbers for entire cross-sections, scored on the 182 ladders with
at least two points inside the unwrapped field:

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
each other (both say 2 wraps) but the label says 1 (`results/annotation_audit.json`, gallery
`results/annotation_audit.png`). In several the CT shows two distinct sheets between the clicked points (likely
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
python exp/e11_generate.py <z_L3> 48                    # certified ladders for one slice -> runs/constraints/pcu_z2_<z>.json
OUT=runs/constraints python exp/e13_band.py <z_lo> <z_hi> 2   # a z-band -> pcu_z2_<z>_gold.json and _silver.json
python exp/e9_certificate.py                             # precision / coverage on the human ladders (results/e9.log)
python exp/e12_validate_generated.py                     # independent validation of generated steps
python exp/e63_merge_gold.py out.json 'runs/constraints/*_gold.json'   # merge into one document for fit_spiral
python exp/e31_eval_fit.py ...                           # slips per wrap of a fit on held-out human ladders
python exp/e43_eval_gold.py ...                          # the same with PCU gold ladders (scrolls without human ladders)
```

Lasagna predictions are read from the organisers' bucket; `exp/e13_band.py` defaults to `OUT=runs/constraints_tiered`. The audit gallery is `results/annotation_audit.png` with `results/annotation_audit.json`.

Code: `pcu/phase.py` (Riesz phase, double-cover residues), `pcu/slice2d.py` (Laplace orientation, Fried
split), `pcu/mcfcut.py` (min-cost-flow branch cuts, cut-avoiding integration), `pcu/constraints.py`
(certificate, generator, VC3D JSON), `pcu/crestgraph.py` (section 4).

Data: Vesuvius Challenge open data. Lasagna and recto-surface models: the organisers' (scrollprize). Constraint files derived from the data: CC BY-NC-SA 4.0.
