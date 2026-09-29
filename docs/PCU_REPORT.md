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
constraints cut winding slips per wrap from 20.3% to 9.3% on Paris 4 (held-out human ladders, ladder-level
bootstrap CI 6.2 to 12.7%, selection-free 10.2%) and from 50.1% to 36.3% on the eligible scroll PHerc0826
measured against PCU's own gold ladders on held-out slices (section 2b).

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
| held-out calibration (20 random half splits by slice: density scale fitted on half the slices, certificate scored on the other half) | 70.4% | 99.70% (worst 99.40%) | | |

Every number in this section is regenerated from `results/e9_rows.json` by `python exp/e9_report.py` (seconds, no
bucket access) and checked by `tests/test_report_numbers.py`. The half-split row is the script's procedure
(scale k = 0.804 +- 0.012 across splits); the original run of the experiment gave 99.64% (worst 99.24%) at
69.5% coverage with k = 0.810.

Figure `docs/fig/pcu_precision_coverage.png` traces precision against coverage as the near-integer
tolerance is swept from 0.05 to 0.5 (density tolerance 1.2x the local one), with the operating point
(0.25 / 0.30, used everywhere in this report) marked. The tolerances were chosen on these same pairs; the
20 half-split calibration above is the guard against that.

**What this test cannot see.** Every human adjacent pair is a true one-wrap pair (`dw` = 1 for all 1,502), so
the test measures how often the certificate says 2 (or 0) where the truth is 1. It cannot measure the error
that hurts a spiral fit most, certifying "1 wrap" across a sheet Lasagna missed, because no two-wrap pairs are
in it (a constant "1" predictor would score 100% here, which is why the 95.6% base rate above is only a
partial yardstick). `exp/e9b_negative.py` runs the same certificate on the (i, i+2) and (i, i+3) pairs of the
same ladders and reports how many are accepted and, of those, how many are certified as 1; its output is
`results/e9b.log` (added when the run completes). The one measurement of this failure so far is on
PHerc0139 (section 5): 5 to 8% of truly two-wrap pairs certified as one.

By wrap spacing (the dense regions are where constraints are needed most):

| Spacing | Pairs | Certified | Precision |
|---|---|---|---|
| < 96 um | 61 | 33% | 100% |
| 96-134 um | 230 | 70% | 100% |
| 134-173 um | 238 | 71% | 100% |
| 173-230 um | 353 | 75% | 99.6% |
| > 230 um | 620 | 70% | 99.3% |

**Generated constraints validated independently.** The generator seeds on sheet crests and steps
outward one wrap at a time, keeping a step only if it is certified. Steps whose endpoints fall within
2.5 px of human ladder points: certified 1,581/1,587 = **99.62%** correct; steps that failed the
certificate 128/148 = 86.5% (the certificate is doing the work).

**Fourth, independent vote (gold tier).** The three measurements above all derive from Lasagna, so they
can agree on the same mistake (a consistently missed sheet). A fourth vote from a different model, the
recto-surface nnU-Net, counts recto surfaces on the step (both ends shifted 1.5 L3 px along the step,
since the recto surface sits 0 to 1 px outward of the Lasagna crest). Generated steps that pass all
four votes: **942 / 942 correct** against human ladders (recto counted at 9.6 um, the setting used for
the released files; 1,047 / 1,047 when counted at 4.8 um), 95% lower confidence bound 99.6%. The gold tier
keeps 59% of the certified steps (942 of 1,587; 66% at 4.8 um); every error of the three-vote tier falls in
the steps the fourth vote rejects. Logs: `results/e12.log` (1,581 / 1,587 and 128 / 148),
`results/e28b_recto_vote_L2.log` (942 / 942, and 639 / 645 for certified steps the recto vote rejects),
`results/e28_recto_vote_L1.log` (1,047 / 1,047). Reproduce with `RECTO_LEVEL=2 python exp/e12_validate_generated.py`
(level 1 gives the 4.8 um figures). The script seeds the generator at human ladder points to make the
matching exact, which is an easier seeding than the release generator's crest seeding.

**Throughput.** About 6 s per slice for three-vote generation (`exp/e11_generate.py`, L3, whole cross-section,
Apple M5 laptop), about 13 s per slice for the full measurement pass of `exp/e9_certificate.py`
(`results/e9.log`), and about 40 s per slice for the released gold band (min-cost-flow unwrapping plus recto
votes at two levels; 175 slices in about 2 h). One slice yields ~700-1,000 ladders and ~2,000-3,000 certified
adjacent pairs. Over 175 slices of PHerc. Paris 4 (L3 z 3200-8800, two slices per 64-slice block): **377,437
gold (four-vote) certified adjacent-wrap pairs** (plus 581,323 silver, three-vote, not released; lab-log
count), merged into one relative-winding document the spiral fit loads next to its own
`relative_windings.json` (`pcu_relative_windings_PHercParis4_gold.json` on the GitHub release).

## 2b. Use in the spiral fit

Setup: the organisers' `fit_spiral.py` (villa `6bbe6e2`), PHerc. Paris 4 band z2 8400-9400, 12k steps on a Kaggle T4, *fully automatic* inputs only (tracks, Lasagna, umbilicus, outer shell; no verified patches, no human windings, no winding model), plus our gold constraints (the z2 8400-9400 subset of the released Paris 4 file, 62,260 ladders) loaded as unattached relative-winding strips (`input_use_pcl_relative: true`). One run (seed) per setting. Scored by `exp/e31b_bootstrap.py` (same assignment as `exp/e31_eval_fit.py`: each ladder point gets the winding whose fitted surface passes closest) on 32 human ladders held out of every fit, about 300 adjacent pairs per run (pairs whose point is more than 10 voxels from every surface are unassigned and not counted). CIs are 95% ladder-level bootstraps; the paired difference resamples the same ladders for both fits. Results file: `results/fit/e31b_paris4_bootstrap.json`; the grids are on release v1.1.

| Fit | Constraints | Strip weight (radius / dt) | Slips per wrap (CI) | Paired difference vs A2 | Strips satisfied |
|---|---|---|---|---|---|
| A2 | none | | 20.3% (17.0 to 24.9), 59 / 291 | | |
| B2 | PCU gold, default sampling | 2 / 4 (fit default) | 14.6% (11.5 to 18.4), 44 / 302 | -5.7 points (-10.4 to -2.0) | 23% |
| B4w4 | PCU gold, 10x sampling | 4 / 4 (radius raised only) | 17.3% (13.9 to 21.0), 47 / 271 | -2.9 (-7.8 to +0.7) | 36% |
| **B3w10** | PCU gold, 10x sampling | **10 / 10** | **9.3%** (6.2 to 12.7), 30 / 324 | **-11.0 (-15.8 to -7.2)** | 45% |
| B4w20 | PCU gold, 10x sampling | 20 / 10 | 10.3% (5.6 to 15.7), 32 / 311 | -10.0 (-16.0 to -4.8) | 51% |
| B3w40 | PCU gold, 10x sampling | 40 / 20 | 21.8% (16.0 to 29.0), 59 / 271 | +1.5 (-4.3 to +8.0) | 48% |

Selection-free estimate (the weight sweep was scored on the same 32 ladders): choose the weight on a random half of the ladders, score the chosen fit on the other half, 200 splits: **10.2% (6.6 to 15.6) against 20.2% for A2 on the same halves**; B3w10 is chosen on 134 splits and B4w20 on 66. The constraint weight is a real lever with an optimum near 10 to 20; at 40 the fit follows the strips at the expense of the tracks. Note that even the best fit satisfies fewer than half of the strips at its own tolerance ("strips satisfied", from the fit's metrics in `results/fit/`): the constraints move the solution without being met individually. "10x sampling" is `sample_count_unattached_pcls_per_step=840` in the job scripts; the fitter divides these counts by the number of z blocks, so the logs print 88 (and 9 for the default 84). The fits are compared on slightly different subsets of the 394 held-out pairs (256 to 365), because pairs with an unassigned point are dropped; the paired bootstrap uses each fit's own assigned pairs within the resampled ladders.

Reference fits with different inputs (`kaggle/pcu-fit-ab/`, `kaggle/pcu-fit-b/`): A and B use the organisers' verified patches and no tracks; A 15.9% (10.4 to 21.4) without our constraints, B 15.6% (8.5 to 22.5) with them (no change: patches already pin those wraps). C, with the organisers' human winding annotations as input (so not held out): 3.0% (0.7 to 5.6), 11 / 365, the floor this metric can reach. Absolute-winding ladders (from `exp/e31_eval_fit.py`, lab-log figures): A2 4/7, B2 5/5, B3w10 8/8 on the modal offset. Jobs: `kaggle/pcu-fit-auto/` (A2), `kaggle/pcu-fit-b/` (B2, B), `kaggle/pcu-fit-ab/` (A, C), `kaggle/pcu-fit-b3/` (B3w10, B3w40), `kaggle/pcu-fit-b4/` (B4w4, B4w20). The scripts load `pcu_gold_band.json`, the z2 8400-9400 subset of the released Paris 4 file (62,260 ladders), shipped on release v1.1 as `pcu_gold_Paris4_z8400_9400_fit_input.json`. `results/fit/fit_logs_excerpt.txt` shows which runs loaded constraints (the first B2 attempt in `pcu-fit-auto` did not, because of a missing JSON version field, and is not used).

**Gold-ladder metric for scrolls without human ladders** (`LADDERS=<gold.json> Z0=.. Z1=.. python exp/e43_eval_gold.py <run>`, or `exp/e31b_bootstrap.py gold` with the same environment): PCU gold ladders from slices held out of the fit serve as ground truth. This measures agreement between the fit and PCU, not with a human, and the certificate's own errors are in the reference. Calibration on Paris 4 against the human-ladder metric: A2 26.7% / B2 26.1% (lab-log figures, log not shipped; harsher, and it discriminates weakly, so treat eligible-scroll numbers as relative).

**PHerc0826 (eligible scroll), band z 8500-9500**, constraints on 63 slices every 16 (`data/gold/pcu_gold_0826_fit_v2.json`), jobs `kaggle/pcu-0826/` (A0, B0), `kaggle/pcu-0826b/` (B1w10, B1w20), `kaggle/pcu-0826c/` (C0, C1), scored on 8 gold slices interleaved between them and never given to any fit (`data/gold/gold0826_heldout8.json`; 3,000 of its 3,426 ladders, random subsample, seed 0; `LADDERS=data/gold/gold0826_heldout8.json Z0=8500 Z1=9500 python exp/e31b_bootstrap.py gold ...`; `results/fit/e31b_pherc0826_bootstrap.json`):

| Fit | Constraints | Slips per wrap (CI) | Paired difference vs A0 | Unassigned ladder points | Strips satisfied |
|---|---|---|---|---|---|
| A0 | none | 50.1% (48.5 to 51.8), 2066 / 4122 | | 1,750 | |
| B0 | gold v1, default weight | 45.1% (43.4 to 46.7) | -5.1 (-6.5 to -3.6) | 1,502 | 27% |
| B1w10 | gold v2, 10x sampling, 10 / 10 | 38.7% (37.1 to 40.4) | -11.4 (-13.1 to -9.7) | 1,304 | 44% |
| **B1w20** | gold v2, 10x sampling, 20 / 10 | **36.3%** (34.8 to 37.9), 1713 / 4717 | **-13.8 (-15.5 to -12.1)** | 1,237 | 47% |
| C0 | none, plus our outer shell (`pcu/shell.py`) | 49.5% (47.9 to 51.3) | -0.6 (-1.9 to +0.8) | 1,795 | |
| C1 | as B1w20 plus our outer shell | 36.4% (34.8 to 37.9) | -13.8 (-15.4 to -12.0) | 1,243 | 47% |

Selection-free (weight chosen on half the ladders): 36.4%, since w20 wins on 199 of 200 splits. The shell alone (C0) changes nothing, so the whole reduction is the constraints. A 27% relative reduction in disagreement with PCU, and the surfaces cover more sheets (fewer unassigned points). The eligible-scroll fit remains far worse than Paris 4 (9 to 20%): sheet placement, not winding labels, is the remaining problem there (the fitted surfaces sit 2 to 8 voxels off the sheet, which `pcu/refine.py` corrects after the fit).

## 3. Whole-slice automatic winding fields (secondary result)

(The tables in sections 3, 4 and 5 are lab-log figures whose run logs are not shipped; the scripts are
`pcu/mcfcut.py`, `exp/e18_crestgraph.py`, `exp/e19_crest3d.py`, `exp/e23_longrange.py` and `exp/e26_p0139.py`,
with `results/e26_pherc0139.log` for section 5.)

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
piece. Ladder slips scored on 8 slices (4225, 4490, 5197, 5857, 6000, 6750, 7530, 7847); the verified-patch
columns on the 4 of them that have patches (so the 5.1% MCF row here is not the 182-ladder figure of
section 3):

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

- Recto-surface prediction as the phase source (L2): worse than Lasagna cos (15% vs 9% slips on an early
  25-ladder-window test, before the section 3 metric existed).
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
python exp/e9b_negative.py                               # the certificate on 2- and 3-wrap pairs (negative set)
python exp/e31_eval_fit.py <run>                         # slips per wrap of a fit on held-out human ladders
python exp/e43_eval_gold.py <run>                        # the same with PCU gold ladders (scrolls without human ladders)
python exp/e31b_bootstrap.py human A2=<dir> B3w10=<dir> --ref A2 --sweep ... --out out.json   # with ladder-level CIs
```

Lasagna predictions are read from the organisers' bucket; `exp/e13_band.py` defaults to `OUT=runs/constraints_tiered`. The audit gallery is `results/annotation_audit.png` with `results/annotation_audit.json`.

Code: `pcu/phase.py` (Riesz phase, double-cover residues), `pcu/slice2d.py` (Laplace orientation, Fried
split), `pcu/mcfcut.py` (min-cost-flow branch cuts, cut-avoiding integration), `pcu/constraints.py`
(certificate, generator, VC3D JSON), `pcu/crestgraph.py` (section 4).

Data: Vesuvius Challenge open data. Lasagna and recto-surface models: the organisers' (scrollprize). Constraint files derived from the data: CC BY-NC-SA 4.0.
