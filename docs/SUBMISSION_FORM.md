# Progress-prize submission text (September 2026)

Form: https://docs.google.com/forms/d/e/1FAIpQLScNBMj25FMnphngRG1Ciryv_2_Mkdq2YPJOD9WqPfZExII2iQ/viewform (deadline 11:59pm Pacific, 30 September 2026). Submitted 2026-09-28 by Jashanjot Singh Gill.

**Correction to the submitted text.** The form quoted TRACE-v3 as 0.879 / 0.855 / 0.843 AUC on PHerc0841 and "+0.10 to 0.13 AUC" in the title; those were the best evaluation of the v3B run (step 8000), not the checkpoint that was released. The released `v3B_best.pth` (step 5000) scores **0.868 / 0.862 / 0.837** (+0.12 to 0.13 over the base model), reproducible with `python eval_trace.py`. The form also said 1,443 human adjacent pairs; the full count is **1,502** (63 slices), with the same precision (99.62% at 70% coverage, 4 errors), and "first public model that separates known text from blank papyrus" is withdrawn: the window statistic separates them on the organisers' segments but not on crushed eligible scrolls (docs/TRACE.md, section 4). The text below is the corrected version.

**Title:** Certified automatic winding constraints (99.6% precision) and TRACE, an ink detector trained on the eligible-protocol scans (+0.12 AUC held out)

**Team:** Jashanjot Singh Gill (github.com/Jashann), Nhat Nam Tran (github.com/nnm2602)

**Repository:** https://github.com/Jashann/vesuvius-scrolling (code MIT; weights and derived data CC BY-NC-SA 4.0, attached to the GitHub release)

**Short description:**

Two tools, both aimed at the open problems for the eligible scrolls.

1. PCU, certified winding constraints from Lasagna. We treat Lasagna's `cos` channel as the real part of a wrapped winding phase (a Riesz transform supplies the quadrature) and count wraps between two points three ways from that prediction: local phase cycles, the wrap-density integral, and a whole-slice winding field unwrapped with min-cost-flow branch cuts after removing the umbilicus vortex analytically. A step is certified only when all three agree; a fourth vote from the recto-surface model gives a gold tier. On PHerc. Paris 4, measured against the public winding annotations: 99.62% precision at 70% coverage on all 1,502 human adjacent-wrap pairs (base rate 95.6%; 95% lower bound 99.0%), 100% below 173 um wrap spacing, held-out calibration 99.64%; generated gold steps 942 / 942 against human ladders; 377k gold certified pairs over 175 slices in VC3D point-collection format, loadable by the spiral fit as an extra relative-winding document. In the organisers' spiral fitter (fully automatic, no human windings) they cut slips per wrap from 20.3% to 9.3% on Paris 4 (32 held-out human ladders; ladder-level bootstrap CI 6.2 to 12.7%; selection-free 10.2%) and from 50.1% to 36.3% on the eligible scroll PHerc0826 measured against PCU's own gold ladders on 8 held-out slices (agreement with PCU, not a human reference). Fit output grids are on release v1.1. Also a crest-graph synchronisation (2.7% ladder slips per wrap vs 5.1% for the best whole-slice unwrapping) and an annotation audit of 13 suspect human pairs. About 6 s per slice on a laptop.

2. TRACE, an ink detector for the 9.362 um / 8.64 um protocols. ink_9um fine-tuned on the organisers' real coarse surface volumes of PHerc0139 and PHerc0814 with their 2.4 um ink predictions as soft targets, then extended with the public coarse-protocol fragment pairs (PHerc0009B, PHerc0343P at 8.64 um; PHerc0500P2 at 9.362 um). Held out on PHerc0841 (human labels, three segments, never trained on): AUC 0.747 / 0.736 / 0.705 for the released model vs 0.868 / 0.862 / 0.837 for TRACE-v3B (checkpoints were selected on this set; disclosed). Against sparse human labels on PHerc0139 w043 the fine-tunes score below the base model (they inherit the teacher's errors), and at the organisers' PHerc1447 ink site every public detector responds only weakly, the released model most (81st to 97th percentile of the crop); a positive-control harness for that site is included.

Around them: an umbilicus estimator and band method for the eligible scrolls without a published centre line, a fit-plus-survey job for Kaggle (T4 x2) that fitted first bands of PHerc0191, 0257, 0358, 0813 and 0800, and a hotspot vetting protocol (depth profile, neighbouring-sheet control, lattice test) that showed eight of nine text-range hotspots on eligible scrolls to be regional texture. Negative results are documented with numbers (de-Paganin, flat fit-free patches, score-guided surface warping, synthetic ink insertion, dual-energy lead detection). No letters were read.

**Open problems addressed:** winding constraints; ink detection at the eligible protocol; label quality (annotation audit); false-positive control for ink surveys.

**Links:** README (results, usage, limitations), docs/PCU_REPORT.md, docs/TRACE.md, MODEL_CARD.md, eval_trace.py (one-command reproduction). Weights: GitHub release assets.
