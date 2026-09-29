# Progress-prize submission text (September 2026), ready to paste

Form: https://docs.google.com/forms/d/e/1FAIpQLScNBMj25FMnphngRG1Ciryv_2_Mkdq2YPJOD9WqPfZExII2iQ/viewform
Deadline: 11:59pm Pacific, 30 September 2026.

**Title:** Certified automatic winding constraints (99.6% precision) and TRACE, an ink detector trained on the eligible-protocol scans (+0.10-0.13 AUC held out)

**Team:** Jashanjot Singh Gill, Nhat Nam Tran

**Repository:** https://github.com/Jashann/vesuvius-scrolling (MIT; weights and derived data CC BY-NC-SA 4.0, attached to the GitHub release)

**Short description:**

Two tools, both aimed at the open problems for the eligible scrolls.

1. PCU, certified winding constraints from Lasagna. We treat Lasagna's `cos` channel as the real part of a wrapped winding phase (a Riesz transform supplies the quadrature) and count wraps between two points three independent ways: local phase cycles, the wrap-density integral, and a whole-slice winding field unwrapped with min-cost-flow branch cuts after removing the umbilicus vortex analytically. A step is certified only when all three agree. On PHerc. Paris 4, measured against the public winding annotations: 99.62% precision at 70% coverage on all 1,443 human adjacent-wrap pairs, 100% below 173 um wrap spacing, held-out calibration 99.64%; 377k gold and 581k silver certified pairs over 175 slices in VC3D point-collection format, loadable by the spiral fit as an extra relative-winding document. In the team's spiral fitter (fully automatic, no human windings) they cut slips per wrap from 20.3% to 9.3% on Paris 4 (held-out human ladders) and from 50.1% to 36.4% on the eligible scroll PHerc0826 (held-out gold ladders). Also a crest-graph synchronisation (2.7% ladder slips per wrap vs 5.1% for the best whole-slice unwrapping) and an annotation audit of 13 suspect human pairs. Runs at about 6 s per slice on a laptop.

2. TRACE, an ink detector for the 9.362 um / 8.64 um protocols. ink_9um fine-tuned on the team's real coarse surface volumes of PHerc0139 and PHerc0814 with the team's 2.4 um ink predictions as soft targets, then extended with the public coarse-protocol fragment pairs (PHerc0009B, PHerc0343P at 8.64 um; PHerc0500P2 at 9.362 um). Held out on PHerc0841 (human labels, three segments): AUC 0.747 / 0.736 / 0.705 for the released model vs 0.879 / 0.855 / 0.843 for TRACE-v3. It is the first public model that separates known text from blank papyrus at the eligible protocol as a whole-winding statistic (p95 0.54-0.74 vs 0.21-0.44), and a positive-control harness at the organisers' PHerc1447 ink site shows how every public detector responds there (weakly: 88th-97th percentile of the crop).

Around them: an umbilicus estimator and band method for the ten eligible scrolls without a published centre line, a fit-plus-survey job for Kaggle (T4 x2) that fitted first bands of PHerc0191, 0257, 0358, 0813 and 0800, and a hotspot vetting protocol (depth profile, neighbouring-sheet control, lattice test) that showed eight of nine text-range hotspots on eligible scrolls to be regional texture. Negative results are documented (de-Paganin, flat fit-free patches, score-guided surface warping, synthetic ink insertion, dual-energy lead detection) with the numbers.

**Open problems addressed:** winding constraints; ink detection at the eligible protocol (native coarse distillation); label quality (annotation audit); false-positive control for ink surveys.

**Links:** repository README (usage), docs/PCU_REPORT.md (constraints, full results), docs/TRACE.md (detector, data, results, negative results). Weights: GitHub release assets.
