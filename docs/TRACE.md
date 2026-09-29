# TRACE: ink detection at the eligible protocol

**Summary.** The released `ink_9um` model was distilled mostly from 2.4 um data downsampled 4x. Fine-tuning it on the organisers' *real* 9.362 um surface volumes, with their 2.4 um ink predictions of the same meshes as soft targets, raises AUC against human labels on a scroll never trained on (PHerc0841) from 0.705 to 0.747 (base) to 0.803 to 0.857 (v1). Adding the public fragment pairs at the coarse protocols (8.64 um PHerc0009B and PHerc0343P, 9.362 um PHerc0500P2) adds another 0.01 to 0.02 (v3B: 0.837 to 0.868), within the checkpoint-selection spread. Reproduce with `python eval_trace.py`; the run we did is in `results/trace_heldout_auc.json` and `results/eval_trace.log`.

## 1. Why

Ink at the eligible protocol (9.362 um, 1.2 m propagation, 113 keV, Paganin) is contextual surface texture, not density: on Paris 4 rendered at 9.6 um the per-pixel density separation d' between ink and papyrus is about 0.2, gradient-energy d' 0.3 to 0.5, and no hand-made texture map separates them. The signal lives in the joint structure of the layers around the sheet, which a detector trained on real coarse renders can learn and a detector trained on pooled fine data cannot.

## 2. Data

| Set | Segments | Input | Target |
|---|---|---|---|
| v1 train | 32 (PHerc0139, PHerc0814; `TRACE_NTRAIN=32` over the 56-entry manifest) | organisers' 9.362 um surface volumes, central 22 layers | organisers' 2.4 um ink prediction on the same mesh, resized to the coarse canvas |
| v2 train | 56 (same scrolls) | same | same |
| v3 train | 128: v2 + PHerc0009B (18, 8.64 um), PHerc0343P (8, 8.64 um), PHerc0500P2 (46, 9.362 um) | 8.64 um volumes resampled in-plane to 9.362 um | fragment 2.2 to 2.4 um ink predictions |
| held out | PHerc0841 (3 segments, the organisers' human ink labels, 9.366 um), PHerc0139 w043 (2.4 um map and sparse human labels; its neighbouring wraps w041, w042, w045 and w046 are in the v1 training subset and w044 too in v3, so w043 is not fully held out) | | |

PHerc0841 segments: `w00` (`20260220213127-w00`), `ag144` (`20260220214732-auto_grown_20260220144552896`), `ag174` (`20260221022814-auto_grown_20260220174252405`); `ag` is "auto-grown", the organisers' naming. Manifests: `runs/trace_manifest.json` (v1/v2), `runs/trace_manifest_v3.json` (v3, built by `exp/e93_manifest_v3.py`).

## 3. Training

`kaggle/pcu-trace-train/pcu_trace_train.py`: 17 x 128 x 128 patches, per-patch `normalize_robust` (as in the organisers' inference), depth jitter +-2 layers, flips, gamma, blur and noise, soft BCE against the resized 2.4 um map, AdamW, fp16 on a T4 (batch 16) or L4 (batch 24). Runs: sA (lr 2e-5), sB (6e-5), sC/sD (1e-4 with aug / 3e-5 without), v2A/v2B (56 segments), v3A (3e-5) / v3B (8e-5) (128 segments, `modal/trace_v3.py`).

**Checkpoint selection (disclosure).** Every 1000 to 1500 steps each run was scored on the four held-out segments against the human labels (PHerc0841 w00, ag144, ag174 and PHerc0139 w043); the checkpoint with the best mean label AUC over the four was kept (the `score` line of `pcu_trace_train.py`) and the layer order confirmed on the same segments (no run preferred the flipped order). Released steps: sA 3000, sB 4500, v3A 4000, v3B 5000 (v3B's step 8000 scored higher on PHerc0841 alone, 0.879 / 0.855 / 0.843, but lower on the four-segment mean, which is why step 5000 was kept). The held-out set therefore chose among about 8 evaluations per run and about 8 runs; the spread between the best and the last evaluation of a run is about 0.01 AUC, small next to the 0.12 gain over the base model, but PHerc0841 is "never trained on", not "never looked at". Students plateau after 1 to 5k steps at every data size tried. The v3 over v1 difference (0.01 to 0.02) is within the selection spread and is not claimed as a real improvement.

## 4. Results

AUC against human labels on PHerc0841 and on PHerc0139 w043; w043 also against the organisers' 2.4 um map (the teacher signal). Released checkpoints only.

| Model | 0841 w00 | 0841 ag144 | 0841 ag174 | w043 vs 2.4 um map | w043 vs human labels |
|---|---|---|---|---|---|
| released ink_9um (seed 42, step 75000) | 0.747 | 0.736 | 0.705 | 0.782 | 0.82 |
| TRACE sA (step 3000) | 0.853 | 0.850 | 0.803 | 0.895 | 0.789 |
| TRACE sB (step 4500) | 0.857 | 0.839 | 0.817 | 0.898 | 0.764 |
| TRACE v3A (step 4000) | 0.866 | 0.860 | 0.832 | 0.886 | 0.745 |
| TRACE v3B (step 5000) | **0.868** | **0.862** | **0.837** | 0.899 | 0.760 |

Two readings. On PHerc0841 (dense human labels, a scroll none of the models saw) every fine-tune beats the base model by 0.10 to 0.13. On w043 the fine-tunes track the teacher map (0.90) but score *below* the base model against the sparse human labels there: they inherit the teacher's errors where it disagrees with the annotators. Pixels are spatially correlated, so the per-segment AUC has no small-sample guarantee; the evidence is that three independent segments move in the same direction by a similar amount, and that the seed-42 vs seed-43 base models (both in `eval_trace.py`) give the run-to-run yardstick. How the PHerc0841 labels were drawn (over CT, or over a 2.4 um prediction) is not documented here; if over predictions, part of the gain is agreement with the teacher.

Reversed layer order scores near chance for every model, so surface volumes must be inner (recto) face first.

**Window statistic (triage only).** For surveys we use the 95th percentile of the probability over the best 2 x 2 cm window of a winding. On known text at the coarse protocol (0139 w043, Paris 4 at 9.6 um) sB gives 0.54 to 0.61 and sA 0.69 to 0.74; on blank organisers' segments of PHerc1447 and PHerc0800 sB gives 0.21 to 0.29 and sA 0.33 to 0.44; the released model gives 0.66 to 0.75 for both. On crushed eligible scrolls, however, the statistic reaches the text range without text (section 5), so it ranks windings for inspection and nothing more.

**Positive control at a real eligible-scroll ink site.** On 2026-09-24 the organisers reported letters on PHerc1447 (8.64 um) near (x 4144, y 2742, z 12557). `exp/e90_control1447.py` scores the published segment that passes 8.6 voxels from that point, for every model, over depth offsets -4, -2, 0, +2, +4 layers, both layer orders, native 8.64 um and resampled to 9.6 um (20 settings per model), and reports the rank of the site's 3 mm-disk mean within a 1.04 cm crop (a shuffled-disk null gives the 50th percentile). Best setting per model:

| Model | Percentile of the site in the crop |
|---|---|
| released ink_9um | 97 |
| TRACE v3A | 94 |
| TRACE sA | 92 |
| TRACE v3B | 92 |
| TRACE sB | 88 |
| KLAVIS (dense9um-all7) | 85 |
| hecate 9.6 um | 81 |

Every public detector responds to the real ink there, weakly, and the base model ranks highest. One site cannot rank detectors (the percentile is the best of 20 settings per model, the crop contains one 3 mm disk of ink, and the ranking disagrees with the PHerc0841 labels); the harness is offered as a control anyone can run, not as evidence for TRACE. All 20 settings per model are in `results/e90_pherc1447_results.json`. Figure: `docs/fig/pherc1447_site.jpg` (two of the detectors; the bright ring at the right of the detector panels is a void edge). For 8.64 um scrolls there is no evidence that TRACE beats the base model.

## 5. Survey pipeline and vetting

`pcu/survey.py`: a spiral-fit winding, snapped per column to the m7 surface-prediction peak (`pcu/refine.py`), ARAP-flattened (`pcu/flatten.py`, as-rigid-as-possible parametrisation from an arc-length layout), rendered natively with own tile splitting (`pcu/render.py`), then scored by the detector ensemble. Kaggle job: `kaggle/pcu-trace-survey/pcu_trace_survey.py`; combined fit-plus-survey for new scrolls: `kaggle/pcu-fit-new/template.py`.

On crushed, void-pitted eligible scrolls the window statistic produces text-range values without text. Every hotspot must pass:

1. **Depth profile** (`exp/e71_candidate.py`): re-render with 41 layers and score 17-layer windows at offsets -8..+8; real surface ink peaks at 0 and falls off symmetrically.
2. **Neighbouring-sheet control** (`exp/e73_vet.py`): the same angular sector on the windings just inside and outside must be blank. Eight of nine hotspots on PHerc0826, 0211, 0813 and 0800 failed this test (regional responses spanning three fitted windings); the ninth (PHerc0826 region A) is sheet-specific but unreadable. Figure: `docs/fig/vetting_neighbour_control.jpg`.
3. **Lattice test** (`exp/e91_lattice.py`): blob centroids of real text show line- and letter-pitch peaks 1.5 to 2.2x a shuffled null; texture does not.

## 6. What did not work

- **De-Paganin / re-filtering** (E52/E53): Paganin retrieval is a linear filter; re-filtering the public uint8 volumes only loses information (probe 0.641 to 0.604).
- **Fit-free flat patches** (E70): on a real mesh the sheet leaves its tangent plane by a median 21 voxels (p90 44) over 1.2 mm while the detector tolerates +-1 to 2; flat or quadratic patches score at chance. A "sweep the whole scroll" detector needs per-pixel sheet following.
- **Rough-surface training** (depth warps of +-5 voxels) cannot represent that deviation.
- **Score-guided surface warping** (E82): gains no more on a candidate than on blank papyrus (+0.02 to 0.05); the renders are already on the sheet.
- **Synthetic ink insertion for calibration** (E95): the ink's mean imprint (+9 grey levels on the recto layers, d' 0.27) is invisible to the detectors; pasting real residual texture or whole letters recovers only AUC 0.6 to 0.77 against 0.85+ for real text. No realistic yardstick could be built.
- **Dual energy for lead ink on Paris 4** (E81/E96): 78/137 keV cannot see lead (attenuation ratio 0.98 vs 0.85 for carbon); the 74/110 keV 45 um scans can in principle (1.48 vs 0.89) but the ratio noise at letter scale (MAD 10 to 18%) hides lead at the levels reported by Brun et al.; no heavily leaded ink in the title box. The detector side of the title-box test (known four-line column passes, blank stretch shows nothing) is `docs/fig/paris4_titlebox_control.jpg`; the dual-energy volumes themselves are not shown.

Numbered experiments refer to our lab log; the scripts in `exp/` carry the same numbers. Licence for the checkpoints: CC BY-NC-SA 4.0 (see `MODEL_CARD.md`).
