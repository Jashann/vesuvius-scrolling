# TRACE: native coarse-protocol ink detection

**One-line summary.** The released `ink_9um` model was distilled mostly from 2.4 um data downsampled 4x. Fine-tuning it on the team's *real* 9.362 um surface volumes, with the team's 2.4 um ink predictions of the same meshes as soft targets, raises held-out AUC on an unseen scroll (PHerc0841, human labels) from 0.71-0.75 to 0.84-0.88. Adding the public fragment pairs at the coarse protocols (8.64 um PHerc0009B and PHerc0343P, 9.362 um PHerc0500P2) adds another +0.02 (v3).

## 1. Why

Ink at the eligible protocol (9.362 um, 1.2 m propagation, 113 keV, Paganin) is contextual surface texture, not density: on Paris 4 rendered at 9.6 um, per-pixel density d' between ink and papyrus is about 0.2, gradient-energy d' 0.3-0.5, and no hand-made texture map separates them. The signal is real but lives in the joint structure of the layers around the sheet, which is what a detector trained on real coarse renders can learn and a detector trained on pooled fine data cannot.

## 2. Data

| Set | Segments | Input | Target |
|---|---|---|---|
| v1 train | 31 (PHerc0139, PHerc0814) | team 9.362 um surface volumes, central 22 layers | team 2.4 um ink prediction on the same mesh, resized to the coarse canvas |
| v2 train | 56 (same scrolls) | same | same |
| v3 train | 128: v2 + PHerc0009B (18, 8.64 um), PHerc0343P (8, 8.64 um), PHerc0500P2 (46, 9.362 um) | 8.64 um volumes resampled in-plane to 9.362 um | fragment 2.2-2.4 um ink predictions |
| held out | PHerc0841 x3 (human ink labels; 9.366 um), PHerc0139 w043 (2.4 um map) | | |

Manifests: `runs/trace_manifest.json` (v1/v2), `runs/trace_manifest_v3.json` (v3, built by `exp/e93_manifest_v3.py` from S3 listings).

## 3. Training

`kaggle/pcu-trace-train/pcu_trace_train.py`: 17 x 128 x 128 patches, per-patch `normalize_robust` (as in the team's inference), depth jitter +-2 layers, flips, gamma, blur and noise augmentation, soft BCE against the resized 2.4 um map, AdamW, fp16 on a T4 (batch 16) or L4 (batch 24). Checkpoints are selected on held-out label AUC every 1000-1500 steps. Students plateau after 1-5k steps at every data size tried (32, 56, 128 segments); v3 gains come from the new protocols, not from more steps.

Runs: sA (lr 2e-5), sB (6e-5), sC/sD (1e-4 aug / 3e-5 no-aug), v2A/v2B (56 segments), v3A/v3B (128 segments, `modal/trace_v3.py`).

## 4. Results

Held-out AUC against human labels on PHerc0841 (three segments) and against the 2.4 um map on PHerc0139 w043:

| Model | 0841 w00 | 0841 ag144 | 0841 ag174 | 0139 w043 |
|---|---|---|---|---|
| released ink_9um | 0.747 | 0.736 | 0.705 | 0.782 |
| TRACE sA | 0.853 | 0.850 | 0.803 | 0.895 |
| TRACE sB | 0.857 | 0.839 | 0.817 | 0.898 |
| TRACE v2A (56 segments) | 0.855 | 0.851 | 0.814 | 0.881 |
| TRACE v3B (128 segments) | **0.879** | **0.855** | **0.843** | 0.901 (map) |

Reversed depth order scores near chance for every model, so surface volumes must be inner face first.

**Text presence at the eligible protocol.** As a whole-winding statistic we use the 95th percentile of the probability over the winding and over the best 2 x 2 cm window. On known text at 9.362 um (0139 w043, Paris 4 at 9.6 um) sB gives 0.54-0.61 and sA 0.69-0.74; on blank team segments of PHerc1447 and PHerc0800 sB gives 0.21-0.29 and sA 0.33-0.44. The released model cannot separate the two (0.66-0.75 for both).

**Positive control at a real eligible-scroll ink site.** On 2026-09-24 the organisers reported letters on PHerc1447 (8.64 um) near (x 4144, y 2742, z 12557). `exp/e90_control1447.py` scores the published segment that passes 8.6 voxels from that point. The site's 3 mm-disk mean ranks at the 97th percentile of a 1.6 cm crop for the released model (resampled to 9.6 um, forward, +2 layers), 94th for TRACE v3A, 92nd for v3B and sA, 88th for sB. So every public detector responds to the real ink there, weakly; none makes it legible without site fine-tuning, which matches the organisers' description.

## 5. Survey pipeline and vetting (what we learned the hard way)

`pcu/survey.py`: a spiral-fit winding -> snap each column to the m7 surface-prediction peak (`pcu/refine.py`) -> ARAP flatten (`pcu/flatten.py`) -> native render with own tile splitting (`pcu/render.py`) -> detector ensemble. Kaggle job: `kaggle/pcu-trace-survey/pcu_trace_survey.py`; combined fit+survey for new scrolls: `kaggle/pcu-fit-new/template.py`.

On crushed, void-pitted eligible scrolls the window statistic produces text-range values without text. Every hotspot must pass:

1. **Depth profile** (`exp/e71_candidate.py`): re-render with 41 layers and score 17-layer windows at offsets -8..+8; real surface ink peaks at 0 and falls off symmetrically.
2. **Neighbouring-sheet control** (`exp/e73_vet.py`): the same angular sector on the windings just inside and outside must be blank. Eight of nine hotspots we found on PHerc0826, 0211, 0813 and 0800 failed this test (regional responses spanning three fitted windings).
3. **Lattice test** (`exp/e91_lattice.py`): blob centroids of real text show line and letter-pitch peaks 1.5-2.2x a shuffled null; texture does not.

## 6. What did not work

- **De-Paganin / re-filtering** (E52/E53): Paganin retrieval is a linear filter; re-filtering the public uint8 volumes only loses information (probe 0.641 -> 0.604).
- **Fit-free flat patches** (E70): on a real mesh the sheet leaves its tangent plane by a median 21 voxels (p90 44) over 1.2 mm, while the detector tolerates +-1-2; flat or quadratic patches score at chance. A "sweep the whole scroll" detector needs per-pixel sheet following.
- **Rough-surface training** (depth warps of +-5 voxels) cannot represent that deviation; killed.
- **Score-guided surface warping** (E82): gains no more on a candidate than on blank papyrus (+0.02-0.05); our renders are already on the sheet.
- **Synthetic ink insertion for calibration** (E95): the ink's mean imprint (+9 grey levels on the recto layers, d' 0.27) is invisible to the detectors; pasting real residual texture or whole letters recovers only AUC 0.6-0.77 against 0.85+ for real text. We could not build a realistic yardstick.
- **Dual energy for lead ink on Paris 4** (E81/E96): 78/137 keV cannot see lead (ratio 0.98 vs 0.85 for carbon); the 74/110 keV 45 um scans can in principle (1.48 vs 0.89) but the ratio noise at letter scale (MAD 10-18%) hides Brun-level lead; no heavily leaded ink in the title box.

Numbered experiments refer to our lab log; the scripts in `exp/` carry the same numbers.
