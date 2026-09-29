# Model card: TRACE checkpoints (release v1.0)

**Files:** `sA_best.pth`, `sB_best.pth` (v1), `v3A_best.pth`, `v3B_best.pth` (v3). ink_9um format; load with `pcu.trace_infer.load_model`.

**Base model:** the organisers' `scrollprize/ink_9um` (`hybrid_3d2d-seed42/step-075000.pth`), a 2D U-Net taking 17 surface layers as channels at 128 x 128 (about 1.2 mm at 9.362 um). All TRACE checkpoints are fine-tunes of it; the architecture is unchanged.

**Input.** A surface volume with rows along z, **inner (recto) face first**, 9.362 um per pixel and per layer (8.64 um scans resampled by 8.64/9.362), the central 17 of at least 17 layers, each 128 x 128 patch normalised with the organisers' `normalize_robust`. Reversed layer order scores near chance for every model.

**Training data.** v1: 31 segments of PHerc0139 and PHerc0814 (organisers' 9.362 um surface volumes; soft targets = the organisers' 2.4 um ink predictions on the same meshes, resized). v3: 128 segments = 56 of PHerc0139/0814 + PHerc0009B (18, 8.64 um) + PHerc0343P (8, 8.64 um) + PHerc0500P2 (46, 9.362 um), fragment targets from their 2.2-2.4 um ink predictions. Soft BCE, AdamW (sA 2e-5, sB 6e-5, v3A 3e-5, v3B 8e-5), fp16, depth jitter +-2, flips, gamma, blur, noise. Manifests: `runs/trace_manifest*.json`; script: `kaggle/pcu-trace-train/pcu_trace_train.py`.

**Checkpoint selection (disclosure).** Every 1000-1500 steps the model was scored on the held-out PHerc0841 segments (human labels) and PHerc0139 w043; the checkpoint with the best mean held-out label AUC was kept, and the layer order was confirmed on the same segments (no flip in every run). The held-out set therefore chose among about 8 evaluations per run. The spread between the best and the last evaluation is about 0.01 AUC (see `docs/TRACE.md`), small next to the +0.12 gain over the base model, but PHerc0841 is "never trained on", not "never looked at".

**Released numbers (AUC vs human labels, PHerc0841 w00 / ag144 / ag174):** base 0.747 / 0.736 / 0.705; sA (step 3000) 0.853 / 0.850 / 0.803; sB (step 4500) 0.857 / 0.839 / 0.817; v3A (step 4000) 0.866 / 0.860 / 0.832; v3B (step 5000) 0.868 / 0.862 / 0.837. Reproduce with `python eval_trace.py`.

**Known failure modes.** Blobs, not strokes, on eligible scrolls (the model sees a third of a letter). Regional responses on crushed, void-pitted papyrus reach the same window statistic as text; always apply the neighbouring-sheet control (`exp/e73_vet.py`). On PHerc0139 w043 the fine-tunes track the 2.4 um teacher map (0.90 AUC) but score lower than the base model against the sparse human labels there (0.76-0.79 vs 0.82): they inherit the teacher's errors. At the organisers' PHerc1447 ink site, no public detector including TRACE makes the ink stand out strongly.

**Licence.** CC BY-NC-SA 4.0 (derived from Vesuvius Challenge data and the organisers' ink_9um weights).
