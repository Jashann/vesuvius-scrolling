# Model card: TRACE checkpoints (release v1.0)

**Files (SHA256):**

| File | Step | SHA256 |
|---|---|---|
| `sA_best.pth` | 3000 | `56ba092d2563d124f55111f828c3f9af4b5db5385b6f3772971d9a69e1d110b4` |
| `sB_best.pth` | 4500 | `8c4075c29c540107fb691afa1f8c7b26ab8f354ce8f37e2f0a0f9f3133fb80bc` |
| `v3A_best.pth` | 4000 | `152d4f8eaee7c0623795f4cb73aac6ab38f754802ce2ab164cc250960fa2c236` |
| `v3B_best.pth` | 5000 | `812ffe2b42b3293974052ebbee60a0fe939ca2bfc3f03a517a0bc2e2339e5da2` |

ink_9um format; load with `pcu.trace_infer.load_model` (`koine_machines.inference.infer.build_repo_training_model_bundle`).

**Base model:** the organisers' `scrollprize/ink_9um` (`hybrid_3d2d-seed42/step-075000.pth`): a hybrid 3D-stem / 2D U-Net that takes a 17-layer surface volume as a 5-D tensor (batch, 1, 17, 128, 128), about 1.2 mm per patch at 9.362 um. All TRACE checkpoints are fine-tunes of it; the architecture is unchanged.

**Input.** A surface volume with rows along z, **inner (recto) face first**, 9.362 um per pixel (8.64 um scans resampled in-plane by 8.64/9.362; the layer spacing is left as scanned, as in training), the central 17 of at least 17 layers, each 128 x 128 patch normalised with the organisers' `normalize_robust`. Reversed layer order scores near chance for every model.

**Training data.** v1: 32 segments of PHerc0139 and PHerc0814 (organisers' 9.362 um surface volumes; soft targets = the organisers' 2.4 um ink predictions on the same meshes, resized). v3: 128 segments = 56 of PHerc0139/0814 + PHerc0009B (18, 8.64 um) + PHerc0343P (8, 8.64 um) + PHerc0500P2 (46, 9.362 um), fragment targets from their 2.2 to 2.4 um ink predictions. Soft BCE, AdamW (sA 2e-5, sB 6e-5, v3A 3e-5, v3B 8e-5), fp16, depth jitter +-2, flips, gamma, blur, noise. Manifests: `runs/trace_manifest*.json`; script: `kaggle/pcu-trace-train/pcu_trace_train.py`.

**Checkpoint selection (disclosure).** Every 1000 to 1500 steps the model was scored on four held-out segments (PHerc0841 w00, ag144, ag174 and PHerc0139 w043, all against the organisers' human ink labels); the checkpoint with the best mean label AUC over the four was kept (`pcu_trace_train.py`, the `score` line), and the layer order was confirmed on the same segments. The held-out set therefore chose among about 8 evaluations per run and about 8 runs. The spread between the best and the last evaluation of a run is about 0.01 AUC, small next to the 0.12 gain over the base model, but PHerc0841 is "never trained on", not "never looked at", and w043's neighbouring wraps are in the training set (w041, w042, w045, w046 for v1; w044 as well for v3).

**Released numbers (AUC vs human labels, PHerc0841 w00 / ag144 / ag174, fresh-clone run of `eval_trace.py`):** ink_9um seed 42 0.746 / 0.736 / 0.705; seed 43 0.772 / 0.709 / 0.749; KLAVIS 0.799 / 0.807 / 0.762; sA 0.853 / 0.850 / 0.803; sB 0.857 / 0.839 / 0.817; v3A 0.866 / 0.860 / 0.832; v3B 0.868 / 0.862 / 0.837. PHerc0139 w043 vs human labels: seed 42 0.825, seed 43 0.830, KLAVIS 0.851, sA 0.789, sB 0.764, v3A 0.745, v3B 0.760. Block-bootstrap intervals (1 cm blocks) in `results/trace_heldout_auc_ci.json`: v3B's paired gain over seed 42 is +0.10 to +0.12 on PHerc0841 with lower bounds +0.05 to +0.06. Reproduce with `python eval_trace.py` (our run: `results/trace_heldout_auc.json`, `results/eval_trace.log`). The v3 over v1 difference is within the selection spread.

**Known failure modes.** Blobs, not strokes, on eligible scrolls (the model sees a third of a letter). Regional responses on crushed, void-pitted papyrus reach the same window statistic as text; always apply the neighbouring-sheet control (`exp/e73_vet.py`). On PHerc0139 w043 the fine-tunes track the 2.4 um teacher map (0.90 AUC) but score lower than the base model against the sparse human labels there: they inherit the teacher's errors. At the organisers' PHerc1447 ink site (8.64 um), no public detector including TRACE makes the ink stand out strongly, and the base model ranks highest; use v3B for 9.36 um segments only.

**Licence.** CC BY-NC-SA 4.0 (derived from Vesuvius Challenge data and the organisers' ink_9um weights, which carry the challenge's non-commercial, share-alike terms).
