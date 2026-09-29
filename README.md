# vesuvius-scrolling

Tools for the Vesuvius Challenge built by Jashanjot Singh Gill and Nhat Nam Tran (September 2026). Two contributions:

1. **PCU: certified automatic winding constraints.** Lasagna's `cos` channel is treated as a wrapped winding phase; wrap counts between two points are measured three independent ways and certified only when all three agree. On PHerc. Paris 4 the certified counts are **99.62% correct at 70% coverage** against all 1,443 human adjacent-wrap pairs (100% below 173 um wrap spacing). Fed to the team's spiral fitter as relative-winding constraints they cut winding slips per wrap from **20.3% to 9.3%** on Paris 4 (fully automatic fit, held-out human ladders) and from **50.1% to 36.4%** on the eligible scroll PHerc0826 (held-out gold ladders). Details: [`docs/PCU_REPORT.md`](docs/PCU_REPORT.md).
2. **TRACE: an ink detector trained on the eligible-protocol scans.** `ink_9um` fine-tuned on the team's real 9.362 um surface volumes (PHerc0139, PHerc0814) with the team's 2.4 um ink predictions as soft targets, then extended (v3) with the 8.64 um and 9.362 um fragment scans (PHerc0009B, PHerc0343P, PHerc0500P2). On the held-out scroll PHerc0841 (human labels, never trained on) it raises AUC from **0.71-0.75 (released ink_9um) to 0.84-0.88**. Details: [`docs/TRACE.md`](docs/TRACE.md).

Plus the pieces around them: a spiral-fit survey pipeline for eligible scrolls (snap to the m7 surface, ARAP flatten, native render, detector ensemble), an umbilicus estimator for scrolls that have none, a hotspot vetting protocol (depth profile, neighbouring-sheet control, lattice test), and a positive-control harness at the PHerc1447 ink site.

## Results at a glance

| What | Baseline | This work |
|---|---|---|
| Winding constraints, Paris 4, precision at 70% coverage (1,443 human pairs) | community constraints ~93% | **99.62%** |
| Spiral fit slips per wrap, Paris 4, fully automatic, held-out ladders | 20.3% | **9.3%** (PCU gold, weights 10/20) |
| Spiral fit slips per wrap, PHerc0826 (eligible), held-out gold ladders | 50.1% | **36.4%** |
| Ink AUC, PHerc0841 held-out (w00 / ag144 / ag174), released ink_9um | 0.747 / 0.736 / 0.705 | TRACE sB **0.857 / 0.839 / 0.817**; v3B **0.879 / 0.855 / 0.843** |
| Ink AUC, PHerc0139 w043 (held-out segment) vs 2.4 um map | 0.782 | **0.898** |

Negative results are documented too (they cost real compute and may save yours): de-Paganin re-filtering, flat fit-free patches, score-guided surface warping, synthetic ink insertion, dual-energy lead detection on Paris 4. See `docs/TRACE.md`, section "What did not work".

## Weights and data

Model checkpoints (ink_9um format, loadable with `koine_machines.inference.infer.build_repo_training_model_bundle`) and the large constraint files are attached to the GitHub release of this repository:

- `v3B_best.pth`, `v3A_best.pth`: TRACE-v3 (recommended)
- `sB_best.pth`, `sA_best.pth`: TRACE v1
- `pcu_relative_windings_PHercParis4_gold.json`: 377k gold certified pairs, Paris 4
- `pcu_gold_0125.json`: PHerc0125 constraints

Smaller constraint files ship in `data/gold/` (PHerc0826 bands, PHerc0211, the PHerc0826 held-out ladder set) and band umbilici in `data/umbilicus/`. Everything derived from scroll data is CC BY-NC-SA 4.0; code is MIT.

## Quick start

Requirements: Python 3.12, `torch`, `zarr==2.18.7`, `numcodecs==0.15.1`, `numpy`, `scipy`, `scikit-image`, `tifffile`, `imagecodecs`, `opencv-python-headless`, `imageio`, `requests`, plus the team's repositories on `sys.path` for the ink models:

```bash
git clone --depth 1 -b merge-ink-pipelines https://github.com/ScrollPrize/villa.git ext/villa-ink   # koine_machines
git clone --depth 1 https://github.com/ScrollPrize/villa.git ext/villa                              # vesuvius.image_proc
mkdir -p runs/kaggle/pcu-trace-train runs/modal_v3   # put sA/sB_best.pth and v3A/v3B_best.pth here
```

All data is read directly from the public S3 bucket through `pcu.data` (chunk cache in `$PCU_CACHE`).

**Score a segment's surface volume with TRACE** (any team surface volume; rows along z, inner face first):

```python
from pcu import data
import numpy as np
from exp.e90_control1447 import predict   # or copy the 30-line predict() from any exp script
sv = np.asarray(data.open_array("PHerc0841/segments/20260220213127-w00/surface-volumes/9.366um-1.2m-113keV-volume-20250821151531.zarr/0")[6:23])
```

then `predict(net, sv)` gives a probability map; `exp/e58_trace_survey.py team` reproduces the team-segment survey.

**Certified winding constraints for a slice / band (Paris 4 Lasagna L2):**

```bash
python exp/e11_generate.py <z_L3> 48        # certified ladders for one slice -> runs/constraints/*.json
python exp/e13_band.py <z_lo> <z_hi> 2       # a z-band
python exp/e9_certificate.py                 # precision/coverage on the human ladders
python exp/e63_merge_gold.py out.json 'runs/constraints/*.json'   # merge into one VC3D point-collection file
```

The merged file loads into `fit_spiral.py` as `relative_windings.json` (`input_use_pcl_relative: true`, integer collection keys). `kaggle/pcu-fit-bands/pcu_fit_bands.py` is the complete fit-plus-export job we ran (two z-bands per T4 pair).

**Fit, render and survey a scroll that has no published umbilicus:**

```bash
python exp/e69_umb_band.py PHerc0813 11300 12300 umb_PHerc0813.json   # band umbilicus (team objective + Theil-Sen)
# kaggle/pcu-fit-new/template.py: spiral fit with CW/ACW pilots, then the in-kernel TRACE survey of every 3rd winding
```

**Vet a hotspot** (depth profile and the neighbouring windings in the same angular sector):

```bash
python exp/e73_vet.py PHerc0800 grids_n800.npz w048 0.35 0.70 w047,w049 s800w048
```

**Train TRACE:** `kaggle/pcu-trace-train/pcu_trace_train.py` (prepare + train; env `TRACE_MANIFEST`, `TRACE_NTRAIN`, `TRACE_MATCH_PX`), or `modal/trace_v3.py` for the exact v3 run (`runs/trace_manifest_v3.json`, 128 segments).

## Layout

- `pcu/`: library. `constraints.py`, `crestgraph.py` (PCU), `data.py` (S3 zarr access with cache), `refine.py`, `flatten.py`, `render.py`, `survey.py` (winding -> surface volume), `rowstack.py` (row periodicity), `umbilicus.py`, `phasesurf.py`, `zarr3.py`.
- `exp/`: the experiment scripts referenced above (numbered as in our lab log).
- `kaggle/`, `modal/`: the jobs exactly as run (Kaggle T4 x2, Modal L4).
- `docs/`: `PCU_REPORT.md`, `TRACE.md`, `SUBMISSION_FORM.md`.
- `data/`: constraint files and band umbilici; `runs/`: training manifests.

## Honest status

No legible letters have been found on an eligible scroll with these tools. Surveys of ~600 windings across PHerc0826, 0125, 0211, 0191, 0257, 0358, 0813, 0800, 1447 and 1203 produced text-range hotspots that all failed the neighbouring-sheet control (regional texture of crushed papyrus), except one sheet-specific but unreadable patch on PHerc0826. At the organisers' PHerc1447 ink site, every public detector responds only weakly (the site ranks in the top 3-10% of a 1.6 cm crop). The value here is in the constraints, the detector gain, and the vetting protocol, not in a reading.
