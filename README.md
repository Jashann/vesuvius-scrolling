# vesuvius-scrolling

Tools for the Vesuvius Challenge by Jashanjot Singh Gill ([@Jashann](https://github.com/Jashann)) and Nhat Nam Tran ([@nnm2602](https://github.com/nnm2602)), September 2026.

## The problem

The scrolls that are still unread are scanned at 8.64 to 9.36 um per voxel. At that resolution two things are hard: telling **which wrap of the rolled papyrus** a point belongs to (the spiral fit needs *winding constraints*, and the organisers' own note is that constraints at roughly 93% accuracy make the fit worse than none), and **seeing ink**, because at four times coarser than the 2.4 um scans the ink model was distilled from, ink is a faint change of surface texture, not a density.

## What we built

1. **PCU, certified automatic winding constraints.** Lasagna (the organisers' sheet model) predicts `cos(2 pi w)` where `w` is the wrap coordinate. We treat that channel as a wrapped phase and count the wraps between two points in three ways from that one prediction (local phase cycles, an integral of wrap density, and a whole-slice unwrapped field). A count is *certified* only when all three agree; a fourth vote from a different model (the recto-surface net) gives a *gold* tier. On PHerc. Paris 4, against all **1,502** human adjacent-wrap pairs (63 slices), certified counts are **99.62% correct at 70% coverage** (4 errors in 1,051; 95% Wilson lower bound 99.0%), and the gold tier of generated steps is **942 / 942** (lower bound 99.6%). The base rate is 95.6% (the best single measurement). Fed to the organisers' spiral fitter as relative-winding constraints they cut winding slips per wrap on Paris 4 from **20.3% to 9.3%** (fully automatic fit, held-out human ladders, 95% CI 6.6 to 12.9%) and on the eligible scroll PHerc0826 from **50.1% to 36.4%** (held-out gold ladders). Details: [`docs/PCU_REPORT.md`](docs/PCU_REPORT.md).

2. **TRACE, an ink detector trained on the eligible-protocol scans.** The organisers' `ink_9um` model fine-tuned on their *real* 9.362 um surface volumes (PHerc0139, PHerc0814) with their 2.4 um ink predictions on the same meshes as soft targets, then extended (v3) with the 8.64 um and 9.362 um fragment scans (PHerc0009B, PHerc0343P, PHerc0500P2). On PHerc0841, a scroll no checkpoint was trained on, against human ink labels, AUC rises from **0.705 to 0.747** (released model) to **0.837 to 0.868** (v3B). Details, including how checkpoints were selected and where the model falls short: [`docs/TRACE.md`](docs/TRACE.md) and [`MODEL_CARD.md`](MODEL_CARD.md).

Around them: a survey pipeline for eligible scrolls (spiral fit, snap to the m7 surface, ARAP flatten, native render, detector ensemble), an umbilicus estimator for the scrolls that have none, a hotspot vetting protocol (depth profile, neighbouring-sheet control, lattice test), a positive-control harness at the organisers' PHerc1447 ink site, and the negative results (they cost real compute and may save yours).

## Results

| What | Baseline | This work |
|---|---|---|
| Winding count precision, Paris 4, 1,502 human adjacent pairs, at 70% coverage | best single measurement 95.6% | **99.62%** (4 errors; 95% lower bound 99.0%) |
| Same, densest bands (wrap spacing below 173 um) | | **100%** (529 pairs, 33 to 71% covered) |
| Generated steps, gold tier (4 votes), matched to human ladders | | **942 / 942** (lower bound 99.6%) |
| Spiral fit slips per wrap, Paris 4, fully automatic, 32 held-out human ladders | 20.3% (no constraints) | **9.3%** (30/324; CI 6.6 to 12.9%) |
| Spiral fit slips per wrap, PHerc0826 (eligible), 8 held-out gold slices | 50.1% | **36.4%** |
| Ink AUC vs human labels, PHerc0841 w00 / ag144 / ag174 (never trained on) | 0.747 / 0.736 / 0.705 | v3B **0.868 / 0.862 / 0.837**; sB 0.857 / 0.839 / 0.817 |
| Ink AUC, PHerc0139 w043 vs the organisers' 2.4 um map (teacher) | 0.782 | 0.899 |
| Ink AUC, PHerc0139 w043 vs sparse human labels | 0.82 | 0.76 to 0.79 (fine-tunes inherit the teacher's errors) |

Held-out means never trained on; the TRACE checkpoints were *selected* on the held-out scores (about 8 evaluations per run, spread about 0.01 AUC), which is disclosed in `docs/TRACE.md`. Slips per wrap: a ladder is a human-clicked line of points across consecutive wraps; a slip is one adjacent pair the fitted surfaces assign to the wrong wrap difference. A *gold ladder* is a PCU-generated one used as ground truth where no human ladders exist.

Reproduce the ink table from public data in one command (GPU, about 40 minutes): `python eval_trace.py` (see below). The winding certificate: `python exp/e9_certificate.py` (laptop, 15 minutes).

### Figures

| | |
|---|---|
| ![precision vs coverage](docs/fig/pcu_precision_coverage.png) | ![PHerc1447 site](docs/fig/pherc1447_site.jpg) |
| PCU certificate on Paris 4: precision of certified wrap counts against the 1,502 human pairs as the agreement tolerance is relaxed. | Positive control at the organisers' PHerc1447 ink site (1.2 cm crops): CT mid layer, TRACE sA, KLAVIS. Every public detector responds only weakly here. |
| ![neighbour control](docs/fig/vetting_neighbour_control.jpg) | ![title box](docs/fig/paris4_titlebox_control.jpg) |
| Why hotspots must be vetted: a text-range response on PHerc0826 w073 appears on the neighbouring wraps too (same angular sector), so it is regional texture. | Paris 4 title box at 9.6 um: the known text passes (positive control); the blank stretch shows nothing to any detector. |

## Setup

Python 3.12. Install `torch` for your platform, then:

```bash
pip install -r requirements.txt
git clone --depth 1 -b merge-ink-pipelines https://github.com/ScrollPrize/villa.git ext/villa-ink   # koine_machines (ink models)
git clone --depth 1 https://github.com/ScrollPrize/villa.git ext/villa                              # vesuvius.image_proc
```

All scroll data is read straight from the Vesuvius Challenge open-data bucket through `pcu.data` (chunk cache in `$PCU_CACHE`, default `data/cache`). The spiral fit is the organisers' `fit_spiral.py` (villa at `6bbe6e2`).

Checkpoints (ink_9um format) and the large constraint files are on the [GitHub release](https://github.com/Jashann/vesuvius-scrolling/releases/tag/v1.0): `v3B_best.pth`, `v3A_best.pth` (TRACE-v3, recommended), `sB_best.pth`, `sA_best.pth` (v1), `pcu_relative_windings_PHercParis4_gold.json` (377k gold certified pairs, Paris 4), `pcu_gold_0125.json` (PHerc0125). Smaller files ship in `data/` (see `data/README.md`).

## How to use

**Score any surface volume with TRACE** (rows along z, inner face first; 8.64 um scans: `--resample 8.64`):

```bash
python -m pcu.trace_infer \
  https://dl.ash2txt.org/PHerc0841/segments/20260220213127-w00/surface-volumes/9.366um-1.2m-113keV-volume-20250821151531.zarr/0 \
  data/cache/checkpoints/v3B.pth out.png
```

or from Python: `from pcu.trace_infer import load_model, predict` (`predict(model, sv)` returns a probability map). `python eval_trace.py --save_maps` reproduces the held-out table (released ink_9um seeds 42 and 43, KLAVIS, all four TRACE checkpoints, three PHerc0841 segments and PHerc0139 w043) and writes the maps.

**Certified winding constraints** (Paris 4, Lasagna L2 predictions from the bucket):

```bash
python exp/e11_generate.py <z_L3> 48            # one slice -> runs/constraints/pcu_z2_<z>.json
OUT=runs/constraints python exp/e13_band.py <z_lo> <z_hi> 2   # a z-band, gold and silver tiers
python exp/e9_certificate.py                     # precision / coverage on the human ladders
python exp/e63_merge_gold.py out.json 'runs/constraints/*_gold.json'   # one VC3D point-collection file
```

The merged file loads into `fit_spiral.py` as an extra relative-winding document (`input_use_pcl_relative: true`, integer collection keys). `kaggle/pcu-fit-b3/` is the fit job that produced the 9.3% result (strip weights radius 10, dt 20, 10x sampling); `kaggle/pcu-fit-auto/` the no-constraint and default-weight baselines.

**Fit, render and survey a scroll with no published umbilicus:**

```bash
python exp/e69_umb_band.py PHerc0813 11300 12300 umb_PHerc0813.json   # band umbilicus
# kaggle/pcu-fit-new/template.py: spiral fit with CW/ACW pilots, then TRACE on every 3rd winding
# (the template's SCROLL/Z placeholders are filled per job; see kaggle/pcu-new-* in the log)
```

**Vet a hotspot** (depth profile, then the neighbouring wraps in the same angular sector):

```bash
python exp/e73_vet.py PHerc0800 grids_n800.npz w048 0.35 0.70 w047,w049 s800w048
```

**Train TRACE:** `kaggle/pcu-trace-train/pcu_trace_train.py` (env `TRACE_MANIFEST`, `TRACE_NTRAIN`, `TRACE_MATCH_PX`) or `modal/trace_v3.py` for the exact v3 run (`runs/trace_manifest_v3.json`, 128 segments).

Scripts in `exp/` that load several checkpoints skip any that are missing locally.

## Limitations

- **No letters were read.** Surveys of about 600 windings across PHerc0826, 0125, 0211, 0191, 0257, 0358, 0813, 0800, 1447 and 1203 produced text-range hotspots that all failed the neighbouring-sheet control (regional texture of crushed papyrus), except one sheet-specific but unreadable patch on PHerc0826. At the organisers' PHerc1447 ink site every public detector responds only weakly (the site ranks between the 81st and 97th percentile of a 1.6 cm crop, and the released model ranks highest there), so one site cannot rank detectors.
- **TRACE learns the teacher.** Against sparse human labels on PHerc0139 w043 the fine-tunes score below the base model. The window statistic (p95) is a triage tool, not a text detector.
- **PCU on eligible scrolls.** The certificate was tuned on Paris 4; on PHerc0139 its estimated precision drops to about 99%, and the eligible-scroll fit after constraints is still far worse than Paris 4 (36% vs 9% slips per wrap). Coverage is 70%, by design.
- The whole-slice winding field is a diagnostic, not a replacement for the spiral fit (it jumps along sheets).

## Layout

`pcu/` library (`constraints.py`, `crestgraph.py`, `phase.py`, `slice2d.py`, `mcfcut.py` for PCU; `data.py`, `zarr3.py` for S3 access; `refine.py`, `flatten.py`, `render.py`, `survey.py`, `umbilicus.py`, `phasesurf.py`, `rowstack.py` for the survey; `trace_infer.py`). `exp/` experiment scripts, numbered as in our lab log. `kaggle/`, `modal/` the jobs exactly as run (Kaggle T4 x2, Modal L4). `docs/` reports and figures. `data/` constraint files and umbilici. `results/` the logs and metric files behind the numbers above. `runs/` training manifests.

## Credits and licence

Built on the organisers' work: Lasagna and the recto/m7 surface models, `ink_9um` (the base model and its `koine_machines` inference code), hecate, `fit_spiral.py`, the tracks, umbilici and `vc_gen_umbilicus` objective, `vc_render_tifxyz`, and the human winding and ink annotations. KLAVIS (domenicor046/ink9um-dense) was used as a comparison detector. Data: Vesuvius Challenge open data.

Code MIT. Model weights, constraint files and everything else derived from scroll data: CC BY-NC-SA 4.0 (see `LICENSE`).
