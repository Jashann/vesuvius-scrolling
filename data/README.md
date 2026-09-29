# Data files

All coordinates are in the organisers' volume frame of the named scan; JSON point collections follow the VC3D point-collection format (`vc_pointcollections_json_version: "1"`, `collections` keyed by id, each `{"name", "metadata", "points": {id: {"p": [x, y, z], "wind_a": w}}}`), the same format as the public `relative_windings.json`, so `fit_spiral.py` loads them as an extra relative-winding document. Derived from Vesuvius Challenge open data; CC BY-NC-SA 4.0.

## gold/ (PCU certified constraints, gold tier)

| File | Scan | Coordinates | Content | SHA256 |
|---|---|---|---|---|
| `pcu_gold_0826_fit_v2.json` | PHerc0826 9.362 um | level 2 (x, y, z), 4x downsampled | gold v2, 63 slices every 16 from z 8500 to 9492, 25,771 ladders: **the constraint input of fits B1w10, B1w20 and C1** | `1b7f092c726c988b3d4dc68cc91da49ed7c9fa92e01e2ba917fb6ed1aa53bea4` |
| `gold0826_heldout8.json` | PHerc0826 | level 2 | 8 slices (8528, 8656, ... 9424, interleaved between the constraint slices) never given to any fit, 3,426 ladders: the held-out scoring set | `edc56561ae2aa836f3088cfc23773617c97b998ee40f05b4803915a541e06f99` |
| `pcu_gold_0826_band_z7500.json` | PHerc0826 | level 2 | gold, band z 7500 to 8492, 21,547 ladders (the band-fit job `kaggle/pcu-fit-bands`, not the runs reported) | `51659a10ead6d87a217f124f6b9e34057442c3b1db9b06271294bc9472400538` |
| `pcu_gold_0826_band_z9500.json` | PHerc0826 | level 2 | gold, band z 9500 to 10492, 26,087 ladders (same job) | `ae33178b6ad22e6dee4cd426461231a87a0b944c34b4fb50a46ee0188522357b` |
| `pcu_gold_0211.json` | PHerc0211 9.362 um | level 2 | gold, band z 9500 to 10492, 30,761 ladders | `d5f2572d7d3e5ffbf854ea5675bcf9ace1f78c98e6892ba0b84e3fdbfc1acdcd` |

The gold v1 file of fit B0 (13,858 ladders, an earlier generator version on the same slices) is superseded by v2 and not shipped.

On the GitHub release v1.0: `pcu_relative_windings_PHercParis4_gold.json` (Paris 4, scan 20260411134726, level 2, 175 slices, 165,227 ladders / 377,437 gold pairs; SHA256 `c632b2fcc8d77d5d121e42b1d1062d1115cb8b6e78f90f754f5825c0f620aa72`; the Paris 4 fits used its z2 8400 to 9400 subset, 62,260 ladders) and `pcu_gold_0125.json` (PHerc0125, band z 10000 to 10992, 55,458 ladders; SHA256 `7639f681fcea873551465f90d53e484358c747f6ec76221564bc9de7ae59ddd0`).

On release v1.1: the spiral-fit output grids (`grids_Paris4_{A2,B2,A,B,B3w10,B3w40,B4w4,B4w20}.npz`, `grids_PHerc0826_{A0,B0,B1w10,B1w20,C1}.npz`; one `wNNN` array of shape (3, rows, cols) per winding, level-2 xyz), scored by `exp/e31b_bootstrap.py`.

## umbilicus/ (band umbilici for scrolls without a published one)

`umb_band_PHerc<id>[_z...].json`: `{"control_points": [{"x", "y", "z", "score"}, ...]}` in level-0 voxel coordinates of the scroll's 9.362 um (or 8.64 um for PHerc1218) volume, one point per 100 slices over the fitted band, estimated with `exp/e69_umb_band.py` (the organisers' `vc_gen_umbilicus` objective plus a Theil-Sen line through the band). Scrolls: 0191 (two bands), 0211, 0257, 0358 (two bands), 0800, 0813, 1218. The format is what `fit_spiral.py` and `kaggle/pcu-fit-new/template.py` read; pass one to `exp/e73_vet.py` with `UMB_FILE=`.

## cache/

`pcu.data` stores downloaded zarr chunks under `$PCU_CACHE` (default `data/cache`); it is not part of the repository.
