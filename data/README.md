# Data files

All coordinates are in the organisers' volume frame of the named scan; JSON point collections follow the VC3D point-collection format (`vc_pointcollections_json_version: "1"`, `collections` keyed by integer id, each a list of points with `x, y, z` and a `winding` value), the same format as the public `relative_windings.json`, so `fit_spiral.py` loads them as an extra relative-winding document. Derived from Vesuvius Challenge open data; CC BY-NC-SA 4.0.

## gold/ (PCU certified constraints, gold tier)

| File | Scan | Level / coordinates | Content | SHA256 |
|---|---|---|---|---|
| `pcu_gold_0826_b0.json` | PHerc0826 9.362 um (20250821 protocol) | level 2 (4x downsampled, x y z in that order) | gold v1, band z 8500 to 9500, 13,858 ladders (used by fits B0) | `51659a10…2400538` |
| `pcu_gold_0826_b2.json` | PHerc0826 | level 2 | gold v2, same band, 25,771 ladders (used by fits B1w10 / B1w20) | `ae33178b…522357b` |
| `gold0826_heldout8.json` | PHerc0826 | level 2 | 8 slices (8528, 8656, ... 9424) never given to any fit, 3,426 ladders: the held-out scoring set | `edc56561…41e06f99` |
| `pcu_gold_0211.json` | PHerc0211 9.362 um | level 2 | gold, band z 9500 to 10500 | `d5f2572d…bfc1acdcd` |

On the GitHub release: `pcu_relative_windings_PHercParis4_gold.json` (Paris 4, scan 20260411134726, level 2, 175 slices, 377,437 gold pairs) and `pcu_gold_0125.json` (PHerc0125, band z 10000 to 11000).

Full hashes: run `shasum -a 256 data/gold/*.json`.

## umbilicus/ (band umbilici for scrolls without a published one)

`umb_band_PHerc<id>[_z...].json`: `{"control_points": [{"x", "y", "z", "score"}, ...]}` in level-0 voxel coordinates of the scroll's 9.362 um (or 8.64 um for PHerc1218) volume, one point per 100 slices over the fitted band, estimated with `exp/e69_umb_band.py` (the organisers' `vc_gen_umbilicus` objective plus a Theil-Sen line through the band). Scrolls: 0191 (two bands), 0211, 0257, 0358 (two bands), 0800, 0813, 1218. The format is what `fit_spiral.py` and `kaggle/pcu-fit-new/template.py` read.

## cache/

`pcu.data` stores downloaded zarr chunks under `$PCU_CACHE` (default `data/cache`); it is not part of the repository.
