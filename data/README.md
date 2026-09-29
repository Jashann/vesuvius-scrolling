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

On the GitHub release v1.0: `pcu_relative_windings_PHercParis4_gold.json` (Paris 4, scan 20260411134726, level 2, 175 slices two per 64, 165,227 ladders / 377,437 gold pairs; SHA256 `c632b2fcc8d77d5d121e42b1d1062d1115cb8b6e78f90f754f5825c0f620aa72`; the Paris 4 fits used the denser band file on release v1.1, below) and `pcu_gold_0125.json` (PHerc0125, band z 10000 to 10992, 55,458 ladders; SHA256 `7639f681fcea873551465f90d53e484358c747f6ec76221564bc9de7ae59ddd0`).

On release v1.1: the spiral-fit output grids (one `wNNN` array of shape (3, rows, cols) per winding, level-2 xyz), scored by `exp/e31b_bootstrap.py`, and the Paris 4 fit input. SHA256:

| File | SHA256 |
|---|---|
| `grids_Paris4_A2.npz` | `d5c5b0eec460200c4dde8af731bb9ea6cdbdca00736862d619d6df27ee12b949` |
| `grids_Paris4_B2.npz` | `1996ad2fb5038b3701310884f035be50ebe1586dd169081d56b686fc241eb01b` |
| `grids_Paris4_A.npz` | `a8ec89c61e4e0a2534380ab2a5059112a48ee3f94607f318a1d36dcf608deb6b` |
| `grids_Paris4_B.npz` | `63bbb7a35af3d0fdcae09994f94690051c3861b6fd88a0483b2293375e9481a3` |
| `grids_Paris4_B3w10.npz` | `6e758196a3702c35a27861ce11d0d60b948107f8cb0f980edf88050573475d31` |
| `grids_Paris4_B3w40.npz` | `5681652954cddde5e504bc4c61116c9e90c8fba52d156e8dc510e317e74762c6` |
| `grids_Paris4_B4w4.npz` | `7f55644134253fa14bd0cfe7c6685f45a6593b770ee02045e1f4a5c7bb2669b9` |
| `grids_Paris4_B4w20.npz` | `63c476cc2e21c7c6b6b4a47fcbd7166d9c34c28ac9208141aff98370f6ef8749` |
| `grids_PHerc0826_A0.npz` | `f7d3a1617273e978396ef1eeeb4f2aaac60b534d1e75667ee4b59579e2ca46ee` |
| `grids_PHerc0826_B0.npz` | `62570d80d174c06ba3565ce9139d797a21144bc8c738fb1b81b080a53fed40c9` |
| `grids_PHerc0826_B1w10.npz` | `99cc9ae63f41c369016c5108037bb58ad27955853b3953c571e4cd2ce838e2e7` |
| `grids_PHerc0826_B1w20.npz` | `594b1417912cc688a13bf57cff7ebe7c4c2f11741fd7f1d78a41c4ce896cdcfd` |
| `grids_PHerc0826_C0.npz` | `dc066ab0eafecd1e570a2bf7c1ac5d409951a76e9d10c9890aeb68303d83c4a9` |
| `grids_PHerc0826_C1.npz` | `3b3efba6c49ad4c46d25a0b5cd7d7ce5dcdfd7d58a1b2c5c9be75dd4d2d5f2d8` |
| `grids_Paris4_A2_seed2.npz`, `grids_Paris4_B3w10_seed2.npz` (second-seed replication, `modal/fit_p4.py`) | see `results/release_v1.1_sha256.txt` |
| `grids_Paris4_A2_z11000.npz`, `grids_Paris4_B3w10_z11000.npz`, `pcu_gold_Paris4_z11000_12000_fit_input.json` (second band, 52,260 ladders on 63 slices every 16 from z 11000, `modal/gold_band.py`) | see `results/release_v1.1_sha256.txt` |
| `pcu_gold_Paris4_z8400_9400_fit_input.json` (62,260 ladders on 63 slices, one every 16 level-2 slices from z 8400, the dense band file the Paris 4 fit jobs loaded; same generator as the whole-scroll file, which has two slices per 64) | `8aa44a98a2a8af7aa396b13060e7ac61d1496f5b1ddfe4b8934d000743fa971b` |

(The Paris 4 fit C grid, human windings as input, is `grids_Paris4_C.npz` if present; it is scored in the report for reference only.)

## umbilicus/ (band umbilici for scrolls without a published one)

`umb_band_PHerc<id>[_z...].json`: `{"control_points": [{"x", "y", "z", "score"}, ...]}` in level-0 voxel coordinates of the scroll's 9.362 um (or 8.64 um for PHerc1218) volume, one point per 100 slices over the fitted band, estimated with `exp/e69_umb_band.py` (the organisers' `vc_gen_umbilicus` objective plus a Theil-Sen line through the band). Scrolls: 0191 (two bands), 0211, 0257, 0358 (two bands), 0800, 0813, 1218. The format is what `fit_spiral.py` reads; `kaggle/pcu-fit-new/template.py` looks for the file as `umb_<scroll>.json` in its Kaggle input dataset (rename when uploading); pass one to `exp/e73_vet.py` with `UMB_FILE=`.

## cache/

`pcu.data` stores downloaded zarr chunks under `$PCU_CACHE` (default `data/cache`); it is not part of the repository.
