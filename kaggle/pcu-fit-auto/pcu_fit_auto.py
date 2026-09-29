# PCU fully-automatic experiment (no verified patches, no human windings, no winding model): tracks + lasagna
# PCU experiment: does the ScrollPrize spiral fit need human winding annotations if it gets
# certified automatic constraints instead?  PHerc Paris 4, z (level-2) band [Z0, Z1).
#   A: no human winding annotations, no winding-model inference (automatic inputs + verified patches)
#   B: A + PCU gold constraints as the relative-winding document
#   C: default inputs (human relative/absolute/same-winding annotations + winding inference)
# Score: slips per wrap along human relative-winding ladders in the band (held out for A and B).
import subprocess, sys, os, time, json, re, glob, shutil
from concurrent.futures import ThreadPoolExecutor

Z0, Z1 = int(os.environ.get("Z0", 8400)), int(os.environ.get("Z1", 9400))
STEPS = int(os.environ.get("STEPS", 12000))
SHA = "6bbe6e2"
DL = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4"
W = "/kaggle/working"
T0 = time.time()


def sh(c, t=36000, log=None):
    print(f"\n$ {c}  [{time.time()-T0:.0f}s]", flush=True)
    if log:
        with open(log, "w") as f:
            return subprocess.run(c, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=t).returncode
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return r.returncode


# ---------- environment
sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":/usr/local/bin:" + os.environ["PATH"]
sh(f"cd /tmp && git clone https://github.com/ScrollPrize/villa.git && cd villa && git checkout -q {SHA} && git log -1 --format='%h %cd'")
assert sh("cd /tmp/villa/spiral-fitting && uv sync -q") == 0
PY = "/tmp/villa/spiral-fitting/.venv/bin/python"
import requests

# ---------- data
D = "/tmp/ds"
os.makedirs(D, exist_ok=True)
sess = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=5)
sess.mount("https://", adapter)


def get(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for _ in range(5):
        try:
            with sess.get(url, stream=True, timeout=600) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(path + ".part", "wb") as f:
                    for b in r.iter_content(1 << 22):
                        f.write(b)
            os.replace(path + ".part", path)
            return True
        except Exception as e:
            print("retry", url, e, flush=True); time.sleep(3)
    return False


for f in ("umbilicus.json", "abs_winding.json", "relative_windings.json", "same_windings.json"):
    get(f"{DL}/{f}", f"{D}/{f}")
for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
    get(f"{DL}/outer_shell/{f}", f"{D}/outer_shell/{f}")
big = []
for store, files in (("las_008_grad_mag.ome.zarr.respool_g4", ("meta.json", "table.npy", "brick_coords.npy", "channel_0.u8")),
                     ("las_008_nx.ome.zarr.respool_g4_pair", ("meta.json", "table.npy", "brick_coords.npy", "channel_0.u8", "channel_1.u8"))):
    for f in files:
        big.append((f"{DL}/lasagna_inputs/{store}/{f}", f"{D}/lasagna_inputs/{store}/{f}"))
with ThreadPoolExecutor(5) as ex:
    print("lasagna downloads", list(ex.map(lambda a: get(*a), big)), f"{time.time()-T0:.0f}s", flush=True)
for f in ("verification.json", "mask_report.json", "batch_verification.json"):
    get(f"{DL}/lasagna_inputs/{f}", f"{D}/lasagna_inputs/{f}")
# winding inference (variant C only)
man = f"{D}/winding_inference/manifest.json"
get(f"{DL}/winding_inference/manifest.json", man)
for f in ("mask_report.json", "verification.json"):
    get(f"{DL}/winding_inference/{f}", f"{D}/winding_inference/{f}")
m = json.load(open(man))
wi = [(f"{DL}/winding_inference/{s['name']}/{a['file']}", f"{D}/winding_inference/{s['name']}/{a['file']}") for s in m["shards"] for a in s["arrays"].values()]
wi += [(f"{DL}/winding_inference/{s['name']}/manifest.json", f"{D}/winding_inference/{s['name']}/manifest.json") for s in m["shards"]]
with ThreadPoolExecutor(16) as ex:
    list(ex.map(lambda a: get(*a), wi))
# automatic surface-prediction tracks (skeletonised m-prediction), packed store + dbm + crossings
TB = f"{DL}/tracks/2um_ds2_ps256_surf_v2.dbm"
tj = [(f"{TB}.db", f"{D}/tracks/2um_ds2_ps256_surf_v2.dbm.db"), (f"{TB}.crossings.npz", f"{D}/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz")]
tj += [(f"{TB}.vctracks/{f}", f"{D}/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks/{f}") for f in
       ("arclengths.f64", "coordinates.i32", "family_codes.i8", "header.bin", "metadata.json", "offsets.i64", "source_ids.u64", "tortuosities.f64", "z_bounds.i32")]
with ThreadPoolExecutor(6) as ex:
    print("tracks downloads", list(ex.map(lambda a: get(*a), tj)), f"{time.time()-T0:.0f}s", flush=True)
os.makedirs(f"{D}/verified_patches", exist_ok=True)
print("data ready", f"{time.time()-T0:.0f}s", flush=True)
sh(f"du -sh {D}; df -h /tmp | tail -1")

# ---------- PCU gold constraints (private Kaggle dataset), restricted to the band
gold_src = glob.glob("/kaggle/input/**/pcu_gold_band.json", recursive=True)
print("gold source", gold_src, flush=True)
gold = json.load(open(gold_src[0]))
gold["collections"] = {k: c for k, c in gold["collections"].items()
                       if all(Z0 <= p["p"][2] < Z1 for p in c["points"].values())}
print("gold ladders in band", len(gold["collections"]), flush=True)


def variant(name, rel_doc=None):
    R = f"/tmp/ds_{name}"
    os.makedirs(R, exist_ok=True)
    for e in os.listdir(D):
        if e in ("relative_windings.json", "abs_winding.json", "same_windings.json"):
            continue
        dst = f"{R}/{e}"
        if not os.path.exists(dst):
            os.symlink(f"{D}/{e}", dst)
    for f in ("abs_winding.json", "same_windings.json"):
        shutil.copy(f"{D}/{f}", f"{R}/{f}")
    if rel_doc is None:
        shutil.copy(f"{D}/relative_windings.json", f"{R}/relative_windings.json")
    else:
        json.dump(rel_doc, open(f"{R}/relative_windings.json", "w"))
    json.dump({"schema_version": 1, "name": f"P4{name}", "voxel_size_um": 9.6, "spiral_outward_sense": "CW"},
              open(f"{R}/spiral-scroll.json", "w"))
    return R


common = {"z_begin": Z0, "z_end": Z1, "optimizer_num_training_steps": STEPS,
          "input_use_fibers": False, "input_use_pcl_drawn_control_points": False,
          "input_use_tracks": True, "input_use_fiber_directions": False, "input_use_verified_patches": False,
          "input_use_winding_inference": False, "input_use_pcl_absolute": False, "input_use_pcl_same_winding": False}
runs = {
    "A2": (variant("A2"), dict(common, input_use_pcl_relative=False)),
    "B2": (variant("B2", gold), dict(common, input_use_pcl_relative=True)),
}


def launch(name, gpu):
    root, cfg = runs[name]
    out = f"/tmp/out_{name}"
    os.makedirs(out, exist_ok=True)
    env = (f"CUDA_VISIBLE_DEVICES={gpu} FIT_SPIRAL_OUT_DIR={out} FIT_SPIRAL_CACHE_DIR=/tmp/cache_{name} "
           f"WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES='{json.dumps(cfg)}'")
    return subprocess.Popen(f"cd /tmp/villa/spiral-fitting && {env} {PY} fit_spiral.py --dataset {root}",
                            shell=True, stdout=open(f"{W}/fit_{name}.log", "w"), stderr=subprocess.STDOUT)


order = ["A2,B2"]
for group in order:
    procs = {n: launch(n, i) for i, n in enumerate(group.split(","))}
    for n, p in procs.items():
        rc = p.wait()
        print(f"fit {n} rc={rc} [{time.time()-T0:.0f}s]", flush=True)
        sh(f"tail -25 {W}/fit_{n}.log")

# ---------- evaluation on human relative-winding ladders in the band
import numpy as np
import tifffile
from scipy.spatial import cKDTree
lad = []
for fname in ("abs_winding.json", "relative_windings.json"):
    for col in json.load(open(f"{D}/{fname}"))["collections"].values():
        P = sorted([p for p in col["points"].values() if p.get("wind_a") is not None], key=lambda p: p["wind_a"])
        if len(P) < 2:
            continue
        zz = np.array([p["p"][2] for p in P])
        if np.ptp(zz) > 2 or not (Z0 + 20 <= zz.mean() < Z1 - 20):
            continue
        lad.append((np.array([p["p"] for p in P]), np.array([p["wind_a"] for p in P], float)))
print("human ladders in band", len(lad), "pairs", sum(len(w) - 1 for _, w in lad), flush=True)
results = {}
for name in runs:
    allm = [m for m in glob.glob(f"/tmp/out_{name}/**/w*", recursive=True) if os.path.exists(f"{m}/x.tif")]
    meshes = sorted(m for m in allm if re.fullmatch(r"w\d{3}", os.path.basename(m)))
    if not meshes:
        meshes = sorted(m for m in allm if re.fullmatch(r"w\d{3}_.*", os.path.basename(m)) and "spliced" not in os.path.basename(m))
    print(name, "mesh dirs found", len(allm), "used", len(meshes), [os.path.basename(m) for m in allm[:6]], flush=True)
    if not meshes:
        results[name] = "no meshes"; continue
    grids = {}
    for m in meshes:
        w_ = int(os.path.basename(m)[1:4])
        grids[f"w{w_:03d}"] = np.stack([tifffile.imread(f"{m}/{a}.tif") for a in "xyz"]).astype(np.float32)
    np.savez_compressed(f"{W}/grids_{name}.npz", **grids)
    Pm, Wm = [], []
    for m in meshes:
        w = int(os.path.basename(m)[1:4])
        x, y, z = (tifffile.imread(f"{m}/{a}.tif") for a in "xyz")
        ok = (z > Z0 - 5) & (z < Z1 + 5) & (x > 0)
        Pm.append(np.stack([x[ok], y[ok], z[ok]], 1)); Wm.append(np.full(ok.sum(), w))
    Pm = np.concatenate(Pm); Wm = np.concatenate(Wm)
    print(name, "mesh coord ranges x", Pm[:, 0].min(), Pm[:, 0].max(), "y", Pm[:, 1].min(), Pm[:, 1].max(), "z", Pm[:, 2].min(), Pm[:, 2].max(), flush=True)
    allpts = np.concatenate([p for p, _ in lad])
    print("ladder coord ranges", allpts.min(0), allpts.max(0), flush=True)

    tree = cKDTree(Pm)
    slips = tot = far = 0
    for pts, w in lad:
        d, j = tree.query(pts)
        v = Wm[j].astype(float); v[d > 6] = np.nan
        far += int(np.sum(d > 6))
        dv = np.diff(v); dw = np.diff(w); ok = np.isfinite(dv)
        sg = np.sign(np.nanmedian(dv)) if ok.any() else 1
        slips += int(np.sum(sg * dv[ok] != dw[ok])); tot += int(ok.sum())
    results[name] = dict(slips=slips, pairs=tot, slip_rate=round(slips / max(tot, 1), 4), points_far_from_mesh=far, meshes=len(meshes))
    for mf in glob.glob(f"/tmp/out_{name}/**/*satisf*.json", recursive=True)[:3]:
        shutil.copy(mf, f"{W}/{name}_{os.path.basename(mf)}")
print("RESULTS", json.dumps(results, indent=1), flush=True)
json.dump(results, open(f"{W}/results.json", "w"), indent=1)
print("total", time.time() - T0)
