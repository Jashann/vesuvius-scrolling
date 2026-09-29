# PCU on the Grand-Prize-eligible scroll PHerc0826 (9.362 um), fully automatic spiral fit:
#   A0: tracks + lasagna normals/density + umbilicus (no human input of any kind)
#   B0: A0 + PCU gold winding constraints
# Exports the fitted winding grids for local rendering + ink detection.
import subprocess, os, time, json, re, glob, shutil
from concurrent.futures import ThreadPoolExecutor

Z0, Z1 = 8500, 9500
STEPS = 12000
SHA = "6bbe6e2"
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
LAS = "PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421"
TR = "https://dl.ash2txt.org/datasets/spiral_datasets/PHerc0826/20250821151701/tracks/PHerc0826_20250821151701_surface_m7_L0_th0.2.dbm"
W = "/kaggle/working"
T0 = time.time()


def sh(c, t=36000):
    print(f"\n$ {c}  [{time.time()-T0:.0f}s]", flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    return r.returncode


sh("curl -LsSf https://astral.sh/uv/install.sh | sh")
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":/usr/local/bin:" + os.environ["PATH"]
sh(f"cd /tmp && git clone https://github.com/ScrollPrize/villa.git && cd villa && git checkout -q {SHA}")
assert sh("cd /tmp/villa/spiral-fitting && uv sync -q") == 0
PY = "/tmp/villa/spiral-fitting/.venv/bin/python"
sh("pip install -q awscli")
import requests
D = "/tmp/ds"
os.makedirs(f"{D}/lasagna_inputs", exist_ok=True)
# lasagna stores (OME-Zarr, level 2) -> pack into resident pools
for ch in ("nx", "ny", "grad_mag"):
    dst = f"{D}/lasagna_inputs/PHerc0826_{ch}.ome.zarr"
    sh(f"aws s3 sync --no-sign-request --only-show-errors s3://vesuvius-challenge-open-data/{LAS}/PHerc0826_{ch}.ome.zarr/2 {dst}/2")
    for f in (".zattrs", ".zgroup"):
        r = requests.get(f"{S3}/{LAS}/PHerc0826_{ch}.ome.zarr/{f}", timeout=120)
        if r.ok:
            open(f"{dst}/{f}", "wb").write(r.content)
sh(f"du -sh {D}/lasagna_inputs/*")
assert sh(f"cd /tmp/villa/spiral-fitting && {PY} pack_resident_pools.py {D}/lasagna_inputs --normal-group 2 --io-threads 32") == 0
# tracks
sess = requests.Session()


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


idx = sess.get(TR + ".vctracks/", timeout=120).text
vfiles = [n for n in re.findall(r'href="([^"/]+)"', idx) if not n.startswith("..")]
tj = [(TR, f"{D}/tracks/PHerc0826.dbm"), (TR + ".crossings.npz", f"{D}/tracks/PHerc0826.dbm.crossings.npz")]
tj += [(f"{TR}.vctracks/{f}", f"{D}/tracks/PHerc0826.dbm.vctracks/{f}") for f in vfiles]
with ThreadPoolExecutor(6) as ex:
    print("tracks", list(ex.map(lambda a: get(*a), tj)), vfiles, f"{time.time()-T0:.0f}s", flush=True)
# umbilicus (control points in base voxels)
u = requests.get(f"{S3}/PHerc0826/representations/umbilicus/20250821151701-umbilicus-20260808113303.json", timeout=120).json()
json.dump(u, open(f"{D}/umbilicus.json", "w"))
empty = {"vc_pointcollections_json_version": "1", "collections": {}}
gold = json.load(open(glob.glob("/kaggle/input/**/pcu_gold_0826.json", recursive=True)[0]))
gold["vc_pointcollections_json_version"] = "1"
gold["collections"] = {k: c for k, c in gold["collections"].items() if all(Z0 <= p["p"][2] < Z1 for p in c["points"].values())}
print("gold ladders in band", len(gold["collections"]), flush=True)


def variant(name, rel):
    R = f"/tmp/ds_{name}"
    os.makedirs(R, exist_ok=True)
    for e in os.listdir(D):
        if not os.path.exists(f"{R}/{e}"):
            os.symlink(f"{D}/{e}", f"{R}/{e}")
    for f in ("abs_winding.json", "same_windings.json"):
        json.dump(empty, open(f"{R}/{f}", "w"))
    json.dump(rel, open(f"{R}/relative_windings.json", "w"))
    json.dump({"schema_version": 1, "name": f"PHerc0826{name}", "voxel_size_um": 9.362, "spiral_outward_sense": "CW",
               "normal_zarr_group": "2", "lasagna_scale": 4,
               "paths": {"tracks_dbm": "tracks/PHerc0826.dbm",
                         "normal_x": "lasagna_inputs/PHerc0826_nx.ome.zarr",
                         "normal_y": "lasagna_inputs/PHerc0826_ny.ome.zarr",
                         "gradient_magnitude": "lasagna_inputs/PHerc0826_grad_mag.ome.zarr"}},
              open(f"{R}/spiral-scroll.json", "w"))
    return R


common = {"z_begin": Z0, "z_end": Z1, "optimizer_num_training_steps": STEPS,
          "input_use_fibers": False, "input_use_pcl_drawn_control_points": False, "input_use_tracks": True,
          "input_use_fiber_directions": False, "input_use_verified_patches": False, "input_use_winding_inference": False,
          "input_use_pcl_absolute": False, "input_use_pcl_same_winding": False, "input_use_outer_shell": False}
runs = {"A0": (variant("A0", empty), dict(common, input_use_pcl_relative=False)),
        "B0": (variant("B0", gold), dict(common, input_use_pcl_relative=True))}
procs = {}
for i, (name, (root, cfg)) in enumerate(runs.items()):
    env = (f"CUDA_VISIBLE_DEVICES={i} FIT_SPIRAL_OUT_DIR=/tmp/out_{name} FIT_SPIRAL_CACHE_DIR=/tmp/cache_{name} "
           f"WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES='{json.dumps(cfg)}'")
    procs[name] = subprocess.Popen(f"cd /tmp/villa/spiral-fitting && {env} {PY} fit_spiral.py --dataset {root}",
                                   shell=True, stdout=open(f"{W}/fit_{name}.log", "w"), stderr=subprocess.STDOUT)
for name, p in procs.items():
    print(f"fit {name} rc={p.wait()} [{time.time()-T0:.0f}s]", flush=True)
    sh(f"grep -v Warning {W}/fit_{name}.log | tail -30")
import numpy as np, tifffile
for name in runs:
    allm = [m for m in glob.glob(f"/tmp/out_{name}/**/w*", recursive=True) if os.path.exists(f"{m}/x.tif")]
    ms = sorted(m for m in allm if re.fullmatch(r"w\d{3}", os.path.basename(m)))
    grids = {os.path.basename(m): np.stack([tifffile.imread(f"{m}/{a}.tif") for a in "xyz"]).astype(np.float32) for m in ms}
    np.savez_compressed(f"{W}/grids_{name}.npz", **grids)
    for mf in glob.glob(f"/tmp/out_{name}/**/*satisf*.json", recursive=True)[:2]:
        shutil.copy(mf, f"{W}/{name}_{os.path.basename(mf)}")
    print(name, "windings exported", len(grids), flush=True)
print("total", time.time() - T0)
