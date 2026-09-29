# Two z-bands of one eligible scroll per kernel (one band per T4): fully automatic spiral fit + PCU gold
# constraints (10x strip sampling, weights 20/10, best on PHerc0826). Exports winding grids per band.
import subprocess, os, time, json, re, glob, shutil
from concurrent.futures import ThreadPoolExecutor

STEPS = 20000
SHA = "6bbe6e2"
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SC = "PHerc0826"
SCR = {SC: dict(ts="20250821151701", umb="20250821151701-umbilicus-20260808113303.json", z=(0, 0))}
BANDS = [("b2", 9500, 10500, "pcu_gold_0826_b2.json"), ("b0", 7500, 8500, "pcu_gold_0826_b0.json")]
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




def prepare(sc):
    c = SCR[sc]; ts = c["ts"]
    LAS = f"{sc}/representations/predictions/lasagna/{ts}-lasagna-20260419180421"
    TR = f"https://dl.ash2txt.org/datasets/spiral_datasets/{sc}/{ts}/tracks/{sc}_{ts}_surface_m7_L0_th0.2.dbm"
    D = f"/tmp/ds_{sc}"
    os.makedirs(f"{D}/lasagna_inputs", exist_ok=True)
    for ch in ("nx", "ny", "grad_mag"):
        dst = f"{D}/lasagna_inputs/{sc}_{ch}.ome.zarr"
        sh(f"aws s3 sync --no-sign-request --only-show-errors s3://vesuvius-challenge-open-data/{LAS}/{sc}_{ch}.ome.zarr/2 {dst}/2")
        for f in (".zattrs", ".zgroup"):
            r = requests.get(f"{S3}/{LAS}/{sc}_{ch}.ome.zarr/{f}", timeout=120)
            if r.ok:
                open(f"{dst}/{f}", "wb").write(r.content)
    assert sh(f"cd /tmp/villa/spiral-fitting && {PY} pack_resident_pools.py {D}/lasagna_inputs --normal-group 2 --io-threads 32") == 0
    idx = sess.get(TR + ".vctracks/", timeout=120).text
    vfiles = [n for n in re.findall(r'href="([^"/]+)"', idx) if not n.startswith("..")]
    tj = [(TR, f"{D}/tracks/{sc}.dbm"), (TR + ".crossings.npz", f"{D}/tracks/{sc}.dbm.crossings.npz")]
    tj += [(f"{TR}.vctracks/{f}", f"{D}/tracks/{sc}.dbm.vctracks/{f}") for f in vfiles]
    with ThreadPoolExecutor(6) as ex:
        print(sc, "tracks", list(ex.map(lambda a: get(*a), tj)), f"{time.time()-T0:.0f}s", flush=True)
    u = requests.get(f"{S3}/{sc}/representations/umbilicus/{c['umb']}", timeout=120).json()
    json.dump(u, open(f"{D}/umbilicus.json", "w"))
    empty = {"vc_pointcollections_json_version": "1", "collections": {}}
    for f in ("abs_winding.json", "same_windings.json", "relative_windings.json"):
        json.dump(empty, open(f"{D}/{f}", "w"))
    json.dump({"schema_version": 1, "name": sc, "voxel_size_um": 9.362, "spiral_outward_sense": "CW",
               "normal_zarr_group": "2", "lasagna_scale": 4,
               "paths": {"tracks_dbm": f"tracks/{sc}.dbm",
                         "normal_x": f"lasagna_inputs/{sc}_nx.ome.zarr",
                         "normal_y": f"lasagna_inputs/{sc}_ny.ome.zarr",
                         "gradient_magnitude": f"lasagna_inputs/{sc}_grad_mag.ome.zarr"}},
              open(f"{D}/spiral-scroll.json", "w"))
    return D


D = prepare(SC)
procs = {}
for gi, (name, z0, z1, gname) in enumerate(BANDS):
    gf = glob.glob(f"/kaggle/input/**/{gname}", recursive=True)
    gold = json.load(open(gf[0])) if gf else {"collections": {}}
    gold["vc_pointcollections_json_version"] = "1"
    assert all(k.isdigit() for k in gold["collections"]), "fit_spiral needs integer collection keys"
    print(name, z0, z1, "gold ladders", len(gold["collections"]), flush=True)
    cfg = {"z_begin": z0, "z_end": z1, "optimizer_num_training_steps": STEPS,
           "input_use_fibers": False, "input_use_pcl_drawn_control_points": False, "input_use_tracks": True,
           "input_use_fiber_directions": False, "input_use_verified_patches": False, "input_use_winding_inference": False,
           "input_use_pcl_absolute": False, "input_use_pcl_same_winding": False, "input_use_outer_shell": False,
           "input_use_pcl_relative": bool(gold["collections"]), "sample_count_unattached_pcls_per_step": 840,
           "loss_weight_unattached_pcl_radius": 20.0, "loss_weight_unattached_pcl_dt": 10.0}
    R = f"/tmp/ds_{name}"
    os.makedirs(R, exist_ok=True)
    for e in os.listdir(D):
        if e != "relative_windings.json":
            os.symlink(f"{D}/{e}", f"{R}/{e}")
    json.dump(gold, open(f"{R}/relative_windings.json", "w"))
    env = (f"CUDA_VISIBLE_DEVICES={gi % int(os.environ.get('NGPU', '2'))} FIT_SPIRAL_OUT_DIR=/tmp/out_{name} FIT_SPIRAL_CACHE_DIR=/tmp/cache_{name} "
           f"WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES='{json.dumps(cfg)}'")
    procs[name] = subprocess.Popen(f"cd /tmp/villa/spiral-fitting && {env} {PY} fit_spiral.py --dataset {R}",
                                   shell=True, stdout=open(f"{W}/fit_{name}.log", "w"), stderr=subprocess.STDOUT)
for name, p in procs.items():
    print(f"fit {name} rc={p.wait()} [{time.time()-T0:.0f}s]", flush=True)
    sh(f"grep -v Warning {W}/fit_{name}.log | grep -i 'unattached\\|satisfied\\|Error' | tail -8")
runs = {name: None for name, *_ in BANDS}
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
