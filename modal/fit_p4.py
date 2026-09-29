"""Replication of the Paris 4 spiral-fit experiment on Modal L4: fully automatic fit without (A2) and with (B3w10) PCU
gold constraints, new random seed and/or new z-band. Same fitter (villa 6bbe6e2), same public inputs, same config as
kaggle/pcu-fit-b3/pcu_fit_b3.py; the public dataset is downloaded once into a Modal volume.

Run:  MODAL_PROFILE=finnestapp modal run modal/fit_p4.py --band 8400,9400 --seed 2 --runs A2,B3w10
Out:  runs/modal_fit/<tag>/{grids_<name>.npz, <name>_satisfaction_metrics_fitted.json, fit_<name>.log}"""
import os, modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHA = "6bbe6e2"
DL = "https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4"
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl", "build-essential", "cmake", "libgl1", "libglib2.0-0")
         .pip_install("numpy", "requests", "tifffile", "scipy")
         .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh",
                       f"cd /opt && git clone -q https://github.com/ScrollPrize/villa.git && cd villa && git checkout -q {SHA}",
                       "cd /opt/villa/spiral-fitting && PATH=/root/.local/bin:$PATH uv sync -q")
         .add_local_file(os.path.join(ROOT, "runs/pcu_relative_windings_PHercParis4_gold.json"), "/opt/gold_full.json")
         .add_local_file(os.path.join(ROOT, "kaggle/ds-gold/pcu_gold_band.json"), "/opt/gold_band8400.json")
         .add_local_dir(os.path.join(ROOT, "runs/gold_bands"), "/opt/gold_bands"))
app = modal.App("pcu-fit-p4", image=image)
vol = modal.Volume.from_name("pcu-p4-dataset", create_if_missing=True)
D = "/data/ds"


@app.function(cpu=8, memory=16384, timeout=3 * 3600, volumes={"/data": vol})
def prep() -> str:
    import json, time, requests
    from concurrent.futures import ThreadPoolExecutor
    if os.path.exists(f"{D}/READY"):
        return "cached"
    os.makedirs(D, exist_ok=True); t0 = time.time()
    sess = requests.Session(); sess.mount("https://", requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=5))

    def get(url, path):
        if os.path.exists(path) and os.path.getsize(path) > 0: return True
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for _ in range(5):
            try:
                with sess.get(url, stream=True, timeout=600) as r:
                    if r.status_code == 404: return False
                    r.raise_for_status()
                    with open(path + ".part", "wb") as f:
                        for b in r.iter_content(1 << 22): f.write(b)
                os.replace(path + ".part", path); return True
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
        for f in files: big.append((f"{DL}/lasagna_inputs/{store}/{f}", f"{D}/lasagna_inputs/{store}/{f}"))
    with ThreadPoolExecutor(5) as ex: print("lasagna", list(ex.map(lambda a: get(*a), big)), flush=True)
    for f in ("verification.json", "mask_report.json", "batch_verification.json"):
        get(f"{DL}/lasagna_inputs/{f}", f"{D}/lasagna_inputs/{f}")
    os.makedirs(f"{D}/verified_patches", exist_ok=True)
    TB = f"{DL}/tracks/2um_ds2_ps256_surf_v2.dbm"
    tj = [(f"{TB}.db", f"{D}/tracks/2um_ds2_ps256_surf_v2.dbm.db"), (f"{TB}.crossings.npz", f"{D}/tracks/2um_ds2_ps256_surf_v2.dbm.crossings.npz")]
    tj += [(f"{TB}.vctracks/{f}", f"{D}/tracks/2um_ds2_ps256_surf_v2.dbm.vctracks/{f}") for f in
           ("arclengths.f64", "coordinates.i32", "family_codes.i8", "header.bin", "metadata.json", "offsets.i64", "source_ids.u64", "tortuosities.f64", "z_bounds.i32")]
    with ThreadPoolExecutor(6) as ex: print("tracks", list(ex.map(lambda a: get(*a), tj)), flush=True)
    open(f"{D}/READY", "w").write("ok"); vol.commit()
    return f"downloaded in {time.time()-t0:.0f}s"


@app.function(gpu="L4", cpu=8, memory=32768, timeout=5 * 3600, volumes={"/data": vol}, ephemeral_disk=524288)
def fit(name: str, z0: int, z1: int, seed: int, steps: int, use_gold: bool, w_radius: float, w_dt: float, gold_path: str = "/opt/gold_band8400.json") -> dict:
    import json, subprocess, shutil, glob, re, time
    import numpy as np, tifffile
    t0 = time.time(); R = f"/tmp/ds_{name}"; os.makedirs(R, exist_ok=True)
    for e in os.listdir(D):
        if e in ("relative_windings.json", "abs_winding.json", "same_windings.json", "READY"): continue
        if not os.path.exists(f"{R}/{e}"): os.symlink(f"{D}/{e}", f"{R}/{e}")
    for f in ("abs_winding.json", "same_windings.json"): shutil.copy(f"{D}/{f}", f"{R}/{f}")
    if use_gold:
        gold = json.load(open(gold_path))
        gold["collections"] = {k: c for k, c in gold["collections"].items() if all(z0 <= p["p"][2] < z1 for p in c["points"].values())}
        gold["vc_pointcollections_json_version"] = "1"; ng = len(gold["collections"])
        json.dump(gold, open(f"{R}/relative_windings.json", "w"))
    else:
        shutil.copy(f"{D}/relative_windings.json", f"{R}/relative_windings.json"); ng = 0
    json.dump({"schema_version": 1, "name": f"P4{name}", "voxel_size_um": 9.6, "spiral_outward_sense": "CW"}, open(f"{R}/spiral-scroll.json", "w"))
    cfg = {"z_begin": z0, "z_end": z1, "optimizer_num_training_steps": steps, "optimizer_random_seed": seed,
           "input_use_fibers": False, "input_use_pcl_drawn_control_points": False, "input_use_fiber_directions": False,
           "input_use_tracks": True, "input_use_verified_patches": False,
           "input_use_pcl_relative": use_gold, "input_use_pcl_absolute": False, "input_use_pcl_same_winding": False,
           "input_use_winding_inference": False}
    if use_gold:
        cfg.update(sample_count_unattached_pcls_per_step=840, loss_weight_unattached_pcl_radius=w_radius, loss_weight_unattached_pcl_dt=w_dt)
    out = f"/tmp/out_{name}"; os.makedirs(out, exist_ok=True)
    env = dict(os.environ, FIT_SPIRAL_OUT_DIR=out, FIT_SPIRAL_CACHE_DIR=f"/tmp/cache_{name}", WANDB_MODE="disabled",
               FIT_SPIRAL_CONFIG_OVERRIDES=json.dumps(cfg), PATH="/root/.local/bin:" + os.environ["PATH"])
    with open(f"/tmp/fit_{name}.log", "w") as lf:
        rc = subprocess.run("cd /opt/villa/spiral-fitting && .venv/bin/python fit_spiral.py --dataset " + R, shell=True, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    log = open(f"/tmp/fit_{name}.log", errors="replace").read()
    res = {"name": name, "rc": rc, "gold_ladders": ng, "cfg": cfg, "log_tail": log[-6000:], "log": log[-200000:], "seconds": time.time() - t0}
    allm = [m for m in glob.glob(f"{out}/**/w*", recursive=True) if os.path.exists(f"{m}/x.tif")]
    meshes = sorted(m for m in allm if re.fullmatch(r"w\d{3}", os.path.basename(m))) or \
             sorted(m for m in allm if re.fullmatch(r"w\d{3}_.*", os.path.basename(m)) and "spliced" not in os.path.basename(m))
    if meshes:
        grids = {f"w{int(os.path.basename(m)[1:4]):03d}": np.stack([tifffile.imread(f"{m}/{a}.tif") for a in "xyz"]).astype(np.float32) for m in meshes}
        np.savez_compressed(f"/tmp/grids_{name}.npz", **grids); res["grids"] = open(f"/tmp/grids_{name}.npz", "rb").read(); res["meshes"] = len(meshes)
    for mf in glob.glob(f"{out}/**/*satisfaction_metrics*fitted*.json", recursive=True)[:1]:
        res["metrics"] = open(mf).read()
    return res


@app.local_entrypoint()
def main(band: str = "8400,9400", seed: int = 2, runs: str = "A2,B3w10", steps: int = 12000, tag: str = "", gold: str = "/opt/gold_band8400.json"):
    """gold: /opt/gold_band8400.json (the dense band file the original fits used), /opt/gold_full.json (released
    whole-scroll file, two slices per 64) or /opt/gold_bands/<file> (a band file generated by modal/gold_band.py)."""
    z0, z1 = [int(x) for x in band.split(",")]; tag = tag or f"z{z0}_s{seed}"
    print("prep:", prep.remote(), flush=True)
    spec = {"A2": (False, 0.0, 0.0), "B3w10": (True, 10.0, 10.0), "B4w20": (True, 20.0, 10.0), "B2": (True, 2.0, 4.0)}
    names = runs.split(","); args = [(n, z0, z1, seed, steps, *spec[n], gold) for n in names]
    outd = f"runs/modal_fit/{tag}"; os.makedirs(outd, exist_ok=True)
    for r in fit.starmap(args):
        n = r["name"]; open(f"{outd}/fit_{n}.log", "w").write(r["log"])
        if "grids" in r: open(f"{outd}/grids_{n}.npz", "wb").write(r["grids"])
        if "metrics" in r: open(f"{outd}/{n}_satisfaction_metrics_fitted.json", "w").write(r["metrics"])
        print(f"{n}: rc {r['rc']} meshes {r.get('meshes')} gold ladders {r['gold_ladders']} in {r['seconds']/60:.0f} min\n{r['log_tail'][-1500:]}", flush=True)
