"""Run E73 hotspot vetting (depth profile + neighbours) on Modal L4 from a fit grid stored in Volume pcu-kout.
Run: MODAL_PROFILE=<p> modal run --detach modal/e73_run.py --scroll PHerc0800 --grids n800/grids_n800.npz
     --umb umb_PHerc0800.json --jobs "w048:0.1:0.3:w047,w049:s800w048;w108:..." """
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl", "libgl1", "libglib2.0-0")
         .pip_install("numpy", "scipy", "zarr==2.18.7", "numcodecs==0.15.1", "requests", "scikit-image", "tifffile",
                      "imagecodecs", "opencv-python-headless", "pynrrd", "s3fs", "cachetools", "edt", "pyyaml", "tqdm",
                      "einops", "timm", "fsspec", "aiohttp", "imageio", "torch", "torchvision")
         .add_local_file(os.path.join(ROOT, "exp/e73_vet.py"), "/work/exp/e73_vet.py"))
kin = modal.Volume.from_name("pcu-kinputs", create_if_missing=True)
kout = modal.Volume.from_name("pcu-kout", create_if_missing=True)
app = modal.App("pcu-e73", image=image)


@app.function(gpu="L4", cpu=8, memory=49152, volumes={"/vol/in": kin, "/vol/out": kout}, timeout=4 * 3600,
              ephemeral_disk=524288)
def run(scroll: str, grids: str, umb: str, jobs: str):
    import subprocess, glob
    os.chdir("/work")
    for d in ("ext/villa-ink", "ext/villa/vesuvius", "runs/kaggle/pcu-trace-train", "runs/modal_v2", "runs/e73"):
        os.makedirs(d, exist_ok=True)
    code = glob.glob("/vol/in/**/pcu/__init__.py", recursive=True)[0].rsplit("/", 2)[0]
    for a, b in ((f"{code}/pcu", "pcu"), (f"{code}/ink", "ext/villa-ink/ink-detection"), (f"{code}/vesuvius_src", "ext/villa/vesuvius/src")):
        if not os.path.exists(b): os.symlink(a, b)
    for nm, dst in (("sA_best.pth", "runs/kaggle/pcu-trace-train/sA_best.pth"), ("sB_best.pth", "runs/kaggle/pcu-trace-train/sB_best.pth"),
                    ("v3B_best.pth", "runs/modal_v2/v2A_best.pth")):
        for p in glob.glob(f"/vol/in/**/{nm}", recursive=True)[:1]:
            os.symlink(p, dst)
    env = dict(os.environ, PCU_CACHE="/tmp/cache", UMB_FILE=glob.glob(f"/vol/in/**/{umb}", recursive=True)[0])
    G = f"/vol/out/{grids}"
    for job in jobs.split(";"):
        w, f0, f1, nb, tag = job.split(":")
        r = subprocess.run(f"python exp/e73_vet.py {scroll} {G} {w} {f0} {f1} {nb} {tag}", shell=True, env=env,
                           capture_output=True, text=True, errors="replace")
        print(r.stdout[-3000:], r.stderr[-1500:], flush=True)
    os.makedirs("/vol/out/e73", exist_ok=True)
    subprocess.run("rm -f runs/e73/*_sv41.npy; cp -r runs/e73/* /vol/out/e73/", shell=True); kout.commit()
    print("E73 copied", os.listdir("/vol/out/e73"), flush=True)


@app.local_entrypoint()
def main(scroll: str, grids: str, umb: str, jobs: str):
    run.remote(scroll, grids, umb, jobs)
