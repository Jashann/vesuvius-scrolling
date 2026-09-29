"""TRACE v2 on Modal: native coarse distillation of ink_9um with all 56 real 9.362 um segments (PHerc0139/0814),
held out PHerc0841 x3 + PHerc0139 w043. Reuses kaggle/pcu-trace-train/pcu_trace_train.py (prepare/train).
Run: MODAL_PROFILE=<profile> modal run modal/trace_train.py"""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl", "libgl1", "libglib2.0-0")
         .pip_install("numpy", "scipy", "zarr==2.18.7", "numcodecs==0.15.1", "requests", "scikit-image", "tifffile",
                      "imagecodecs", "opencv-python-headless", "pynrrd", "s3fs", "cachetools", "edt", "pyyaml", "tqdm",
                      "einops", "timm", "fsspec", "aiohttp", "torch", "torchvision")
         .run_commands("git clone -q --filter=blob:none --depth 1 -b merge-ink-pipelines --single-branch "
                       "https://github.com/ScrollPrize/villa.git /src/villa-ink",
                       "git clone -q --filter=blob:none --depth 1 https://github.com/ScrollPrize/villa.git /src/villa",
                       "mkdir -p /code && ln -s /src/villa-ink/ink-detection /code/ink && ln -s /src/villa/vesuvius/src /code/vesuvius_src")
         .add_local_dir(os.path.join(ROOT, "pcu"), "/code/pcu")
         .add_local_file(os.path.join(ROOT, "kaggle/pcu-trace-train/pcu_trace_train.py"), "/code/pcu_trace_train.py")
         .add_local_file(os.path.join(ROOT, "runs/trace_manifest.json"), "/code/trace_manifest.json"))
vol = modal.Volume.from_name("pcu-trace", create_if_missing=True)
app = modal.App("pcu-trace-v2", image=image)
ENV = {"TRACE_DIR": "/vol/trace", "TRACE_OUT": "/vol/out", "TRACE_CODE": "/code", "TRACE_MANIFEST": "/code/trace_manifest.json",
       "TRACE_NTRAIN": "56", "TRACE_EVAL": "1000", "TRACE_BS": "24"}


def _load():
    import runpy, sys
    for k, v in ENV.items():
        os.environ.setdefault(k, v)
    sys.path.insert(0, "/code")
    g = runpy.run_path("/code/pcu_trace_train.py", run_name="trace")
    for fn in ("prepare", "train", "evaluate", "load_ink9"):
        g[fn].__globals__.update(CODE="/code", W="/vol/out", D=os.environ["TRACE_DIR"])
    os.makedirs("/vol/out", exist_ok=True); os.makedirs("/vol/trace", exist_ok=True)
    return g


@app.function(cpu=8, memory=32768, volumes={"/vol": vol}, timeout=4 * 3600, ephemeral_disk=524288)
def prep():
    import shutil, glob
    os.environ["TRACE_DIR"] = "/tmp/trace"  # memmap writes on local disk, then copy into the volume
    g = _load()
    g["prepare"].__globals__.update(D="/tmp/trace")
    g["prepare"]()
    for f in glob.glob("/tmp/trace/*.npy") + glob.glob("/tmp/trace/*.pth"):
        shutil.copy(f, "/vol/trace/")
    vol.commit()
    print("copied", len(os.listdir("/vol/trace")), "files")


@app.function(gpu="L4", cpu=8, memory=49152, volumes={"/vol": vol}, timeout=14 * 3600)
def train(name: str, lr: float, steps: int):
    g = _load()
    vol.reload()
    g["train"](0, name, lr, steps, True)
    vol.commit()


@app.local_entrypoint()
def main(skip_prep: bool = False):
    if not skip_prep:
        prep.remote()
    runs = [train.spawn("v2A", 2e-5, 14000), train.spawn("v2B", 6e-5, 14000)]
    for r in runs:
        r.get()
