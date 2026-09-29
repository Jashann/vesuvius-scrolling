"""Run a Kaggle-style kernel script on Modal when the Kaggle GPU quota is exhausted. Inputs (extracted datasets)
live in Volume pcu-kinputs and appear at /kaggle/input; /kaggle/working is on the local ephemeral disk; at the
end everything in /kaggle/working except the survey's per-winding npz is copied to Volume pcu-kout/<name>.
Run: MODAL_PROFILE=<profile> modal run --detach modal/kernel_runner.py --script pcu_new_0800.py --name n800"""
import os
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "curl", "libgl1", "libglib2.0-0", "build-essential", "rsync")
         .pip_install("numpy", "scipy", "zarr==2.18.7", "numcodecs==0.15.1", "requests", "scikit-image", "tifffile",
                      "imagecodecs", "opencv-python-headless", "pynrrd", "s3fs", "cachetools", "edt", "pyyaml", "tqdm",
                      "einops", "timm", "fsspec", "aiohttp", "imageio", "torch", "torchvision", "awscli"))
kin = modal.Volume.from_name("pcu-kinputs", create_if_missing=True)
kout = modal.Volume.from_name("pcu-kout", create_if_missing=True)
app = modal.App("pcu-kernel-runner", image=image)


@app.function(gpu="L4", cpu=8, memory=49152, volumes={"/vol/in": kin, "/vol/out": kout}, timeout=12 * 3600,
              ephemeral_disk=524288)
def run(script: str, name: str):
    import subprocess, shutil, glob
    os.makedirs("/kaggle/working", exist_ok=True)
    if not os.path.exists("/kaggle/input"):
        os.symlink("/vol/in", "/kaggle/input")
    env = dict(os.environ, NGPU="1", PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"])
    sp = (glob.glob(f"/vol/in/**/{script}", recursive=True) + [f"/vol/in/{script}"])[0]
    r = subprocess.run(f"cd /kaggle/working && python {sp}", shell=True, env=env,
                       capture_output=True, text=True, errors="replace")
    open(f"/kaggle/working/{name}_kernel.log", "w").write(r.stdout + "\n--- stderr ---\n" + r.stderr)
    print(r.stdout[-6000:], r.stderr[-3000:], flush=True)
    dst = f"/vol/out/{name}"; os.makedirs(dst, exist_ok=True)
    subprocess.run(f"rsync -a --exclude '*_grid.npz' /kaggle/working/ {dst}/", shell=True)
    kout.commit()
    print("copied to", dst, os.listdir(dst)[:20], flush=True)


@app.local_entrypoint()
def main(script: str = "pcu_new_0800.py", name: str = "n800"):
    run.remote(script, name)
