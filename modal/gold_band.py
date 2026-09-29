"""Generate a dense PCU gold constraint file for a Paris 4 z-band on Modal CPU (same generator as the released file:
exp/e13_band.py with the recto vote), one slice every 16 level-2 slices, merged into one VC3D document.
Run: MODAL_PROFILE=finnestapp modal run modal/gold_band.py --z0 11000 --z1 12000   (z in level-2 units)
Out: runs/gold_bands/pcu_gold_band<z0>.json"""
import os, modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "scipy", "zarr==2.18.7", "numcodecs==0.15.1", "requests", "scikit-image", "networkx",
                      "fsspec", "aiohttp", "s3fs", "tifffile", "imagecodecs")
         .add_local_dir(os.path.join(ROOT, "pcu"), "/work/pcu")
         .add_local_file(os.path.join(ROOT, "exp/e13_band.py"), "/work/exp/e13_band.py")
         .add_local_file(os.path.join(ROOT, "exp/e63_merge_gold.py"), "/work/exp/e63_merge_gold.py"))
app = modal.App("pcu-gold-band", image=image)


@app.function(cpu=8, memory=32768, timeout=4 * 3600)
def run(z0: int, z1: int) -> dict:
    import subprocess, glob, json
    # e13_band takes level-3 z; STEP is the level-3 stride (8 -> 16 level-2 slices, as in the original band file)
    cmd = (f"cd /work && PCU_CACHE=/tmp/cache OUT=/work/runs/constraints STEP=8 python -u exp/e13_band.py {z0 // 2} {z1 // 2} 2 && "
           f"python exp/e63_merge_gold.py /work/merged.json '/work/runs/constraints/*_gold.json'")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors="replace")
    out = {"log": r.stdout[-8000:] + r.stderr[-4000:], "files": len(glob.glob("/work/runs/constraints/*_gold.json"))}
    if os.path.exists("/work/merged.json"):
        out["merged"] = open("/work/merged.json").read()
    return out


@app.local_entrypoint()
def main(z0: int = 11000, z1: int = 12000):
    o = run.remote(z0, z1)
    print(o["log"][-3000:]); print("slice files:", o["files"])
    if "merged" in o:
        os.makedirs("runs/gold_bands", exist_ok=True)
        p = f"runs/gold_bands/pcu_gold_band{z0}.json"; open(p, "w").write(o["merged"]); print("saved", p, len(o["merged"]))
