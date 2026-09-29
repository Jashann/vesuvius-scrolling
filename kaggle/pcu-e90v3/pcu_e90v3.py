# Kaggle wrapper for E90 with TRACE-v3 (PHerc1447 positive control). Sets up the repo layout from the pcu-src,
# pcu-trace-ckpt and pcu-trace-ckpt-v3 datasets, then runs the experiment script inlined below.
import os, sys, glob, subprocess
subprocess.run("pip install -q 'zarr==2.18.7' 'numcodecs==0.15.1' fsspec aiohttp tifffile imagecodecs pynrrd s3fs cachetools edt pyyaml", shell=True)
CODE = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/pcu/__init__.py", recursive=True)[0]))
W = "/kaggle/working/repo"
for d in ("exp", "ext/villa-ink", "ext/villa/vesuvius", "runs/kaggle/pcu-trace-train", "runs/modal_v2", "runs/modal_v3", "runs"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
for a, b in ((f"{CODE}/pcu", f"{W}/pcu"), (f"{CODE}/ink", f"{W}/ext/villa-ink/ink-detection"), (f"{CODE}/vesuvius_src", f"{W}/ext/villa/vesuvius/src")):
    if not os.path.exists(b): os.symlink(a, b)
for p in glob.glob("/kaggle/input/**/s[AB]_best.pth", recursive=True):
    d = f"{W}/runs/kaggle/pcu-trace-train/{os.path.basename(p)}"
    if not os.path.exists(d): os.symlink(p, d)
for p in glob.glob("/kaggle/input/**/v3[AB]_best.pth", recursive=True):
    d = f"{W}/runs/modal_v3/{os.path.basename(p)}"
    if not os.path.exists(d): os.symlink(p, d)
for p in glob.glob("/kaggle/input/**/trace_manifest.json", recursive=True):
    d = f"{W}/runs/trace_manifest.json"
    if not os.path.exists(d): os.symlink(p, d)
os.environ["PCU_CACHE"] = "/tmp/cache"
os.chdir(W)
SCRIPT = r"""
'''E90: positive-control harness at the organisers' PHerc1447 ink site (8.64 um, letters found 2026-09-24 near
x 4144, y 2742, z 12557; the published segment 20250702235910-auto_grown passes 8.6 vox from it, at surface-volume
pixel about (row 1340, col 1720)). Scores every detector we have on a 1.6 cm crop around the site: native 8.64
and resampled to 9.6 um, forward and reversed depth, depth offsets -4..+4. Reports the site's 3 mm-disk mean
probability and its percentile among all 3 mm disks in the crop, per detector and variant. Also renders a TRACE
sA map of a PHerc0139 w043 crop (known text at 9.362 um) as a positive for the lattice test (E91).
Runs on a GPU box with the repo layout (pcu/, ext/, runs/kaggle/pcu-trace-train/, runs/modal_v2/).'''
import sys, os, json, time, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ext"))
sys.path[:0] = [f"{root}/villa-ink/ink-detection", f"{root}/villa/vesuvius/src"]
import numpy as np, torch, imageio, requests, zarr
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from pcu import data
from koine_machines.inference.infer import build_repo_training_model_bundle
from vesuvius.image_proc.intensity.normalization import normalize_robust

OUT = "runs/e90"; os.makedirs(OUT, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
HF = "https://huggingface.co"
SV = "PHerc1447/segments/20250702235910-auto_grown_20250702235910292/surface-volumes/8.64um-1.2m-116keV-volume-20250521151220.zarr"
SITE = (1340, 1720); HALF = 900
T0 = time.time()


def get(url, dst):
    if not os.path.exists(dst):
        subprocess.run(f"curl -sL --retry 5 -o '{dst}' '{url}'", shell=True, check=True)
    return dst


CK = {"base": get(f"{HF}/scrollprize/ink_9um/resolve/main/hybrid_3d2d-seed42/step-075000.pth", "/tmp/ink9.pth"),
      **{k: p for k, p in {"sA": "runs/kaggle/pcu-trace-train/sA_best.pth", "sB": "runs/kaggle/pcu-trace-train/sB_best.pth",
                           "v2A": "runs/modal_v2/v2A_best.pth", "v3A": "runs/modal_v3/v3A_best.pth", "v3B": "runs/modal_v3/v3B_best.pth"}.items() if os.path.exists(p)},
      "klavis": get(f"{HF}/domenicor046/ink9um-dense/resolve/main/dense9um-all7-step075000.pth", "/tmp/klavis.pth")}
nets = {}
for k, p in CK.items():
    try:
        nets[k] = build_repo_training_model_bundle(torch.load(p, map_location="cpu", weights_only=False), p).model.to(dev).eval()
    except Exception as e:
        print("skip", k, repr(e)[:160], flush=True)
get(f"{HF}/scrollprize/hecate/resolve/main/hecate_9.6um.pth", "/tmp/hecate_9.6um.pth")
get(f"{HF}/scrollprize/hecate/resolve/main/hecate.py", "/tmp/hecate.py")

a = data.open_array(f"{SV}/0")
r0, c0 = max(0, SITE[0] - HALF), max(0, SITE[1] - HALF)
sv = np.asarray(a[:, r0:r0 + 2 * HALF, c0:c0 + 2 * HALF])
site = (SITE[0] - r0, SITE[1] - c0)
print("sv crop", sv.shape, "site", site, f"[{time.time()-T0:.0f}s]", flush=True)
imageio.imwrite(f"{OUT}/site_mid.jpg", sv[sv.shape[0] // 2])


def predict(net, v, P=128, Z=17, stride=32):
    H, Wd = v.shape[1:]
    out = np.zeros((H, Wd), np.float32); wt = np.zeros((H, Wd), np.float32)
    w1 = np.hanning(P).astype(np.float32); w2 = np.maximum(np.outer(w1, w1), 1e-3)
    co = [(r, q) for r in range(0, max(H - P, 0) + 1, stride) for q in range(0, max(Wd - P, 0) + 1, stride)]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for k in range(0, len(co), 64):
            b = co[k:k + 64]
            x = np.stack([v[:, r:r + P, q:q + P] for r, q in b]).astype(np.float32)
            keep = (x[:, Z // 2] > 0).reshape(len(b), -1).mean(1) > 0.3
            if not keep.any():
                continue
            x = np.stack([normalize_robust(p) for p in x]).astype(np.float32)
            pr = torch.sigmoid(net(torch.from_numpy(x)[:, None].to(dev)).float())[:, 0].cpu().numpy()
            for (r, q), pp, kk in zip(b, pr, keep):
                if kk:
                    out[r:r + P, q:q + P] += pp * w2; wt[r:r + P, q:q + P] += w2
    return np.where(wt > 0, out / np.maximum(wt, 1e-6), 0)


def site_stat(p, valid, s, px_um):
    '''mean prob in a 3 mm disk at the site, and its percentile among all 3 mm disks of the crop'''
    rad = int(1500 / px_um)
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]; disk = (yy * yy + xx * xx <= rad * rad).astype(np.float32)
    num = fftconvolve(p * valid, disk, mode="same"); den = fftconvolve(valid.astype(np.float32), disk, mode="same")
    m = np.where(den > 0.7 * disk.sum(), num / np.maximum(den, 1), np.nan)
    v = m[s[0], s[1]]
    ok = np.isfinite(m)
    return round(float(v), 4) if np.isfinite(v) else None, round(float((m[ok] < v).mean()), 4) if np.isfinite(v) and ok.any() else None


results = []
variants = {"native8.64": (sv, 8.64, site)}
f = 8.64 / 9.6
svr = ndi.zoom(sv, (f, f, f), order=1)
variants["resamp9.6"] = (svr, 9.6, (int(site[0] * f), int(site[1] * f)))
for vname, (v, px, s) in variants.items():
    L = v.shape[0]; c = L // 2; valid = v[c] > 0
    for direction in ("fwd", "rev"):
        vv = v if direction == "fwd" else v[::-1]
        for off in (-4, -2, 0, 2, 4):
            z0 = c + off - 8
            if z0 < 0 or z0 + 17 > L:
                continue
            win = vv[z0:z0 + 17]
            for k, net in nets.items():
                p = predict(net, win)
                val, pct = site_stat(p, valid, s, px)
                r = dict(variant=vname, direction=direction, offset=off, model=k, site_mean=val, site_percentile=pct,
                         crop_p50=round(float(np.percentile(p[valid], 50)), 3), crop_p95=round(float(np.percentile(p[valid], 95)), 3))
                results.append(r); print(json.dumps(r), flush=True)
                if off == 0:
                    imageio.imwrite(f"{OUT}/{vname}_{direction}_{k}.png", (p * 255).astype(np.uint8))
    # hecate 9.6 via its CLI on the resampled stack (both directions)
    if vname == "resamp9.6":
        for direction, arr in (("fwd", v), ("rev", v[::-1])):
            zp = f"/tmp/site_{direction}.zarr"
            zarr.save_array(zp, np.ascontiguousarray(arr), chunks=(arr.shape[0], 256, 256))
            outp = f"{OUT}/hecate_{direction}.png"
            subprocess.run(f"python /tmp/hecate.py --checkpoint /tmp/hecate_9.6um.pth --input {zp} --array '' --spacing-um 9.6 "
                           f"--output {outp} --device {dev} --precision fp32 --batch-size 8", shell=True)
            if os.path.exists(outp):
                p = imageio.imread(outp).astype(np.float32) / 255
                p = p[:valid.shape[0], :valid.shape[1]]
                vv2 = valid[:p.shape[0], :p.shape[1]]
                val, pct = site_stat(p, vv2, s, px)
                r = dict(variant=vname, direction=direction, offset=0, model="hecate9.6", site_mean=val, site_percentile=pct,
                         crop_p50=round(float(np.percentile(p[vv2], 50)), 3), crop_p95=round(float(np.percentile(p[vv2], 95)), 3))
                results.append(r); print(json.dumps(r), flush=True)
json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
# site crops for the eye and for E91
for k in ("sA", "sB", "v2A", "v3A", "v3B", "klavis", "base"):
    pth = f"{OUT}/native8.64_fwd_{k}.png"
    if os.path.exists(pth):
        p = imageio.imread(pth)
        y, x = site; imageio.imwrite(f"{OUT}/site_crop_{k}.png", p[max(0, y - 600):y + 600, max(0, x - 600):x + 600])
y, x = site; imageio.imwrite(f"{OUT}/site_crop_mid.jpg", sv[sv.shape[0] // 2][max(0, y - 600):y + 600, max(0, x - 600):x + 600])

# positive for the lattice test: PHerc0139 w043 (native 9.362 um, known text), central crop, TRACE sA
try:
    man = json.load(open("runs/trace_manifest.json"))["test"]
    e = [t for t in man if "w043" in t["seg"]][0]
    b = data.open_array(f"{e['sv']}/0")
    L0, H, Wd = b.shape; zs = L0 // 2 - 8
    crop = np.asarray(b[zs:zs + 17, H // 2 - 1000:H // 2 + 1000, Wd // 2 - 1500:Wd // 2 + 1500])
    p = predict(nets["sA"], crop)
    imageio.imwrite(f"{OUT}/pos_0139w043_sA.png", (p * 255).astype(np.uint8))
    imageio.imwrite(f"{OUT}/pos_0139w043_mid.jpg", crop[8])
    print("positive 0139 w043 crop done", crop.shape, flush=True)
except Exception as ex:
    print("positive failed", repr(ex)[:200], flush=True)
print("E90 done", f"[{time.time()-T0:.0f}s]", flush=True)

"""
open(f"{W}/exp/e90_control1447.py", "w").write(SCRIPT)
sys.argv = ["e90_control1447.py"]
import runpy; runpy.run_path(f"{W}/exp/e90_control1447.py", run_name="__main__")
subprocess.run(f"cp -r {W}/runs/e90 /kaggle/working/e90", shell=True)
