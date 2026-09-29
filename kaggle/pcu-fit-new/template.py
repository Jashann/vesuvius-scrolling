# New eligible scrolls (no published umbilicus; ours from pcu.umbilicus.slice_centre_vc, E67): one
# fully automatic spiral fit per T4, then the TRACE survey of every 3rd winding of each fit in-kernel.
import subprocess, os, time, json, re, glob, shutil
from concurrent.futures import ThreadPoolExecutor

STEPS = 20000
SHA = "6bbe6e2"
S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
JOBS = __JOBS__
SURVEY_B64 = "IyBUUkFDRSBzdXJ2ZXkgb2YgYW4gZWxpZ2libGUgc2Nyb2xsOiBldmVyeSBmaXR0ZWQgd2luZGluZyAtPiBzbmFwIHRvIHNoZWV0IC0+IGlzb21ldHJpYyBmbGF0dGVuIC0+IHJlbmRlciBhdCB0aGUKIyBuYXRpdmUgOS4zNjIgdW0gLT4gcmVsZWFzZWQgaW5rXzl1bSBhbmQgVFJBQ0Ugc3R1ZGVudHMgc0Evc0IgaW4tcHJvY2Vzcy4gVGV4dC1wcmVzZW5jZSBzdGF0aXN0aWM6IHA5NSBvZiB0aGUKIyBwcm9iYWJpbGl0eSAoY2FsaWJyYXRlZDoga25vd24gdGV4dCAwLjY5LzAuNTQgZm9yIHNBL3NCIHZzIGJsYW5rIDAuMzQvMC4yMiksIHBsdXMgYmVzdCAyeDIgY20gd2luZG93IHA5NS4KaW1wb3J0IG9zLCBzeXMsIGpzb24sIHRpbWUsIGdsb2IsIHN1YnByb2Nlc3MKVDAgPSB0aW1lLnRpbWUoKQpXID0gb3MuZW52aXJvbi5nZXQoIlNVUlZFWV9XIiwgIi9rYWdnbGUvd29ya2luZyIpCkJVREdFVCA9IGZsb2F0KG9zLmVudmlyb24uZ2V0KCJCVURHRVRfSCIsICIxMC44IikpICogMzYwMApTQyA9IG9zLmVudmlyb24uZ2V0KCJTQyIsICJQSGVyYzA4MjYiKTsgRklUID0gb3MuZW52aXJvbi5nZXQoIkZJVCIsICJCMXcyMCIpOyBFVkVSWSA9IGludChvcy5lbnZpcm9uLmdldCgiRVZFUlkiLCAiMiIpKQoKCmRlZiBzaChjKToKICAgIHByaW50KGYiXG4kIHtjfSAgW3t0aW1lLnRpbWUoKS1UMDouMGZ9c10iLCBmbHVzaD1UcnVlKQogICAgciA9IHN1YnByb2Nlc3MucnVuKGMsIHNoZWxsPVRydWUsIGNhcHR1cmVfb3V0cHV0PVRydWUsIHRleHQ9VHJ1ZSwgZXJyb3JzPSJyZXBsYWNlIikKICAgIHByaW50KHIuc3Rkb3V0Wy0xNTAwOl0sIHIuc3RkZXJyWy0xNTAwOl0sIGZsdXNoPVRydWUpCgoKc2goInBpcCBpbnN0YWxsIC1xICd6YXJyPT0yLjE4LjcnICdudW1jb2RlY3M9PTAuMTUuMScgZnNzcGVjIGFpb2h0dHAgdGlmZmZpbGUgaW1hZ2Vjb2RlY3MgcHlucnJkIHMzZnMgY2FjaGV0b29scyBlZHQgcHl5YW1sIikKc2goImN1cmwgLXNMIC0tcmV0cnkgNSAtbyAvdG1wL2luazkucHRoIGh0dHBzOi8vaHVnZ2luZ2ZhY2UuY28vc2Nyb2xscHJpemUvaW5rXzl1bS9yZXNvbHZlL21haW4vaHlicmlkXzNkMmQtc2VlZDQyL3N0ZXAtMDc1MDAwLnB0aCIpCmltcG9ydCBudW1weSBhcyBucCwgdG9yY2gsIGltYWdlaW8KQ09ERSA9IG9zLnBhdGguZGlybmFtZShvcy5wYXRoLmRpcm5hbWUoZ2xvYi5nbG9iKCIva2FnZ2xlL2lucHV0LyoqL3BjdS9fX2luaXRfXy5weSIsIHJlY3Vyc2l2ZT1UcnVlKVswXSkpCnN5cy5wYXRoWzowXSA9IFtDT0RFLCBmIntDT0RFfS9pbmsiLCBmIntDT0RFfS92ZXN1dml1c19zcmMiXQpvcy5lbnZpcm9uWyJQQ1VfQ0FDSEUiXSA9ICIvdG1wL2NhY2hlIgpmcm9tIHBjdSBpbXBvcnQgZGF0YSwgc3VydmV5CmZyb20ga29pbmVfbWFjaGluZXMuaW5mZXJlbmNlLmluZmVyIGltcG9ydCBidWlsZF9yZXBvX3RyYWluaW5nX21vZGVsX2J1bmRsZQpmcm9tIHZlc3V2aXVzLmltYWdlX3Byb2MuaW50ZW5zaXR5Lm5vcm1hbGl6YXRpb24gaW1wb3J0IG5vcm1hbGl6ZV9yb2J1c3QKQ0sgPSB7ImJhc2UiOiAiL3RtcC9pbms5LnB0aCJ9CmZvciBwIGluIGdsb2IuZ2xvYigiL2thZ2dsZS9pbnB1dC8qKi9zW0EtRF1fYmVzdC5wdGgiLCByZWN1cnNpdmU9VHJ1ZSkgKyBnbG9iLmdsb2IoIi9rYWdnbGUvaW5wdXQvKiovdjNbQUJdX2Jlc3QucHRoIiwgcmVjdXJzaXZlPVRydWUpOgogICAgQ0tbb3MucGF0aC5iYXNlbmFtZShwKS5zcGxpdCgiXyIpWzBdXSA9IHAKZGV2ID0gImN1ZGEiCm5ldHMgPSB7azogYnVpbGRfcmVwb190cmFpbmluZ19tb2RlbF9idW5kbGUodG9yY2gubG9hZChwLCBtYXBfbG9jYXRpb249ImNwdSIsIHdlaWdodHNfb25seT1GYWxzZSksIHApLm1vZGVsLnRvKGRldikuZXZhbCgpCiAgICAgICAgZm9yIGssIHAgaW4gQ0suaXRlbXMoKX0KcHJpbnQoIm1vZGVscyIsIGxpc3QobmV0cyksIGZsdXNoPVRydWUpCkdSSURTID0gb3MuZW52aXJvbi5nZXQoIkdSSURTIikgb3IgZ2xvYi5nbG9iKGYiL2thZ2dsZS9pbnB1dC8qKi9ncmlkc197RklUfS5ucHoiLCByZWN1cnNpdmU9VHJ1ZSlbMF0KCgpkZWYgcHJlZGljdChuZXQsIHN2LCBQPTEyOCwgWj0xNywgc3RyaWRlPTY0KToKICAgIGMgPSBzdi5zaGFwZVswXSAvLyAyOyB6MCA9IGMgLSBaIC8vIDIKICAgIEgsIFdkID0gc3Yuc2hhcGVbMTpdCiAgICBvdXQgPSBucC56ZXJvcygoSCwgV2QpLCBucC5mbG9hdDMyKTsgd3QgPSBucC56ZXJvcygoSCwgV2QpLCBucC5mbG9hdDMyKQogICAgdzEgPSBucC5oYW5uaW5nKFApLmFzdHlwZShucC5mbG9hdDMyKTsgdzIgPSBucC5tYXhpbXVtKG5wLm91dGVyKHcxLCB3MSksIDFlLTMpCiAgICBjb29yZHMgPSBbKHIsIHEpIGZvciByIGluIHJhbmdlKDAsIG1heChIIC0gUCwgMCkgKyAxLCBzdHJpZGUpIGZvciBxIGluIHJhbmdlKDAsIG1heChXZCAtIFAsIDApICsgMSwgc3RyaWRlKV0KICAgIHdpdGggdG9yY2gubm9fZ3JhZCgpLCB0b3JjaC5hdXRvY2FzdCgiY3VkYSIsIGR0eXBlPXRvcmNoLmZsb2F0MTYpOgogICAgICAgIGZvciBrIGluIHJhbmdlKDAsIGxlbihjb29yZHMpLCA2NCk6CiAgICAgICAgICAgIGJhdGNoID0gY29vcmRzW2s6ayArIDY0XQogICAgICAgICAgICB4ID0gbnAuc3RhY2soW3N2W3owOnowICsgWiwgcjpyICsgUCwgcTpxICsgUF0gZm9yIHIsIHEgaW4gYmF0Y2hdKS5hc3R5cGUobnAuZmxvYXQzMikKICAgICAgICAgICAga2VlcCA9ICh4WzosIFogLy8gMl0gPiAwKS5yZXNoYXBlKGxlbihiYXRjaCksIC0xKS5tZWFuKDEpID4gMC4zCiAgICAgICAgICAgIGlmIG5vdCBrZWVwLmFueSgpOgogICAgICAgICAgICAgICAgY29udGludWUKICAgICAgICAgICAgeCA9IG5wLnN0YWNrKFtub3JtYWxpemVfcm9idXN0KHApIGZvciBwIGluIHhdKS5hc3R5cGUobnAuZmxvYXQzMikKICAgICAgICAgICAgcHIgPSB0b3JjaC5zaWdtb2lkKG5ldCh0b3JjaC5mcm9tX251bXB5KHgpWzosIE5vbmVdLnRvKGRldikpLmZsb2F0KCkpWzosIDBdLmNwdSgpLm51bXB5KCkKICAgICAgICAgICAgZm9yIChyLCBxKSwgcHAsIGtrIGluIHppcChiYXRjaCwgcHIsIGtlZXApOgogICAgICAgICAgICAgICAgaWYga2s6CiAgICAgICAgICAgICAgICAgICAgb3V0W3I6ciArIFAsIHE6cSArIFBdICs9IHBwICogdzI7IHd0W3I6ciArIFAsIHE6cSArIFBdICs9IHcyCiAgICByZXR1cm4gbnAud2hlcmUod3QgPiAwLCBvdXQgLyBucC5tYXhpbXVtKHd0LCAxZS02KSwgMCkKCgpkZWYgYmVzdDRfcDk1KHAsIHYsIHB4X21tPTAuMDA5MzYyKToKICAgIHdpbiA9IGludCgyMCAvIHB4X21tKTsgYmVzdCA9IDAuMAogICAgZm9yIHIwIGluIHJhbmdlKDAsIG1heChwLnNoYXBlWzBdIC0gd2luLCAwKSArIDEsIHdpbiAvLyA0KToKICAgICAgICBmb3IgYzAgaW4gcmFuZ2UoMCwgbWF4KHAuc2hhcGVbMV0gLSB3aW4sIDApICsgMSwgd2luIC8vIDQpOgogICAgICAgICAgICB3ID0gdltyMDpyMCArIHdpbiwgYzA6YzAgKyB3aW5dCiAgICAgICAgICAgIGlmIHcuc2l6ZSBhbmQgdy5tZWFuKCkgPiAwLjY6CiAgICAgICAgICAgICAgICBiZXN0ID0gbWF4KGJlc3QsIGZsb2F0KG5wLnBlcmNlbnRpbGUocFtyMDpyMCArIHdpbiwgYzA6YzAgKyB3aW5dW3ddLCA5NSkpKQogICAgcmV0dXJuIGJlc3QKCgpTID0gZGF0YS5TQ1JPTExTW1NDXQpWT1ggPSBmbG9hdChTLmdldCgicHhfdW0iLCA5LjM2MikpOyBQWCA9IDkuMzYyIC8gVk9YICAjIHJlbmRlciBhdCB0aGUgZGV0ZWN0b3IgcGl4ZWwgKDkuMzYyIHVtKSB3aGF0ZXZlciB0aGUgc2NhbiB2b3hlbAp2b2wgPSBkYXRhLm9wZW5fYXJyYXkoU1sidm9sIl0gKyAiLzAiKTsgc3VyZiA9IGRhdGEub3Blbl9hcnJheShTWyJzdXJmIl0gKyAiLzAiKQp1bWIgPSAoanNvbi5sb2FkKG9wZW4ob3MuZW52aXJvblsiVU1CX0ZJTEUiXSkpIGlmIG9zLmVudmlyb24uZ2V0KCJVTUJfRklMRSIpIGVsc2UgZGF0YS51bWJpbGljdXNfZm9yKFNDKSlbImNvbnRyb2xfcG9pbnRzIl0KY3ogPSBucC5hcnJheShbcFsieiJdIGZvciBwIGluIHVtYl0pCkcgPSBucC5sb2FkKEdSSURTKQpTVEFSVCA9IG9zLmVudmlyb24uZ2V0KCJTVEFSVCIsICJ3MDY1IiBpZiBTQyA9PSAiUEhlcmMwODI2IiBlbHNlICJ3MDAwIikKbmFtZXMgPSBbbiBmb3IgbiBpbiBzb3J0ZWQoRy5maWxlcylbNTotMjpFVkVSWV0gaWYgbiA+PSBTVEFSVF0KcHJpbnQoU0MsIEZJVCwgIndpbmRpbmdzIiwgbGVuKG5hbWVzKSwgZmx1c2g9VHJ1ZSkKcmVzdWx0cyA9IFtdCmZvciB3IGluIG5hbWVzOgogICAgaWYgdGltZS50aW1lKCkgLSBUMCA+IEJVREdFVDoKICAgICAgICBicmVhawogICAgdCA9IHRpbWUudGltZSgpCiAgICB0cnk6CiAgICAgICAgZyA9IEdbd107IGkgPSBpbnQobnAuYXJnbWluKGFicyhjeiAtIG5wLm1lZGlhbihnWzJdW2dbMl0gPiAwXSkpKSkKICAgICAgICBzdiwgZmcsIHN0ID0gc3VydmV5LndpbmRpbmdfdm9sdW1lKGcsIHZvbCwgc3VyZiwgKHVtYltpXVsieCJdLCB1bWJbaV1bInkiXSksIGxheWVycz0yMSwgcHg9UFgpCiAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICAgICAgcHJpbnQoInJlbmRlciBmYWlsZWQiLCB3LCByZXByKGUpWzozMDBdLCBmbHVzaD1UcnVlKTsgY29udGludWUKICAgIHYgPSBzdltzdi5zaGFwZVswXSAvLyAyXSA+IDAKICAgIHIgPSBkaWN0KHc9dywgc2hhcGU9bGlzdChzdi5zaGFwZSksIGFyZWFfY20yPXJvdW5kKGZsb2F0KHYuc3VtKCkpICogMC4wMDkzNjIgKiogMiAvIDEwMCwgMiksIHJlbmRlcl9zPXJvdW5kKHRpbWUudGltZSgpIC0gdCkpCiAgICBucC5zYXZlel9jb21wcmVzc2VkKGYie1d9L3t3fV9ncmlkLm5weiIsIGdyaWQ9ZmcpCiAgICBpbWFnZWlvLmltd3JpdGUoZiJ7V30ve3d9X21pZC5qcGciLCBzdltzdi5zaGFwZVswXSAvLyAyXVs6OjIsIDo6Ml0pCiAgICBmb3IgaywgbmV0IGluIG5ldHMuaXRlbXMoKToKICAgICAgICBwID0gcHJlZGljdChuZXQsIHN2KQogICAgICAgIGltYWdlaW8uaW13cml0ZShmIntXfS97d31fe2t9LnBuZyIsIChwWzo6MiwgOjoyXSAqIDI1NSkuYXN0eXBlKG5wLnVpbnQ4KSkKICAgICAgICByW2Yie2t9X3A5NSJdID0gcm91bmQoZmxvYXQobnAucGVyY2VudGlsZShwW3ZdLCA5NSkpLCAzKSBpZiB2LmFueSgpIGVsc2UgTm9uZQogICAgICAgIHJbZiJ7a31fYmVzdDRfcDk1Il0gPSByb3VuZChiZXN0NF9wOTUocCwgdiksIDMpCiAgICByWyJzZWNzIl0gPSByb3VuZCh0aW1lLnRpbWUoKSAtIHQpCiAgICBzdWJwcm9jZXNzLnJ1bigicm0gLXJmIC90bXAvY2FjaGUiLCBzaGVsbD1UcnVlKSAgIyB0aGUgQ1QgY2h1bmsgY2FjaGUgZmlsbGVkIHRoZSBkaXNrIChzZXNzaW9uIGtpbGxlZCBhdCB+MjMgd2luZGluZ3MpCiAgICByZXN1bHRzLmFwcGVuZChyKTsgcHJpbnQociwgZmx1c2g9VHJ1ZSkKICAgIGpzb24uZHVtcChyZXN1bHRzLCBvcGVuKGYie1d9L3RyYWNlX3N1cnZleV97U0N9X3tGSVR9Lmpzb24iLCAidyIpLCBpbmRlbnQ9MSkKcHJpbnQoInRvdGFsIiwgdGltZS50aW1lKCkgLSBUMCkK"
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




VOXEL = {"PHerc0800": 8.64, "PHerc1218": 8.64, "PHerc0268": 8.64}


def prepare(sc, ts):
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
    u = json.load(open(glob.glob(f"/kaggle/input/**/umb_{sc}.json", recursive=True)[0]))
    json.dump(u, open(f"{D}/umbilicus.json", "w"))
    empty = {"vc_pointcollections_json_version": "1", "collections": {}}
    for f in ("abs_winding.json", "same_windings.json", "relative_windings.json"):
        json.dump(empty, open(f"{D}/{f}", "w"))
    json.dump({"schema_version": 1, "name": sc, "voxel_size_um": VOXEL.get(sc, 9.362), "spiral_outward_sense": "CW",
               "normal_zarr_group": "2", "lasagna_scale": 4,
               "paths": {"tracks_dbm": f"tracks/{sc}.dbm",
                         "normal_x": f"lasagna_inputs/{sc}_nx.ome.zarr",
                         "normal_y": f"lasagna_inputs/{sc}_ny.ome.zarr",
                         "gradient_magnitude": f"lasagna_inputs/{sc}_grad_mag.ome.zarr"}},
              open(f"{D}/spiral-scroll.json", "w"))
    return D



PILOT = 2500


def launch(name, D, z0, z1, gpu, sense, steps):
    R = f"/tmp/ds_run_{name}"
    os.makedirs(R, exist_ok=True)
    for e in os.listdir(D):
        if e != "spiral-scroll.json" and not os.path.exists(f"{R}/{e}"):
            os.symlink(f"{D}/{e}", f"{R}/{e}")
    spec = json.load(open(f"{D}/spiral-scroll.json")); spec["spiral_outward_sense"] = sense
    json.dump(spec, open(f"{R}/spiral-scroll.json", "w"))
    cfg = {"z_begin": z0, "z_end": z1, "optimizer_num_training_steps": steps,
           "input_use_fibers": False, "input_use_pcl_drawn_control_points": False, "input_use_tracks": True,
           "input_use_fiber_directions": False, "input_use_verified_patches": False, "input_use_winding_inference": False,
           "input_use_pcl_absolute": False, "input_use_pcl_same_winding": False, "input_use_outer_shell": False,
           "input_use_pcl_relative": False}
    env = (f"CUDA_VISIBLE_DEVICES={gpu} FIT_SPIRAL_OUT_DIR=/tmp/out_{name} FIT_SPIRAL_CACHE_DIR=/tmp/cache_{name} "
           f"WANDB_MODE=disabled FIT_SPIRAL_CONFIG_OVERRIDES='{json.dumps(cfg)}'")
    print("launch", name, sense, steps, f"[{time.time()-T0:.0f}s]", flush=True)
    return subprocess.Popen(f"cd /tmp/villa/spiral-fitting && {env} {PY} fit_spiral.py --dataset {R}",
                            shell=True, stdout=open(f"{W}/fit_{name}.log", "w"), stderr=subprocess.STDOUT)


def satisfied(name):
    t = open(f"{W}/fit_{name}.log", errors="replace").read()
    m = re.findall(r"satisfied_track_points = \d+/\d+ \(([\d.]+)%\)", t)
    return float(m[-1]) if m else -1.0


with ThreadPoolExecutor(2) as ex:
    DS = dict(zip([j[1] for j in JOBS], ex.map(lambda j: prepare(j[1], j[2]), JOBS)))
senses = {}
for name, sc, ts, z0, z1 in JOBS:  # spiral outward sense: no automatic method exists (team reads it by hand)
    NG = int(os.environ.get("NGPU", "2"))
    ps = {sen: launch(f"{name}_{sen}", DS[sc], z0, z1, g % NG, sen, PILOT) for g, sen in enumerate(("CW", "ACW"))}
    for sen, p in ps.items():
        p.wait()
    sat = {sen: satisfied(f"{name}_{sen}") for sen in ps}
    senses[name] = max(sat, key=sat.get)
    print(name, "pilot satisfied_track_points %", sat, "->", senses[name], f"[{time.time()-T0:.0f}s]", flush=True)
    sh(f"rm -rf /tmp/out_{name}_CW /tmp/out_{name}_ACW /tmp/cache_{name}_CW /tmp/cache_{name}_ACW")
json.dump(senses, open(f"{W}/senses.json", "w"))
procs = {name: launch(name, DS[sc], z0, z1, gi % int(os.environ.get("NGPU", "2")), senses[name], STEPS) for gi, (name, sc, ts, z0, z1) in enumerate(JOBS)}
for name, p in procs.items():
    print(f"fit {name} rc={p.wait()} satisfied {satisfied(name)} [{time.time()-T0:.0f}s]", flush=True)
    sh(f"grep -v Warning {W}/fit_{name}.log | grep -i 'Error\\|Traceback' | tail -8")
import numpy as np, tifffile
for name, *_ in JOBS:
    allm = [m for m in glob.glob(f"/tmp/out_{name}/**/w*", recursive=True) if os.path.exists(f"{m}/x.tif")]
    ms = sorted(m for m in allm if re.fullmatch(r"w\d{3}", os.path.basename(m)))
    grids = {os.path.basename(m): np.stack([tifffile.imread(f"{m}/{a}.tif") for a in "xyz"]).astype(np.float32) for m in ms}
    np.savez_compressed(f"{W}/grids_{name}.npz", **grids)
    print(name, "windings exported", len(grids), flush=True)
sh("rm -rf /tmp/ds_* /tmp/cache_*")
import base64
open("/tmp/survey.py", "w").write(base64.b64decode(SURVEY_B64).decode())
for k, (name, sc, ts, z0, z1) in enumerate(JOBS):
    left = (11.6 * 3600 - (time.time() - T0)) / 3600 / (len(JOBS) - k)
    if left < 0.3 or not os.path.exists(f"{W}/grids_{name}.npz"):
        continue
    os.makedirs(f"{W}/survey_{name}", exist_ok=True)
    env = dict(os.environ, SC=sc, FIT=name, GRIDS=f"{W}/grids_{name}.npz", SURVEY_W=f"{W}/survey_{name}", START="w000",
               EVERY="3", BUDGET_H=f"{left:.2f}", UMB_FILE=glob.glob(f"/kaggle/input/**/umb_{sc}.json", recursive=True)[0])
    r = subprocess.run("python /tmp/survey.py", shell=True, env=env, capture_output=True, text=True, errors="replace")
    open(f"{W}/survey_{name}.log", "w").write(r.stdout + r.stderr)
    print(name, "survey rc", r.returncode, r.stdout[-1500:], r.stderr[-800:], flush=True)
print("total", time.time() - T0)
