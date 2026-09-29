"""Minimal reader for 2D uint8 zarr v3 arrays with sharding_indexed + blosc inner codec (the format of the
team's published ink labels), without needing zarr 3."""
import json
import numpy as np
import numcodecs
import requests


def read_2d(url):
    meta = requests.get(f"{url}/zarr.json", timeout=120).json()
    H, W = meta["shape"]
    shard = meta["chunk_grid"]["configuration"]["chunk_shape"]
    cfg = meta["codecs"][0]["configuration"]
    ih, iw = cfg["chunk_shape"]
    blosc = numcodecs.Blosc()
    out = np.zeros((H, W), np.uint8)
    for sr in range(0, H, shard[0]):
        for sc in range(0, W, shard[1]):
            r = requests.get(f"{url}/c/{sr // shard[0]}/{sc // shard[1]}", timeout=600)
            if r.status_code != 200:
                continue
            b = r.content
            n = (shard[0] // ih) * (shard[1] // iw)
            idx = np.frombuffer(b[-(n * 16 + 4):-4], dtype="<u8").reshape(n, 2)
            for k, (off, nb) in enumerate(idx):
                if off == 2 ** 64 - 1:
                    continue
                cr, cc = divmod(k, shard[1] // iw)
                chunk = np.frombuffer(blosc.decode(b[off:off + nb]), np.uint8).reshape(ih, iw)
                r0, c0 = sr + cr * ih, sc + cc * iw
                if r0 >= H or c0 >= W:
                    continue
                out[r0:min(r0 + ih, H), c0:min(c0 + iw, W)] = chunk[:min(ih, H - r0), :min(iw, W - c0)]
    return out
