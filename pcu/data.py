"""Remote access to the Vesuvius open-data bucket with a local chunk cache."""
import json
import os
from functools import lru_cache

import numpy as np
import requests
import zarr
from zarr.storage import FSStore, LRUStoreCache

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
DL = "https://dl.ash2txt.org"
ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.environ.get("PCU_CACHE", os.path.join(ROOT, "cache"))

P4 = "PHercParis4"
P4_VOL = f"{P4}/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
P4_LAS = f"{P4}/representations/predictions/lasagna/20260411134726-lasagna-20260419180421-L2/PHercParis4-20260411134726-las-sd2-5b17ff6c"
P4_SURF = f"{P4}/representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr"
P4_FIB = f"{P4}/representations/predictions/fibers/20260411134726-fibers-20260915212757-L1"
P4_UMB = f"{P4}/representations/umbilicus/20260411134726-umbilicus-20260524235033.json"
P4_SPIRAL = f"{DL}/datasets/spiral_datasets/PHercParis4"


class DiskCacheStore(FSStore):
    """FSStore that mirrors fetched chunks to local disk."""

    def __init__(self, url, local_dir, **kw):
        super().__init__(url, mode="r", **kw)
        self.local_dir = local_dir

    def __getitem__(self, key):
        key = self._normalize_key(key)
        path = os.path.join(self.local_dir, key)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        import time
        for attempt in range(6):  # transient S3/aiohttp drops must not kill a multi-hour run
            try:
                data = super().__getitem__(key)
                break
            except KeyError:
                raise
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 * (attempt + 1))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import threading
        tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"  # unique: parallel readers race otherwise
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        return data

    def getitems(self, keys, **kwargs):
        # zarr batches reads through getitems; route them through the disk cache in parallel
        from concurrent.futures import ThreadPoolExecutor

        def one(k):
            try:
                return k, self[k]
            except KeyError:
                return k, None

        with ThreadPoolExecutor(32) as ex:
            return {k: v for k, v in ex.map(one, keys) if v is not None}


@lru_cache(maxsize=None)
def open_array(rel_path):
    """Open one zarr array (a pyramid level path) from the S3 bucket."""
    store = DiskCacheStore(f"{S3}/{rel_path}", os.path.join(CACHE, rel_path))
    return zarr.open(LRUStoreCache(store, max_size=2**30), mode="r")


def prefetch(arr, region, workers=48):
    """Download all chunks overlapping region (tuple of slices) in parallel into the disk cache."""
    from concurrent.futures import ThreadPoolExecutor
    import itertools

    store = getattr(arr.store, "_store", arr.store)
    ranges = []
    for sl, c, n in zip(region, arr.chunks, arr.shape):
        start = 0 if sl.start is None else sl.start
        stop = n if sl.stop is None else min(sl.stop, n)
        ranges.append(range(start // c, (stop - 1) // c + 1))
    keys = [arr._chunk_key(idx) for idx in itertools.product(*ranges)]

    def get(k):
        try:
            store[k]
        except KeyError:
            pass

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, keys))
    return len(keys)


def read(arr, region, workers=48):
    """Fast region read: fetch chunks in parallel via the disk cache, decode ourselves."""
    from concurrent.futures import ThreadPoolExecutor
    import itertools
    import numcodecs

    store = getattr(arr.store, "_store", arr.store)
    comp = numcodecs.get_codec(arr.compressor.get_config()) if arr.compressor else None
    lo = [0 if sl.start is None else sl.start for sl in region]
    hi = [n if sl.stop is None else min(sl.stop, n) for sl, n in zip(region, arr.shape)]
    out = np.full([h - l for l, h in zip(lo, hi)], arr.fill_value or 0, dtype=arr.dtype)
    idxs = list(itertools.product(*[range(l // c, (h - 1) // c + 1) for l, h, c in zip(lo, hi, arr.chunks)]))

    def one(idx):
        try:
            raw = store[arr._chunk_key(idx)]
        except KeyError:
            return
        buf = comp.decode(raw) if comp else raw
        ch = np.frombuffer(buf, dtype=arr.dtype).reshape(arr.chunks)
        src, dst = [], []
        for i, l, h, c in zip(idx, lo, hi, arr.chunks):
            a, b = max(i * c, l), min((i + 1) * c, h)
            src.append(slice(a - i * c, b - i * c))
            dst.append(slice(a - l, b - l))
        out[tuple(dst)] = ch[tuple(src)]

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, idxs))
    return out


def lasagna(channel, level):
    return open_array(f"{P4_LAS}_{channel}.ome.zarr/{level}")


def fetch_json(url):
    local = os.path.join(CACHE, url.split("://", 1)[1])
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        tmp = f"{local}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            f.write(r.content)
        os.replace(tmp, local)
    with open(local) as f:
        return json.load(f)


def fetch_file(url):
    local = os.path.join(CACHE, url.split("://", 1)[1])
    if not os.path.exists(local):
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            tmp = f"{local}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                for b in r.iter_content(1 << 20):
                    f.write(b)
            os.replace(tmp, local)
    return local


def umbilicus_p4():
    return fetch_json(f"{S3}/{P4_UMB}")


SCROLLS = {
    "PHercParis4": dict(
        las=P4_LAS, vol=P4_VOL, umb=P4_UMB),
    "PHerc0826": dict(
        las="PHerc0826/representations/predictions/lasagna/20250821151701-lasagna-20260419180421/PHerc0826",
        vol="PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr",
        umb="PHerc0826/representations/umbilicus/20250821151701-umbilicus-20260808113303.json",
        surf="PHerc0826/representations/predictions/surfaces/20250821151701-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    "PHerc0125": dict(
        las="PHerc0125/representations/predictions/lasagna/20250821151825-lasagna-20260419180421/PHerc0125",
        vol="PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr",
        umb="PHerc0125/representations/umbilicus/20250821151825-umbilicus-20260808111524.json",
        surf="PHerc0125/representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    "PHerc0211": dict(
        las="PHerc0211/representations/predictions/lasagna/20250821151803-lasagna-20260419180421/PHerc0211",
        vol="PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr",
        umb="PHerc0211/representations/umbilicus/20250821151803-umbilicus-20260808112626.json",
        surf="PHerc0211/representations/predictions/surfaces/20250821151803-surface-20260413222639-surface-m7-L0-th0.2.zarr"),
    **{sc: dict(  # eligible 9.362 um scrolls without a published umbilicus (ours: runs/umb_{sc}.json, E67)
        las=f"{sc}/representations/predictions/lasagna/{ts}-lasagna-20260419180421/{sc}",
        vol=f"{sc}/volumes/{ts}-9.362um-1.2m-113keV-masked.zarr",
        surf=f"{sc}/representations/predictions/surfaces/{ts}-surface-20260413222639-surface-m7-L0-th0.2.zarr")
       for sc, ts in (("PHerc0191", "20250821151635"), ("PHerc0257", "20250821151750"),
                      ("PHerc0358", "20250821151737"), ("PHerc0813", "20250821151723"))},
    **{sc: dict(  # 8.64 um / 116 keV eligible scrolls (same protocol as PHerc1447, where letters were found)
        las=f"{sc}/representations/predictions/lasagna/{ts}-lasagna-20260419180421/{sc}",
        vol=f"{sc}/volumes/{ts}-8.640um-1.2m-116keV-masked.zarr",
        surf=f"{sc}/representations/predictions/surfaces/{ts}-surface-20260413222639-surface-m7-L0-th0.2.zarr", px_um=8.64)
       for sc, ts in (("PHerc0800", "20250521135224"), ("PHerc1218", "20250521120456"))},
    "PHerc1203": dict(
        las="PHerc1203/representations/predictions/lasagna/20250820131727-lasagna-20260419180421/PHerc1203",
        vol="PHerc1203/volumes/20250820131727-9.362um-1.2m-113keV-masked.zarr",
        vol24="PHerc1203/volumes/20260319130212-2.403um-0.2m-77keV-masked.zarr",
        surf="PHerc1203/representations/predictions/surfaces/20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr",
        surf24="PHerc1203/representations/predictions/surfaces/20260319130212-surface-20260413222639-surface-m7-L2-th0.2.zarr"),
    "PHerc0139": dict(
        las="PHerc0139/representations/predictions/lasagna/20260102150214-lasagna-20260419180421-L2/PHerc0139-20260102150214-lasagna-20260724",
        vol="PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr",
        umb="PHerc0139/representations/umbilicus/20260102150214-umbilicus-20260826112529.json"),
}


def lasagna_for(scroll, channel, level):
    return open_array(f"{SCROLLS[scroll]['las']}_{channel}.ome.zarr/{level}")


def umbilicus_for(scroll):
    return fetch_json(f"{S3}/{SCROLLS[scroll]['umb']}")
