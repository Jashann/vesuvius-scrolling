"""E93: TRACE-v3 manifest. Adds every public segment that has a coarse-protocol surface volume (8.64 or 9.362 um)
and a fine-scan ink prediction on the same mesh: PHerc0009B (8.64), PHerc0343P (8.64), PHerc0500P2 (9.362),
on top of the v1/v2 training set (PHerc0139/0814). Held out unchanged: PHerc0841 x3 and PHerc0139 w043.
Writes runs/trace_manifest_v3.json. Usage: e93_manifest_v3.py"""
import re, json, requests

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
sess = requests.Session()


def ls(prefix):
    keys, pre, tok = [], [], None
    while True:
        u = f"{S3}/?list-type=2&delimiter=/&prefix={prefix}&max-keys=1000" + (f"&continuation-token={requests.utils.quote(tok)}" if tok else "")
        t = sess.get(u, timeout=120).text
        keys += re.findall(r"<Key>([^<]*)</Key>", t); pre += re.findall(r"<Prefix>([^<]*)</Prefix>", t)[1:]
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", t)
        if not m:
            return keys, pre
        tok = m.group(1)


def coarse_sv(seg, px):
    _, pre = ls(f"{seg}surface-volumes/")
    return [p.rstrip("/") for p in pre if px in p and p.rstrip("/").endswith(".zarr")]


def fine_ink(seg):
    keys, _ = ls(f"{seg}ink-detection/")
    tifs = [k for k in keys if k.endswith(".tif") and "downsampled" not in k]
    pref = [k for k in tifs if "new_canon" in k] or tifs
    return pref


old = json.load(open("runs/trace_manifest.json"))
train = list(old["train"]); added = []
for sc, px in (("PHerc0009B", "8.64um"), ("PHerc0343P", "8.64um"), ("PHerc0500P2", "9.362um")):
    _, segs = ls(f"{sc}/segments/")
    for seg in segs:
        if seg.endswith("raw/"):
            continue
        svs = coarse_sv(seg, px); inks = fine_ink(seg)
        if not svs or not inks:
            continue
        meta = sess.get(f"{S3}/{svs[0]}/0/.zarray", timeout=60)
        shape = meta.json()["shape"] if meta.ok else None
        e = dict(scroll=sc, seg=seg.split("/")[2], sv=svs[0], ink11=None, ink24=inks[0], labels=[], px=px, shape=shape)
        train.append(e); added.append(e)
        print(sc, e["seg"], shape, inks[0].split("/")[-1][:60], flush=True)
print("added", len(added), "train total", len(train))
json.dump({"train": train, "test": old["test"]}, open("runs/trace_manifest_v3.json", "w"), indent=1)
