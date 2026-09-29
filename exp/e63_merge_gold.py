"""Merge per-slice gold files into one VC3D point-collection file with INTEGER keys (fit_spiral requirement).
Usage: e63_merge_gold.py OUT.json GLOB"""
import sys, json, glob
cols = {}
for f in sorted(glob.glob(sys.argv[2])):
    for c in json.load(open(f))["collections"].values():
        cols[str(len(cols) + 1)] = c
json.dump({"vc_pointcollections_json_version": "1", "collections": cols}, open(sys.argv[1], "w"))
print(len(cols), "ladders ->", sys.argv[1])
