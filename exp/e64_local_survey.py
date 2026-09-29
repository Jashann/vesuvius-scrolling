"""E64: local TRACE survey of fitted windings (render native 9.362 um -> base/sA/sB scores), for when the Kaggle
GPU quota is exhausted. Usage: e64_local_survey.py SCROLL grids.npz w056,w058,..."""
import sys, os, json, time
sys.argv, args = [sys.argv[0], "none"], sys.argv[1:]
__file__ = os.path.abspath(os.path.join(os.path.dirname(__file__), "e58_trace_survey.py"))
exec(open(__file__).read().split("done = set()")[0])  # models, predict, best4, run() from E58
from pcu import data, survey
sc, npz, names = args[0], args[1], args[2].split(",")
S = data.SCROLLS[sc]
vol = data.open_array(S["vol"] + "/0"); surf = data.open_array(S["surf"] + "/0")
u = data.umbilicus_for(sc)["control_points"]; cz = np.array([p["z"] for p in u])
G = np.load(npz)
done = {json.loads(l)["tag"] for l in open(f"{OUT}/results.jsonl")} if os.path.exists(f"{OUT}/results.jsonl") else set()
for w in names:
    tag = f"{sc}_{os.path.basename(npz)[:-4]}_{w}"
    if tag in done or w not in G.files:
        continue
    t = time.time()
    g = G[w]; i = int(np.argmin(abs(cz - np.median(g[2][g[2] > 0]))))
    sv, fg, st = survey.winding_volume(g, vol, surf, (u[i]["x"], u[i]["y"]), layers=21, px=1.0)
    run(tag, sv)
    print(tag, f"{time.time()-t:.0f}s", flush=True)
