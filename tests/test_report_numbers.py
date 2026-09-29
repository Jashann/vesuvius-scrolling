"""The numbers quoted in the docs must follow from the committed result files. Run: python -m pytest tests"""
import json, math, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_certificate_numbers():
    rows = json.load(open(f"{ROOT}/results/e9_rows.json"))
    assert len(rows) == 1502
    loc = np.array([r["loc"] for r in rows]); den = np.array([r["den"] for r in rows]); glob = np.array([r["glob"] for r in rows]); dw = np.array([r["dw"] for r in rows])
    rl, rd = np.round(loc), np.round(den)
    c = (rl == rd) & (rl == glob) & (np.abs(loc - rl) < 0.25) & (np.abs(den - rd) < 0.3)
    assert int(c.sum()) == 1052 and int(np.sum(rl[c] != dw[c])) == 4
    assert abs(c.mean() - 0.700) < 0.001 and abs(np.mean(rl[c] == dw[c]) - 0.9962) < 0.0001
    assert abs(np.mean(rl == dw) - 0.9561) < 0.0001


def test_negative_set_numbers():
    rows = json.load(open(f"{ROOT}/results/e9b_rows.json"))
    loc = np.array([r["loc"] for r in rows]); den = np.array([r["den"] for r in rows]); glob = np.array([r["glob"] for r in rows]); dw = np.array([r["dw"] for r in rows])
    rl, rd = np.round(loc), np.round(den)
    c = (rl == rd) & (rl == glob) & (np.abs(loc - rl) < 0.25) & (np.abs(den - rd) < 0.3)
    two = dw == 2
    assert int((c & two).sum()) == 636 and int(np.sum((rl == 1) & c & two)) == 1
    assert int((c & (dw == 3)).sum()) == 411 and int(np.sum((rl == 1) & c & (dw == 3))) == 0


def test_fit_numbers():
    p4 = json.load(open(f"{ROOT}/results/fit/e31b_paris4_bootstrap.json"))
    assert p4["A2"]["slips"] == 59 and p4["A2"]["pairs"] == 291
    assert p4["B3w10"]["slips"] == 30 and p4["B3w10"]["pairs"] == 324
    assert p4["B3w10"]["ci95"][1] < p4["A2"]["ci95"][0], "B3w10 and A2 intervals must not overlap"
    assert p4["selection_free"]["held_out_rate_mean"] < 0.12
    s2 = json.load(open(f"{ROOT}/results/fit/e31b_paris4_seed2_bootstrap.json"))
    assert s2["A2_s2"]["slips"] == 66 and s2["B3w10_s2"]["slips"] == 30 and s2["B3w10_s2"]["pairs"] == 339
    assert s2["B3w10_s2"]["ci95"][1] < s2["A2_s2"]["ci95"][0]
    b2 = json.load(open(f"{ROOT}/results/fit/e31b_paris4_z11000_bootstrap.json"))
    assert b2["A2"]["slips"] == 48 and b2["B3w10"]["slips"] == 46 and b2["B3w10"]["diff_vs_ref"]["ci95"][1] > 0, "band 2 shows no gain; docs must say so"
    s = json.load(open(f"{ROOT}/results/fit/e31b_pherc0826_bootstrap.json"))
    assert abs(s["A0"]["rate"] - 0.501) < 0.002 and abs(s["B1w20"]["rate"] - 0.363) < 0.002


def test_shipped_constraint_files():
    for name, n in (("pcu_gold_0826_fit_v2.json", 25771), ("gold0826_heldout8.json", 3426), ("pcu_gold_0211.json", 30761)):
        d = json.load(open(f"{ROOT}/data/gold/{name}"))
        assert d["vc_pointcollections_json_version"] == "1" and len(d["collections"]) == n
    fit = json.load(open(f"{ROOT}/data/gold/pcu_gold_0826_fit_v2.json")); held = json.load(open(f"{ROOT}/data/gold/gold0826_heldout8.json"))
    zf = {p["p"][2] for c in fit["collections"].values() for p in c["points"].values()}
    zh = {p["p"][2] for c in held["collections"].values() for p in c["points"].values()}
    assert not (zf & zh), "held-out slices must not be constraint slices"


def test_report_script_runs():
    out = subprocess.run([sys.executable, f"{ROOT}/exp/e9_report.py", f"{ROOT}/results/e9_rows.json"], capture_output=True, text=True, check=True).stdout
    assert "0.9962" in out and "1052" in out
