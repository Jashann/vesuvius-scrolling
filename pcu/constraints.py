"""Certified automatic relative-winding ladders (VC3D point-collection JSON, level-2 coordinates).

A ladder starts on a sheet crest and steps outward one wrap at a time along the oriented normal.
A step a->b is kept only if three independent measurements agree that exactly one wrap separates
the points: local phase count along the segment, Lasagna wrap-density integral, and the whole-slice
MCF winding field; the first two must also be within tolerance of an integer. On human ladders
this certificate had 99.6% precision at 70% coverage (E9).
"""
import numpy as np
from scipy import ndimage as ndi

from . import phase as ph

K_DENSITY = 0.808  # grad_mag integral calibration (E0c)
PX = 8.0  # base-resolution voxels per cos pixel (Paris 4: cos at pyramid level 3)


def _next_crest(phi, mx, my, p, direction, max_len=40.0, step=0.25):
    """March from p along +-normal until the (orientation-propagated) phase advances by 2pi."""
    y, x = p
    H, W = phi.shape
    cur_m = np.array([ndi.map_coordinates(mx, [[y], [x]], order=0)[0], ndi.map_coordinates(my, [[y], [x]], order=0)[0]]) * direction
    cur_p = ndi.map_coordinates(phi, [[y], [x]], order=0)[0] * direction
    acc = 0.0
    tr = [(y, x)]
    for _ in range(int(max_len / step)):
        y2, x2 = y + step * cur_m[1], x + step * cur_m[0]
        if not (0 <= y2 < H - 1 and 0 <= x2 < W - 1):
            return None
        m2 = np.array([ndi.map_coordinates(mx, [[y2], [x2]], order=0)[0], ndi.map_coordinates(my, [[y2], [x2]], order=0)[0]])
        s = 1.0 if m2 @ cur_m >= 0 else -1.0
        p2 = ndi.map_coordinates(phi, [[y2], [x2]], order=0)[0] * s
        acc += ph.wrap(p2 - cur_p)
        cur_p, cur_m = p2, m2 * s
        y, x = y2, x2
        tr.append((y, x))
        if abs(acc) >= 2 * np.pi:
            return np.array([y, x])
    return None


def certify_k(a, b, k, phi, g3, Wf, tol_loc=0.3, tol_den=0.35):
    """Certify that exactly k wraps separate a and b (same three-way agreement, looser tolerance)."""
    L = np.linalg.norm(b - a)
    n = max(int(L * 4), 8)
    t = np.linspace(0, 1, n)
    ys, xs = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
    loc = abs(np.sum(ph.wrap(np.diff(ndi.map_coordinates(phi, [ys, xs], order=0))))) / (2 * np.pi)
    den = float(np.sum(ndi.map_coordinates(g3, [ys, xs], order=1)) * L / n * PX * K_DENSITY)
    wa, wb = ndi.map_coordinates(Wf, [[a[0], b[0]], [a[1], b[1]]], order=0)
    glob = abs(np.round(wb) - np.round(wa))
    return (np.round(loc) == k and np.round(den) == k and glob == k and abs(loc - k) < tol_loc and abs(den - k) < tol_den)


def certify(a, b, phi, g3, Wf, tol_loc=0.25, tol_den=0.3):
    L = np.linalg.norm(b - a)
    n = max(int(L * 4), 8)
    t = np.linspace(0, 1, n)
    ys, xs = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
    loc = abs(np.sum(ph.wrap(np.diff(ndi.map_coordinates(phi, [ys, xs], order=0))))) / (2 * np.pi)
    den = float(np.sum(ndi.map_coordinates(g3, [ys, xs], order=1)) * L / n * PX * K_DENSITY)
    wa, wb = ndi.map_coordinates(Wf, [[a[0], b[0]], [a[1], b[1]]], order=0)
    glob = abs(np.round(wb) - np.round(wa))
    ok = (np.round(loc) == 1) and (np.round(den) == 1) and (glob == 1) and abs(loc - 1) < tol_loc and abs(den - 1) < tol_den
    return ok, dict(loc=float(loc), den=den, glob=float(glob))


def recto_count(rmap, a, b, rf, shift=1.5, thr=0.35):
    """Recto bands crossed between crests a and b (L3 coords): both ends shifted `shift` L3 px along
    the step (the recto surface sits 0..+1 L3 px outward of the Lasagna crest), counted on a recto
    map at rf x the L3 resolution, in the same crop frame."""
    d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
    a2, b2 = (a + shift * d) * rf, (b + shift * d) * rf
    n = max(int(np.linalg.norm(b2 - a2) * 2), 8); t = np.linspace(0, 1, n)
    prof = ndi.map_coordinates(rmap, [a2[0] + t * (b2[0] - a2[0]), a2[1] + t * (b2[1] - a2[1])], order=1)
    above = prof > thr
    return int(np.sum(np.diff(above.astype(int)) == 1)) + int(above[0])


def generate(phi, mx, my, R, g3, Wf, mask, amp, seed_step=48, max_wraps=12, rng=None, rmap=None, rf=2):
    """Return list of ladders, each a list of (y, x) points at consecutive wraps (outward).
    With a recto map, each ladder also carries per-step gold flags (fourth, independent vote)."""
    gy, gx = np.gradient(ndi.gaussian_filter(R, 3))
    H, W = phi.shape
    crest = (np.cos(phi) > 0.95) & mask & (amp > np.percentile(amp[mask], 40))
    ladders = []
    for y0 in range(seed_step // 2, H, seed_step):
        for x0 in range(seed_step // 2, W, seed_step):
            win = crest[max(y0 - 6, 0): y0 + 7, max(x0 - 6, 0): x0 + 7]
            if not win.any():
                continue
            yy, xx = np.nonzero(win)
            p = np.array([yy[0] + max(y0 - 6, 0), xx[0] + max(x0 - 6, 0)], float)
            # outward direction sign relative to the local line field
            mloc = np.array([mx[int(p[0]), int(p[1])], my[int(p[0]), int(p[1])]])
            direction = 1.0 if mloc @ np.array([gx[int(p[0]), int(p[1])], gy[int(p[0]), int(p[1])]]) >= 0 else -1.0
            pts = [p]
            gold = []
            for _ in range(max_wraps):
                q = _next_crest(phi, mx * direction, my * direction, pts[-1], 1.0)
                if q is None or not mask[int(q[0]), int(q[1])]:
                    break
                ok, _info = certify(pts[-1], q, phi, g3, Wf)
                if not ok:
                    break
                gold.append(rmap is not None and recto_count(rmap, pts[-1], q, rf) == 1)
                pts.append(q)
            if len(pts) >= 2:
                ladders.append((np.array(pts), np.array(gold)) if rmap is not None else np.array(pts))
    return ladders


def gold_runs(ladders):
    """Split (points, gold_flags) ladders into maximal runs of consecutive gold steps."""
    out = []
    for pts, g in ladders:
        start = 0
        for i in range(len(g) + 1):
            if i == len(g) or not g[i]:
                if i - start >= 1:
                    out.append(pts[start:i + 1])
                start = i + 1
    return out


def to_vc3d_json(ladders, z2, X0, Y0, name_prefix="pcu"):
    """Level-3 (y, x) crop coordinates -> level-2 (x, y, z) VC3D point collections."""
    cols = {}
    pid = 1
    for i, L in enumerate(ladders):
        pts = {}
        for k, (y, x) in enumerate(L):
            pts[str(pid)] = {"p": [float((x + X0) * 2), float((y + Y0) * 2), float(z2)], "wind_a": float(k + 1)}
            pid += 1
        cols[str(i + 1)] = {"name": f"{name_prefix}_{z2}_{i+1}", "metadata": {"winding_is_absolute": False,
                            "generator": "pcu certified ladder (local phase = density = global field)"}, "points": pts}
    return {"vc_pointcollections_json_version": "1", "collections": cols}
