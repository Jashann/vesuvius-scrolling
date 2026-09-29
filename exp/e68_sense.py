"""E68: automatic spiral outward sense (CW/ACW), which the team reads off by hand. Polar-resample the m7
surface prediction around the umbilicus; the radial sheet profile at angle theta+d is the profile at theta
shifted by pitch*d/2pi, outward in the winding direction. Sign of the mean sub-pixel shift over many angles,
radii bands and slices gives the sense; calibrated on scrolls with a known sense, checked by mirroring.
Usage: e68_sense.py SCROLL [UMB.json] [LEVEL]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from scipy import ndimage as ndi
from pcu import data

sc = sys.argv[1]
umb = json.load(open(sys.argv[2]))["control_points"] if len(sys.argv) > 2 and sys.argv[2] != "-" else data.umbilicus_for(sc)["control_points"]
lev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
f = 2 ** lev
a = data.open_array(f"{data.SCROLLS[sc]['surf']}/{lev}")
cz = np.array([p["z"] for p in umb])
NT = 1440  # theta bins (0.5 deg), compare theta vs theta + 10 deg


def shift(p, q, maxs=4):
    """sub-pixel lag s maximising corr(p(r), q(r + s)), both zero-mean"""
    s = np.arange(-maxs, maxs + 1)
    c = np.array([np.sum(p[maxs:-maxs] * q[maxs + k:len(q) - maxs + k]) for k in s])
    i = int(np.argmax(c))
    if 0 < i < len(c) - 1:
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        return s[i] + (0.5 * (c[i - 1] - c[i + 1]) / d if d != 0 else 0)
    return float(s[i])


res = {"orig": [], "mirror": []}
for zf in np.linspace(0.25, 0.75, 3):
    z = int(zf * a.shape[0] * f)
    i = int(np.argmin(abs(cz - z))); ux, uy = umb[i]["x"] / f, umb[i]["y"] / f
    sl = np.asarray(a[z // f]).astype(np.float32)
    for mode in res:
        img = sl[:, ::-1] if mode == "mirror" else sl
        cx = img.shape[1] - 1 - ux if mode == "mirror" else ux
        R = np.arange(20, min(img.shape) // 2, 1.0)
        th = np.linspace(0, 2 * np.pi, NT, endpoint=False)
        X = cx + R[None] * np.cos(th)[:, None]; Y = uy + R[None] * np.sin(th)[:, None]
        P = ndi.map_coordinates(img, [Y, X], order=1, cval=0)
        P = ndi.gaussian_filter1d(P, 1.0, axis=1)
        # follow single sheets once around: net radial change over 2 pi = +-pitch (distortions integrate to 0)
        Pk = (P > np.roll(P, 1, 1)) & (P >= np.roll(P, -1, 1)) & (P > 96)
        sh = []
        for t0 in range(0, NT, NT // 4):
            for r0 in np.flatnonzero(Pk[t0])[::4]:
                r = float(r0); miss = 0
                for k in range(1, NT + 1):
                    row = Pk[(t0 + k) % NT]; lo, hi = int(r) - 2, int(r) + 3
                    c = np.flatnonzero(row[max(lo, 0):hi]) + max(lo, 0)
                    if not len(c):
                        miss += 1
                        if miss > NT // 5:
                            break
                        continue
                    r = float(c[np.argmin(abs(c - r))])
                if miss <= NT // 5:
                    sh.append(r - r0)
        sh = np.array(sh)
        res[mode].append(float(np.median(sh)) if len(sh) else np.nan)
        print(sc, z, mode, "n", len(sh), "median dr", res[mode][-1], "frac>0", round(float((sh > 0).mean()), 3) if len(sh) else None,
              "frac<0", round(float((sh < 0).mean()), 3) if len(sh) else None, flush=True)
for mode, v in res.items():
    v = np.array(v); print(sc, mode, "slices>0", int((v > 0).sum()), "/", len(v), "median", round(float(np.nanmedian(v)), 4))
