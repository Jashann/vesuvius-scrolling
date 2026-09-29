"""TRACE [D]: label-free text screening by row stacking.

Text lines run along constant z at a regular pitch (~3-8 mm) across a whole column. Averaging an ink score
along rows inside column-wide blocks multiplies SNR by ~sqrt(block width / stroke width); a periodicity test
on the row profile then detects the presence of text below single-letter legibility. Null: the same
statistic after randomly permuting the rows of the block (destroys periodicity, keeps the value
distribution), plus air/deep-layer controls chosen by the caller."""
import numpy as np


def row_power(profile, px_mm, pmin=3.0, pmax=8.0):
    """Fraction of (detrended) spectral power in the text-line period band."""
    x = profile - np.convolve(profile, np.ones(31) / 31, mode="same")
    x = x - x.mean()
    F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), d=px_mm)
    band = (f > 1 / pmax) & (f < 1 / pmin)
    return float(F[band].sum() / (F[1:].sum() + 1e-12))


def red_noise_peak(profile, px_mm, pmin=3.0, pmax=8.0, bin_px=4):
    """Max over the text-pitch band of periodogram / fitted AR(1) red-noise spectrum (climate-style test).
    A narrow line-pitch peak scores high; smooth or random structure (red/white spectra) scores ~1-4."""
    x = profile.reshape(-1)[: len(profile) // bin_px * bin_px].reshape(-1, bin_px).mean(1)
    x = x - x.mean()
    rho = float(np.clip(np.corrcoef(x[:-1], x[1:])[0, 1], -0.95, 0.99))
    F = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), d=1.0)
    red = (1 - rho ** 2) / (1 - 2 * rho * np.cos(2 * np.pi * f) + rho ** 2)
    red *= F[1:].mean() / red[1:].mean()
    fm = f / (px_mm * bin_px)
    band = (fm > 1 / pmax) & (fm < 1 / pmin)
    return float((F[band] / red[band]).max()) if band.any() else float("nan")


def screen(prob, valid, px_mm=0.0096, block_mm=20.0, n_null=50, seed=0, min_h_mm=15.0):
    """prob, valid: (H, W) map with rows along constant z. Returns per-block (col0, red-noise peak ratio).
    Under the red-noise null the ratio is ~chi2(2)/2 per frequency; the max over ~10 band bins exceeds ~7
    with ~5% probability. Compare against control maps (deep layers, verso) from the same render."""
    rng = np.random.default_rng(seed)
    bw = int(block_mm / px_mm)
    out = []
    for c0 in range(0, prob.shape[1] - bw + 1, bw // 2):
        v = valid[:, c0:c0 + bw]
        rows = v.mean(1) > 0.6
        if rows.sum() < int(min_h_mm / px_mm):  # need enough height for several text lines
            continue
        prof = np.where(v, prob[:, c0:c0 + bw], 0).sum(1) / np.maximum(v.sum(1), 1)
        prof = prof[rows]
        out.append((c0, red_noise_peak(prof, px_mm)))
    return out
