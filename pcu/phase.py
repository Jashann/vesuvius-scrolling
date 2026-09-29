"""Wrapped winding phase from a fringe signal, orientation, Fried split and residues."""
import numpy as np
from scipy import ndimage as ndi


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def riesz2d(sig, band=None):
    """Monogenic decomposition of a 2D fringe signal. Returns even part, Riesz (rx, ry)."""
    F = np.fft.fft2(sig)
    h, w = sig.shape
    u = np.fft.fftfreq(w)[None, :]
    v = np.fft.fftfreq(h)[:, None]
    rho = np.hypot(u, v)
    rho[0, 0] = 1
    H = np.ones_like(rho)
    if band is not None:  # log-Gabor centred on 1/period
        f0, sig_on_f = band
        H = np.exp(-(np.log(rho / f0)) ** 2 / (2 * np.log(sig_on_f) ** 2))
    H[0, 0] = 0
    e = np.real(np.fft.ifft2(F * H))
    rx = np.real(np.fft.ifft2(F * H * (-1j * u / rho)))
    ry = np.real(np.fft.ifft2(F * H * (-1j * v / rho)))
    return e, rx, ry


def structure_orientation(rx, ry, sigma):
    """Smoothed doubled-angle orientation of the local wave vector (a line field)."""
    j_xx = ndi.gaussian_filter(rx * rx, sigma)
    j_yy = ndi.gaussian_filter(ry * ry, sigma)
    j_xy = ndi.gaussian_filter(rx * ry, sigma)
    ang2 = np.arctan2(2 * j_xy, j_xx - j_yy)  # doubled angle
    coh = np.hypot(j_xx - j_yy, 2 * j_xy) / (j_xx + j_yy + 1e-12)
    return ang2 / 2, coh


def residues(phi):
    d = [wrap(phi[:-1, 1:] - phi[:-1, :-1]), wrap(phi[1:, 1:] - phi[:-1, 1:]),
         wrap(phi[1:, :-1] - phi[1:, 1:]), wrap(phi[:-1, :-1] - phi[1:, :-1])]
    return np.round(sum(d) / (2 * np.pi)).astype(np.int8)


def phase_with_linefield(e, rx, ry, sigma=2.0):
    """Phase relative to a per-pixel (arbitrarily signed) smoothed wave direction m."""
    ang, coh = structure_orientation(rx, ry, sigma)
    mx, my = np.cos(ang), np.sin(ang)
    phi = np.arctan2(rx * mx + ry * my, e)
    return phi, mx, my, coh


def residues_linefield(phi, mx, my):
    """Residues on the double cover: align (phi, m) ~ (-phi, -m) around each plaquette.

    Returns (charge, fold) where charge in {-1,0,1} is the phase circulation for
    orientable plaquettes and fold marks plaquettes whose line field is non-orientable
    (a half-integer orientation defect, i.e. a fold or crease of the layering).
    """
    corners = [(slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)),
               (slice(1, None), slice(1, None)), (slice(1, None), slice(None, -1))]
    P = [phi[c] for c in corners]
    MX = [mx[c] for c in corners]
    MY = [my[c] for c in corners]
    cur_p, cur_x, cur_y = P[0], MX[0], MY[0]
    total = np.zeros_like(P[0])
    for k in range(1, 5):
        j = k % 4
        s = np.sign(cur_x * MX[j] + cur_y * MY[j])
        s[s == 0] = 1
        pj, xj, yj = P[j] * s, MX[j] * s, MY[j] * s
        total += wrap(pj - cur_p)
        cur_p, cur_x, cur_y = pj, xj, yj
    # after the loop cur_* is corner 0 re-expressed; orientable iff it matches the original sign
    fold = (cur_x * MX[0] + cur_y * MY[0]) < 0
    charge = np.round(total / (2 * np.pi)).astype(np.int8)
    charge[fold] = 0
    return charge, fold


def poincare_index_line(theta, r=1):
    """Poincare index of a line field (angle mod pi) around each 2x2 plaquette: 0, +-1/2."""
    t2 = 2 * theta
    d = [wrap(t2[:-1, 1:] - t2[:-1, :-1]), wrap(t2[1:, 1:] - t2[:-1, 1:]),
         wrap(t2[1:, :-1] - t2[1:, 1:]), wrap(t2[:-1, :-1] - t2[1:, :-1])]
    return np.round(sum(d) / (2 * np.pi)) / 2
