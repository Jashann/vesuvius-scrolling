"""Partial inversion of Paganin phase retrieval on a reconstructed volume.

The eligible scans were reconstructed with Paganin delta/beta = 1000 at 1.2 m (plus sharpening). Paganin
divides each projection's spectrum by 1 + pi*lambda*z*(delta/beta)*|f|^2; for a parallel beam that is the
same isotropic filter on the reconstructed volume (Fourier slice theorem). Re-targeting delta/beta from
d_orig to d_new multiplies the volume spectrum by (1 + a*d_orig*f^2) / (1 + a*d_new*f^2), a = pi*lambda*z,
f in cycles/voxel scaled by the voxel size. d_new < d_orig restores fine texture (and amplifies noise)."""
import numpy as np

KEV_NM = 1.23984198  # lambda[nm] = 1.23984 / E[keV]


def alpha_vox2(kev, z_m, voxel_um):
    lam = KEV_NM / kev * 1e-9
    return np.pi * lam * z_m / (voxel_um * 1e-6) ** 2


def make_filter(kev=113.0, z_m=1.2, voxel_um=9.362, db_orig=1000.0, db_new=200.0, fmax_gain=None):
    a = alpha_vox2(kev, z_m, voxel_um)

    def f(box):
        if min(box.shape) < 4:
            return box
        valid = box > 0
        mean = box[valid].mean() if valid.any() else 0.0
        x = np.where(valid, box, mean).astype(np.float32)
        F = np.fft.rfftn(x)
        fz = np.fft.fftfreq(x.shape[0])[:, None, None]; fy = np.fft.fftfreq(x.shape[1])[None, :, None]
        fx = np.fft.rfftfreq(x.shape[2])[None, None, :]
        f2 = fz ** 2 + fy ** 2 + fx ** 2
        G = (1 + a * db_orig * f2) / (1 + a * db_new * f2)
        if fmax_gain is not None:
            G = np.minimum(G, fmax_gain)
        y = np.fft.irfftn(F * G, s=x.shape).astype(np.float32)
        return np.where(valid, np.clip(y, 1, 255), 0)
    return f
