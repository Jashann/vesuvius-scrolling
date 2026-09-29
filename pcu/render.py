"""Surface-volume rendering from a tifxyz-style grid (3, h, w in x, y, z voxel coords of a CT level):
densify, compute outward normals, sample `layers` slices along the normal from the CT, tile by tile
(each tile reads only its own small bounding box)."""
import numpy as np
from scipy import ndimage as ndi

from . import data


def densify(grid, factor):
    """Bilinear upsampling of a (3, h, w) coordinate grid; invalid cells (x <= 0) stay invalid."""
    valid = grid[0] > 0
    g = grid.copy()
    # fill invalid cells with nearest valid values so interpolation near holes is sane
    if not valid.all():
        _, (iy, ix) = ndi.distance_transform_edt(~valid, return_indices=True)
        g = g[:, iy, ix]
    up = np.stack([ndi.zoom(g[a], factor, order=1) for a in range(3)])  # factor: float or (fy, fx)
    vm = ndi.zoom(valid.astype(np.uint8), factor, order=0).astype(bool)
    return up, vm[: up.shape[1], : up.shape[2]]


def normals(P, center_xy=None):
    """P: (3, H, W) x/y/z. Unit normals from the cross product of grid derivatives; if center_xy is
    given, oriented away from it (outward from the umbilicus)."""
    du = np.stack(np.gradient(P, axis=2))  # along columns
    dv = np.stack(np.gradient(P, axis=1))  # along rows
    n = np.cross(du, dv, axis=0)
    n /= np.linalg.norm(n, axis=0, keepdims=True) + 1e-9
    if center_xy is not None:
        radial = np.stack([P[0] - center_xy[0], P[1] - center_xy[1], np.zeros_like(P[0])])
        flip = np.sign(np.sum(n * radial, axis=0, keepdims=True)); flip[flip == 0] = 1
        # orient each connected region by majority to avoid speckle
        n = n * np.sign(np.median(flip))
    return n


def render(grid, vol, factor, layers=21, spacing=1.0, center_xy=None, tile=256, ct_scale=1.0, box_filter=None, pad=2):
    """grid: (3, h, w) coordinates in the CT level `vol` voxel frame times ct_scale (coords / ct_scale
    gives voxel indices). Returns (layers, H, W) uint8 surface volume and validity mask.
    box_filter: optional f(box float32) -> box applied to each CT read box (e.g. de-Paganin); pad = extra
    voxels read around each box so the filter has context."""
    P, vm = densify(grid, factor)
    P = P / ct_scale
    n = normals(P, None if center_xy is None else (center_xy[0] / ct_scale, center_xy[1] / ct_scale))
    H, W = P.shape[1:]
    out = np.zeros((layers, H, W), np.uint8)
    offs = (np.arange(layers) - (layers - 1) / 2) * spacing
    MAXV = 48_000_000  # max voxels per CT read box (~190 MB float32): folded/stretched tiles are split

    def do(r0, r1, c0, c1):
        sl = (slice(r0, r1), slice(c0, c1))
        m = vm[sl]
        if not m.any():
            return
        p = P[(slice(None),) + sl]; nn = n[(slice(None),) + sl]
        pts = p[:, None] + nn[:, None] * offs[None, :, None, None]  # (3, L, h, w) in x, y, z
        flat = pts[:, :, m]  # bounding box from valid cells only
        lo = np.floor(flat.reshape(3, -1).min(1)).astype(int) - pad
        hi = np.ceil(flat.reshape(3, -1).max(1)).astype(int) + pad + 1
        lo = np.maximum(lo, 0); hi = np.minimum(hi, [vol.shape[2], vol.shape[1], vol.shape[0]])
        if (hi <= lo).any():
            return
        if np.prod(hi - lo) > MAXV:
            if r1 - r0 <= 16 and c1 - c0 <= 16:
                return  # pathological cell (fold): leave empty
            rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
            for a, b in ((r0, rm), (rm, r1)):
                for c, d in ((c0, cm), (cm, c1)):
                    if b > a and d > c:
                        do(a, b, c, d)
            return
        box = data.read(vol, (slice(lo[2], hi[2]), slice(lo[1], hi[1]), slice(lo[0], hi[0]))).astype(np.float32)
        if box_filter is not None:
            box = box_filter(box)
        co = [pts[2] - lo[2], pts[1] - lo[1], pts[0] - lo[0]]
        v = ndi.map_coordinates(box, [c.reshape(-1) for c in co], order=1, mode="nearest").reshape(pts.shape[1:])
        v[:, ~m] = 0
        out[(slice(None),) + sl] = np.clip(v, 0, 255).astype(np.uint8)

    for r0 in range(0, H, tile):
        for c0 in range(0, W, tile):
            do(r0, min(r0 + tile, H), c0, min(c0 + tile, W))
    return out, vm
