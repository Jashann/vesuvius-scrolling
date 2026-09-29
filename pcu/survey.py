"""One winding of a spiral fit -> snap to sheet -> isometric flatten -> surface volume (inner face first),
plus model-free views (mid layer, per-pixel max over the layers near the surface) for ink seen by eye."""
import numpy as np

from . import render, flatten, refine


def winding_volume(g, vol, surf, center_xy, layers=21, step=5.0, zweight=0.001, px=1.0):
    """g: (3, h, w) fit grid in the vol/surf voxel frame. Returns (surface volume (layers, H, W) uint8 with
    layer 0 = inner face, flattened grid (3, H/step, W/step), stats). px = output pixel and layer spacing in
    scan voxels (hecate 9.6 um on a 9.362 um scan: px = 9.6 / 9.362)."""
    gr, rs = refine.refine(g, surf, center_xy=center_xy)
    fg, fs = flatten.flatten(gr, step=step * px, zweight=zweight, sub=4)
    sv, vm = render.render(fg, vol, factor=int(step), layers=layers, spacing=px, center_xy=center_xy)
    return sv[::-1].copy(), fg, dict(refine=rs, flatten=fs)


def model_free(sv, band=3):
    """Views where ink can show without a model: the centre layer and the max over +-band layers."""
    c = sv.shape[0] // 2
    return sv[c], sv[c - band:c + band + 1].max(0)
