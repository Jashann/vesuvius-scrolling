"""Run the public Lasagna 3D U-Net (scrollprize/lasagna, MIT) locally on CT crops at full output
resolution (the published volumes pool the cos channel 2x for storage)."""
import os, sys
import numpy as np
import torch

HERE = os.path.dirname(__file__)
LAS = os.path.join(HERE, "..", "ext", "villa", "lasagna")
sys.path[:0] = [LAS, os.path.join(HERE, "..", "ext", "villa", "vesuvius", "src")]
WEIGHTS = os.path.join(HERE, "..", "data", "models", "lasagna_model_current.pt")
_model = None


def model(device="mps"):
    global _model
    if _model is None:
        import train_unet_3d as t
        m, nt, up, sig = t.build_model(192, device, weights=WEIGHTS, strict=True)
        m.eval()
        _model = (m, sig, device)
    return _model


def predict(vol_u8, tile=192, overlap=64, device="mps"):
    """vol_u8: (D, H, W) uint8 CT crop. Returns float32 (8, D, H, W) in [0,1]:
    ch0 cos, ch1 grad_mag, ch2..7 direction encodings."""
    m, sig, dev = model(device)
    D, H, W = vol_u8.shape
    pad = [max(0, tile - s) for s in (D, H, W)]
    v = np.pad(vol_u8, [(0, p) for p in pad])
    D2, H2, W2 = v.shape
    out = np.zeros((8, D2, H2, W2), np.float32)
    wsum = np.zeros((D2, H2, W2), np.float32)
    step = tile - overlap
    def starts(n):
        s = list(range(0, max(n - tile, 0) + 1, step))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    ramp = np.minimum(ramp / (overlap / 2 + 1), 1.0)
    w3 = ramp[:, None, None] * ramp[None, :, None] * ramp[None, None, :]
    with torch.no_grad():
        for z in starts(D2):
            for y in starts(H2):
                for x in starts(W2):
                    t = torch.from_numpy(v[z:z + tile, y:y + tile, x:x + tile].astype(np.float32) / 255.0)[None, None].to(dev)
                    o = m(t)
                    o = o["output"] if isinstance(o, dict) else o
                    o = torch.sigmoid(o) if sig else o.clamp(0, 1)
                    o = o[0].float().cpu().numpy()
                    out[:, z:z + tile, y:y + tile, x:x + tile] += o * w3
                    wsum[z:z + tile, y:y + tile, x:x + tile] += w3
    out /= np.maximum(wsum, 1e-6)
    return out[:, :D, :H, :W]
