"""Whole-head context features (brain size, ventricle proxy, cortical uptake pattern) - age/atrophy cues."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

P = ROOT
sys.path.insert(0, str(P / "solution"))
import datlib  # noqa: E402

C = 4.0  # coarse mm


def one(args):
    uid, path = args
    try:
        data, zooms = datlib.load_canonical(path)
        sig = np.maximum(0.0, 0.5 * C / zooms - 0.35)
        src = ndi.gaussian_filter(data, sig) if np.any(sig > 0.05) else data
        v = ndi.zoom(src, zooms / C, order=1, prefilter=False, mode="nearest").astype(np.float32)
        sm = ndi.gaussian_filter(v, 1.0)
        hi = float(np.percentile(sm, 99.5))
        cl = np.minimum(sm, hi)
        m = cl > max(datlib._otsu(cl), 0.08 * hi)
        m = ndi.binary_fill_holes(ndi.binary_closing(datlib._largest_cc(ndi.binary_opening(m, iterations=1)), iterations=2))
        vals = sm[m]
        lo, h2 = np.percentile(vals, [25, 85])
        bg = float(vals[(vals >= lo) & (vals <= h2)].mean())
        x = sm / max(bg, 1e-6)
        zs = np.where(m.any(axis=(0, 1)))[0]
        ztop = int(zs.max())
        z1, z2 = max(0, ztop - int(110 / C)), max(1, ztop - int(20 / C))
        brain = np.zeros_like(m)
        brain[:, :, z1:z2] = m[:, :, z1:z2]          # head above the skull base ~ brain
        nb = int(brain.sum())
        vol_ml = nb * C ** 3 / 1000.0
        idx = np.argwhere(brain)
        ext = (idx.max(0) - idx.min(0) + 1) * C if nb else np.zeros(3)
        cent = idx.mean(0) if nb else np.zeros(3)
        # interior low-uptake fraction (ventricles / atrophy proxy): eroded brain voxels well below background
        inner = ndi.binary_erosion(brain, iterations=3)
        iv = x[inner] if inner.sum() > 50 else x[brain]
        out = dict(uid=uid, c_vol_ml=vol_ml, c_ext_x=ext[0], c_ext_y=ext[1], c_ext_z=ext[2],
                   c_low04=float((iv < 0.4).mean()), c_low06=float((iv < 0.6).mean()),
                   c_iqr=float(np.percentile(iv, 75) - np.percentile(iv, 25)),
                   c_cv=float(iv.std() / (iv.mean() + 1e-6)),
                   c_p10=float(np.percentile(iv, 10)), c_p90=float(np.percentile(iv, 90)))
        # anterior/posterior and superior/inferior cortical uptake balance (within brain, striatum excluded later)
        yy = idx[:, 1]
        zz = idx[:, 2]
        ant = brain & (np.arange(brain.shape[1])[None, :, None] > np.median(yy))
        pos = brain & (np.arange(brain.shape[1])[None, :, None] <= np.median(yy))
        sup = brain & (np.arange(brain.shape[2])[None, None, :] > np.median(zz))
        inf = brain & (np.arange(brain.shape[2])[None, None, :] <= np.median(zz))
        out["c_ant_post"] = float(x[ant].mean() / (x[pos].mean() + 1e-6))
        out["c_sup_inf"] = float(x[sup].mean() / (x[inf].mean() + 1e-6))
        out["c_top_mean"] = float(np.sort(x[brain])[::-1][:300].mean())
        out["c_ok"] = 1
        return out
    except Exception:
        return {"uid": uid, "c_ok": 0}


if __name__ == "__main__":
    tab = pd.read_csv(P / "cache/table.csv", dtype={"uid": str})
    files = {p.name.split(".")[0]: p for p in (P / "data/niftis").rglob("*.nii*")}
    args = [(u, files[u]) for u in tab.uid]
    rows = []
    with ProcessPoolExecutor(6) as ex:
        for i, r in enumerate(ex.map(one, args, chunksize=8)):
            rows.append(r)
            if (i + 1) % 400 == 0:
                print("  ", i + 1, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(P / "cache/context.csv", index=False)
    print("context features:", df.shape, "failures:", int((df.c_ok == 0).sum()))
