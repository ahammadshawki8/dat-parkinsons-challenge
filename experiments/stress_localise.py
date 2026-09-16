"""Stress-test the striatum localiser against realistic acquisition problems (CPU only).

Reports how far the detected striatal centre moves (mm) vs the clean scan. Large shifts = the crop
is wrong = confident wrong predictions on such test scans.
"""
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

N = 160


def centre_of(data, zooms):
    """Re-implements datlib.preprocess's localisation and returns the crop centre in mm."""
    C = datlib.COARSE
    sig = np.maximum(0.0, 0.5 * C / zooms - 0.35)
    src = ndi.gaussian_filter(data, sig) if np.any(sig > 0.05) else data
    small = ndi.zoom(src, zooms / C, order=1, prefilter=False, mode="nearest").astype(np.float32)
    scale = (np.array(data.shape) - 1) / np.maximum(np.array(small.shape) - 1, 1)
    sm = ndi.gaussian_filter(small, 1.0)
    hi = float(np.percentile(sm, 99.5))
    cl = np.minimum(sm, hi)
    m = cl > max(datlib._otsu(cl), 0.08 * hi)
    m = ndi.binary_fill_holes(ndi.binary_closing(datlib._largest_cc(ndi.binary_opening(m, iterations=1)), iterations=2))
    if m.sum() < 200:
        m = sm > np.percentile(sm, 60)
    zs = np.where(m.any(axis=(0, 1)))[0]
    ztop = int(zs.max())
    zlo = max(0, int(round(ztop - 125.0 / C)))
    zhi = max(zlo + 1, int(round(ztop - 25.0 / C)))
    slab = np.zeros_like(m)
    slab[:, :, zlo:zhi + 1] = m[:, :, zlo:zhi + 1]
    if slab.sum() < 50:
        slab = m
    cx, cy, _ = ndi.center_of_mass(slab)
    sm2 = ndi.gaussian_filter(small, 1.5)
    ii, jj, _ = np.meshgrid(*[np.arange(s) for s in small.shape], indexing="ij")
    allow = slab & (np.abs(ii - cx) * C <= 35.0) & (np.abs(jj - cy) * C <= 50.0)
    if allow.sum() < 10:
        allow = m
    px, py, pz = np.unravel_index(int(np.argmax(np.where(allow, sm2, -np.inf))), small.shape)
    zl2, zh2 = max(0, int(pz - 5)), int(pz + 6)
    band = np.zeros_like(m)
    band[:, :, zl2:zh2] = m[:, :, zl2:zh2]
    if band.sum() > 50:
        cx = ndi.center_of_mass(band)[0]
    return np.array([cx, py, pz]) * scale * zooms      # mm in original space


CORRUPTIONS = {
    "clean": lambda d, z: d,
    "top of head cut (30mm)": lambda d, z: d[:, :, :max(4, d.shape[2] - int(30 / z[2]))],
    "bottom cut (40mm)": lambda d, z: d[:, :, int(40 / z[2]):],
    "heavy smoothing (10mm)": lambda d, z: ndi.gaussian_filter(d, 10.0 / z / 2.355),
    "low counts (Poisson x0.25)": lambda d, z: np.random.poisson(np.clip(d, 0, None) * 0.25 / max(d.mean(), 1e-6) * 40).astype(np.float32),
    "head tilt 15 deg": lambda d, z: ndi.rotate(d, 15, axes=(1, 2), reshape=False, order=1),
    "yaw 20 deg": lambda d, z: ndi.rotate(d, 20, axes=(0, 1), reshape=False, order=1),
    "lateral FOV crop": lambda d, z: d[int(25 / z[0]):d.shape[0] - int(25 / z[0])],
    "hot spot outside brain": lambda d, z: _hot(d, z),
}


def _hot(d, z):
    d = d.copy()
    i, j, k = [int(s * f) for s, f in zip(d.shape, (0.5, 0.12, 0.25))]   # e.g. salivary gland / neck uptake
    r = [max(1, int(12 / zz)) for zz in z]
    d[i - r[0]:i + r[0], j - r[1]:j + r[1], k - r[2]:k + r[2]] = np.percentile(d, 99.9) * 1.6
    return d


def work(path):
    try:
        data, zooms = datlib.load_canonical(path)
        base = centre_of(data, zooms)
        out = {}
        rng = np.random.default_rng(0)
        for name, fn in CORRUPTIONS.items():
            if name == "clean":
                out[name] = 0.0
                continue
            try:
                d2 = np.asarray(fn(data, zooms), dtype=np.float32)
                z2 = zooms
                c2 = centre_of(d2, z2)
                # correct for the coordinate origin shift caused by cropping
                if "bottom cut" in name:
                    c2 = c2 + np.array([0, 0, 40.0])
                if "lateral" in name:
                    c2 = c2 + np.array([25.0, 0, 0])
                out[name] = float(np.linalg.norm(c2 - base))
            except Exception:
                out[name] = np.nan
        return out
    except Exception:
        return {}


if __name__ == "__main__":
    tab = pd.read_csv(P / "cache/table.csv", dtype={"uid": str})
    files = {p.name.split(".")[0]: p for p in (P / "data/niftis").rglob("*.nii*")}
    sel = tab["uid"].sample(N, random_state=0).tolist()
    with ProcessPoolExecutor(6) as ex:
        rows = [r for r in ex.map(work, [files[u] for u in sel]) if r]
    df = pd.DataFrame(rows)
    print(f"striatum localisation shift vs clean scan, mm  (n={len(df)})")
    print(f"{'corruption':>28} {'median':>8} {'p90':>8} {'worst':>8} {'>15mm':>8}")
    for c in df.columns:
        v = df[c].dropna()
        print(f"{c:>28} {v.median():8.1f} {v.quantile(0.9):8.1f} {v.max():8.1f} {(v > 15).mean() * 100:7.0f}%")
