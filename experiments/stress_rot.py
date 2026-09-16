"""True localisation error under head rotation (compares against where the striatum actually moved)."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

P = ROOT
sys.path.insert(0, str(P / "solution"))
import datlib  # noqa: E402

spec = importlib.util.spec_from_file_location("sl", str(P / "stress_localise.py"))
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)

CASES = (("head tilt 15 deg", 15, (1, 2)), ("head tilt 25 deg", 25, (1, 2)),
         ("yaw 20 deg", 20, (0, 1)), ("roll 15 deg", 15, (0, 2)))


def rot_expected(c_vox, shape, angle_deg, axes):
    c = (np.array(shape) - 1) / 2.0
    a = np.radians(angle_deg)
    R = np.eye(3)
    i, j = axes
    R[i, i] = np.cos(a); R[i, j] = -np.sin(a); R[j, i] = np.sin(a); R[j, j] = np.cos(a)
    return R @ (c_vox - c) + c


def work(path):
    try:
        data, zooms = datlib.load_canonical(path)
        base_vox = sl.centre_of(data, zooms) / zooms
        out = {}
        for name, ang, axes in CASES:
            d2 = np.asarray(ndi.rotate(data, ang, axes=axes, reshape=False, order=1), np.float32)
            got = sl.centre_of(d2, zooms) / zooms
            exp = rot_expected(base_vox, data.shape, ang, axes)
            out[name] = float(np.linalg.norm((got - exp) * zooms))
        return out
    except Exception:
        return {}


if __name__ == "__main__":
    tab = pd.read_csv(P / "cache/table.csv", dtype={"uid": str})
    files = {p.name.split(".")[0]: p for p in (P / "data/niftis").rglob("*.nii*")}
    sel = tab["uid"].sample(120, random_state=0).tolist()
    with ProcessPoolExecutor(5) as ex:
        rows = [r for r in ex.map(work, [files[u] for u in sel]) if r]
    df = pd.DataFrame(rows)
    print(f"TRUE localisation error after head rotation (mm), n={len(df)}")
    print(f"{'corruption':>20} {'median':>8} {'p90':>8} {'worst':>8} {'>15mm':>7}")
    for c in df.columns:
        v = df[c].dropna()
        print(f"{c:>20} {v.median():8.1f} {v.quantile(0.9):8.1f} {v.max():8.1f} {(v > 15).mean() * 100:6.0f}%")
