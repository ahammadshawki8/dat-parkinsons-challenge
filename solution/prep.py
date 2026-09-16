"""Preprocess every scan once into a cache: crops.npy + table.csv (meta + features)."""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datlib  # noqa: E402


def find_niftis(root: Path) -> dict:
    out = {}
    for p in root.rglob("*.nii*"):
        if p.is_file() and (p.name.endswith(".nii.gz") or p.name.endswith(".nii")):
            out[p.name.split(".")[0]] = p
    return out


def _work(args):
    uid, path = args
    try:
        crop, meta = datlib.preprocess(path)
        return uid, crop, {**meta, "ok": 1}
    except Exception as e:  # never crash the whole run
        return uid, np.ones(datlib.STORE, np.float16), {"ok": 0, "err": f"{type(e).__name__}"}


def run(nifti_root, uids, out_dir, workers=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = find_niftis(Path(nifti_root))
    missing = [u for u in uids if u not in files]
    if missing:
        raise FileNotFoundError(f"{len(missing)} labelled uids have no nifti file")
    workers = workers or max(1, (os.cpu_count() or 2))
    crops = np.zeros((len(uids),) + datlib.STORE, np.float16)
    rows = [None] * len(uids)
    t0 = time.time()
    idx = {u: i for i, u in enumerate(uids)}
    with ProcessPoolExecutor(workers) as ex:
        for n, (uid, crop, row) in enumerate(ex.map(_work, [(u, files[u]) for u in uids], chunksize=4)):
            crops[idx[uid]] = crop
            rows[idx[uid]] = {"uid": uid, **row}
            if (n + 1) % 200 == 0:
                print(f"  preprocessed {n + 1}/{len(uids)}  {time.time() - t0:.0f}s", flush=True)
    tab = pd.DataFrame(rows)
    np.save(out_dir / "crops.npy", crops)
    tab.to_csv(out_dir / "table.csv", index=False)
    print(f"done {len(uids)} scans in {time.time() - t0:.0f}s, failures={int((tab['ok'] == 0).sum())}")
    return crops, tab


if __name__ == "__main__":
    data = Path(sys.argv[1])
    out = Path(sys.argv[2])
    lab = pd.read_csv(data / "train_labels.csv", dtype={"uid": str})
    run(data / "niftis", lab["uid"].tolist(), out)
