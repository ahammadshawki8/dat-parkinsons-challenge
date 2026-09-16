"""DaT Parkinson's Challenge inference entrypoint.

Each test scan is preprocessed and scored independently. Logs contain only
progress counts (no information about individual test cases).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# no network in the runtime; never let any library try to reach a hub, never echo file paths via warnings
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
import datlib  # noqa: E402

DATA_DIR = Path(os.environ.get("DAT_DATA_DIR", "/code_execution/data"))
OUT_PATH = Path(os.environ.get(
    "DAT_OUT", "/code_execution/submission.csv" if Path("/code_execution").is_dir() else "submission.csv"))
ASSETS = ROOT / "assets"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} | {msg}", flush=True)


def _nifti_path(uid):
    for ext in (".nii.gz", ".nii"):
        p = DATA_DIR / "niftis" / f"{uid}{ext}"
        if p.exists():
            return p
    return DATA_DIR / "niftis" / f"{uid}.nii.gz"


def _work(uid):
    try:
        crop, meta = datlib.preprocess(_nifti_path(uid))
        return crop, meta, True
    except Exception:
        return None, None, False


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    t0 = time.time()
    cfg = json.loads((ASSETS / "config.json").read_text())
    stack = cfg["stack"]
    fmt = pd.read_csv(DATA_DIR / "submission_format.csv", dtype={"uid": str})
    uids = fmt["uid"].tolist()
    n = len(uids)
    log(f"scoring {n} examinations")

    # ---- preprocessing (CPU, one process per scan) -----------------------
    import multiprocessing as mp
    ctx = mp.get_context("fork") if sys.platform.startswith("linux") else None
    workers = max(1, min(22, (os.cpu_count() or 2) - 1))
    try:  # ~1.5 GB per worker is a safe upper bound for the largest volumes
        import psutil
        workers = max(1, min(workers, int(psutil.virtual_memory().available / 1.5e9)))
    except Exception:
        pass
    crops = np.ones((n,) + datlib.STORE, np.float16)
    metas = [None] * n
    ok = np.zeros(n, bool)
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        for i, (crop, meta, good) in enumerate(ex.map(_work, uids, chunksize=2)):
            if good:
                crops[i] = crop
                metas[i] = meta
                ok[i] = True
            if (i + 1) % 200 == 0:
                log(f"preprocessed {i + 1}/{n}")
    for i in np.where(~ok)[0]:  # retry failures sequentially (e.g. transient memory pressure)
        crop, meta, good = _work(uids[i])
        if good:
            crops[i], metas[i], ok[i] = crop, meta, True
    log(f"preprocessing done in {time.time() - t0:.0f}s with {workers} workers; unreadable={int((~ok).sum())}")
    dummy = dict(zoom_x=2.0, zoom_y=2.0, zoom_z=2.0, peak_bg=1.0)
    metas = [m if m is not None else dummy for m in metas]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = datlib.build_torch()
    template = np.load(ASSETS / "template.npy")
    atlas = datlib.make_atlas(template)
    crops, rows = datlib.postprocess(crops, metas, template, atlas, T, device,
                                     paths=[_nifti_path(u) for u in uids])
    X = np.array([[r.get(c, 0.0) for c in cfg["feat_cols"]] for r in rows], dtype=np.float64)
    X = np.nan_to_num(X)
    log(f"registration + features done in {time.time() - t0:.0f}s")

    cols = {}
    # ---- tabular ---------------------------------------------------------
    import lightgbm as lgb
    gw = np.asarray(cfg.get("gbm_weights") or [1.0] * len(cfg["gbm"]), float)
    cols["gbm"] = np.average([lgb.Booster(model_file=str(ASSETS / f)).predict(X, raw_score=True)
                              for f in cfg["gbm"]], axis=0, weights=gw)
    zoom = np.array([float(mm.get("zoom_x", 2.4)) for mm in metas])
    lw = np.asarray(cfg.get("lr_weights") or [1.0] * len(cfg["lr"]), float)
    cols["lr"] = np.average([datlib.lr_apply(p, X, zoom) for p in cfg["lr"]], axis=0, weights=lw)

    # ---- networks --------------------------------------------------------
    store = torch.from_numpy(crops).unsqueeze(1)
    for ncfg in cfg["nets"]:
        acc = np.zeros(n)
        fw = list(ncfg.get("weights") or [1.0] * len(ncfg["files"]))
        for fn, wt in zip(ncfg["files"], fw):
            net = T["make_net"](ncfg, pretrained=False)
            sd = torch.load(ASSETS / fn, map_location="cpu", weights_only=True)
            net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
            net = net.to(device).eval()
            fcols = ncfg.get("fusion_cols")
            Fall = None
            if fcols:
                Fall = torch.tensor(np.nan_to_num(np.array([[r.get(c, 0.0) for c in fcols] for r in rows], float)),
                                    dtype=torch.float32)
            with torch.no_grad():
                for b in range(0, n, 64):
                    v = store[b:b + 64].to(device).float()
                    kw = {} if Fall is None else {"feats": Fall[b:b + 64].to(device)}
                    with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                        wd = bool(ncfg.get("aug", {}).get("wide", False))
                        z = 0.5 * (net(T["prep_input"](T["center"](v, wide=wd)), **kw).float()
                                   + net(T["prep_input"](T["center"](v, flip=True, wide=wd)), **kw).float())
                    acc[b:b + 64] += wt * z.cpu().numpy()
            del net
        cols[ncfg["name"]] = acc / float(sum(fw))
        log(f"model {ncfg['name']} done ({len(ncfg['files'])} checkpoints)")

    for g, members in cfg.get("groups", {}).items():
        gwt = cfg.get("group_weights", {}).get(g, {})
        cols[g] = np.average([cols[mm] for mm in members], axis=0,
                             weights=[float(gwt.get(mm, 1.0)) for mm in members])
    Z = np.stack([cols[c] for c in stack["cols"]], 1)
    p = sigmoid((Z @ np.array(stack["coef"]) + stack["b"]) / float(stack.get("temp", 1.0)))
    eps = max(float(stack["eps"]), 1e-3)
    p = np.clip(p, eps, 1 - eps)
    p = np.where(ok & np.isfinite(p), p, stack["prior"])

    out = pd.DataFrame({"uid": uids, "is_pathologic": p.astype(float)})
    out.to_csv(OUT_PATH, index=False)
    log(f"wrote {OUT_PATH} with {len(out)} rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
