"""How much does the MODEL degrade when test scans are messier? (existing v3 checkpoints, OOF folds)"""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn.functional as Fn, lightgbm as lgb
from scipy import ndimage as ndi

P = ROOT
sys.path.insert(0, str(P / "solution"))
import datlib, train  # noqa: E402

dev = "cuda"
T = datlib.build_torch()
A = P / "runs/v3_bag/assets"
cfg = json.load(open(A / "config.json"))
o = pd.read_csv(P / "runs/v3_bag/oof.csv", dtype={"uid": str})
tab = pd.read_csv(P / "cache/table.csv", dtype={"uid": str})
C = np.load(P / "cache/crops.npy", mmap_mode="r")
y = o.y.to_numpy()
folds = o.fold.to_numpy()
zoom = tab["zoom_x"].to_numpy(float)
tpl = np.load(A / "template.npy")
atlas = datlib.make_atlas(tpl)
rng = np.random.default_rng(0)
SUB = np.sort(rng.choice(len(y), 400, replace=False))       # 400 scans keeps it quick


def corrupt(vol, kind):
    v = vol.astype(np.float32)
    if kind == "clean":
        return v
    if kind == "blur 8mm":
        return ndi.gaussian_filter(v, 8.0 / datlib.SP / 2.355)
    if kind == "noise x3":
        return np.clip(v + rng.normal(0, 0.3 * v.mean(), v.shape), 0, None)
    if kind == "tilt 12deg":
        return ndi.rotate(v, 12, axes=(0, 1), reshape=False, order=1)
    if kind == "shift 10mm":
        return ndi.shift(v, (0, 5, 5), order=1)
    if kind == "intensity x0.7":
        return v * 0.7
    if kind == "crop edges":
        v = v.copy(); v[:6] = 0; v[-6:] = 0
        return v
    raise ValueError(kind)


def predict(regs, nccs, idx):
    rows = [dict(datlib.features(regs[i], tab.iloc[idx[i]].to_dict()),
                 **datlib.voi_features(regs[i], atlas, nccs[i])) for i in range(len(idx))]
    X = np.array([[r.get(c, 0.0) for c in cfg["feat_cols"]] for r in rows], float)
    cols = {}
    z = np.zeros(len(idx)); zl = np.zeros(len(idx))
    for f in range(5):
        m = folds[idx] == f
        if not m.any():
            continue
        z[m] = np.mean([lgb.Booster(model_file=str(A / fn)).predict(X[m], raw_score=True)
                        for fn in cfg["gbm"] if fn.endswith(f"_f{f}.txt")], 0)
        zl[m] = np.mean([datlib.lr_apply(p, X[m], zoom[idx][m]) for i, p in enumerate(cfg["lr"]) if i % 5 == f], 0)
    cols["gbm"], cols["lr"] = z, zl
    store = torch.from_numpy(np.asarray(regs)).unsqueeze(1)
    for n in cfg["nets"]:
        acc = np.zeros(len(idx)); cnt = np.zeros(len(idx))
        for fn in n["files"]:
            f = int(fn.split("_f")[1].split(".")[0])
            m = np.where(folds[idx] == f)[0]
            if not len(m):
                continue
            net = T["make_net"](n, pretrained=False)
            sd = torch.load(A / fn, map_location="cpu", weights_only=True)
            net.load_state_dict({k: (t.float() if t.is_floating_point() else t) for k, t in sd.items()})
            net = net.to(dev).eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for b in range(0, len(m), 64):
                    ii = m[b:b + 64]
                    v = store[ii].to(dev).float()
                    acc[ii] += 0.5 * (net(T["prep_input"](T["center"](v))).float()
                                      + net(T["prep_input"](T["center"](v, flip=True))).float()).cpu().numpy()
                    cnt[ii] += 1
            del net
        cols[n["name"]] = acc / np.maximum(cnt, 1)
    cnn = np.mean([cols[n["name"]] for n in cfg["nets"]], 0)
    st = cfg["stack"]
    Z = np.stack([{"cnn2d": cnn, "gbm": cols["gbm"], "lr": cols["lr"]}[c] for c in st["cols"]], 1)
    return train.sigmoid(Z @ np.array(st["coef"]) + st["b"])


print(f"{'corruption':>16} {'logloss':>9} {'auc':>7} {'vs clean':>9}")
base = None
for kind in ("clean", "blur 8mm", "noise x3", "tilt 12deg", "shift 10mm", "intensity x0.7", "crop edges"):
    crops = np.stack([corrupt(np.asarray(C[i]), kind) for i in SUB]).astype(np.float16)
    regs, nccs = T["register_all"](crops, tpl, dev)
    p = predict(regs, nccs, SUB)
    ll = train.logloss(y[SUB], p)
    if base is None:
        base = ll
    print(f"{kind:>16} {ll:9.4f} {train.auc(y[SUB], p):7.4f} {ll - base:+9.4f}", flush=True)
