"""Correct evaluation of multi-registration TTA: single split (v3_s2026) models on THEIR OWN folds."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch, lightgbm as lgb

P = ROOT
sys.path.insert(0, str(P / "solution"))
import datlib, train  # noqa: E402

dev = "cuda"
T = datlib.build_torch()
RUN = P / "runs/v3_s2026"
cfg = json.load(open(RUN / "assets/config.json"))
o = pd.read_csv(RUN / "oof.csv", dtype={"uid": str})
tab = pd.read_csv(P / "cache/table.csv", dtype={"uid": str})
y = o.y.to_numpy()
folds = o.fold.to_numpy()
zoom = tab["zoom_x"].to_numpy(float)
F2 = pd.read_csv(P / "cache/features_alt.csv")
ncc2 = np.load(P / "cache/ncc_alt.npy")
REG2 = np.load(P / "cache/reg_crops_alt.npy", mmap_mode="r")
X2 = F2[cfg["feat_cols"]].fillna(0).to_numpy(float)

# sanity: every asset in this run belongs to ONE fold partition (no cross-split mixing)
assert len(cfg["gbm"]) == 5 and len(cfg["lr"]) == 5, "expected a single-split run"

alt = {}
for name in ("gbm", "lr"):
    z = np.zeros(len(y))
    for f in range(5):
        va = folds == f
        if name == "gbm":
            z[va] = lgb.Booster(model_file=str(RUN / "assets" / cfg["gbm"][f])).predict(X2[va], raw_score=True)
        else:
            z[va] = datlib.lr_apply(cfg["lr"][f], X2[va], zoom[va])
    alt[name] = z
store2 = torch.from_numpy(np.asarray(REG2)).unsqueeze(1)
for n in cfg["nets"]:
    z = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for fn in n["files"]:
        f = int(fn.split("_f")[1].split(".")[0])
        va = np.where(folds == f)[0]
        net = T["make_net"](n, pretrained=False)
        sd = torch.load(RUN / "assets" / fn, map_location="cpu", weights_only=True)
        net.load_state_dict({k: (t_.float() if t_.is_floating_point() else t_) for k, t_ in sd.items()})
        net = net.to(dev).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(va), 64):
                ii = va[b:b + 64]
                v = store2[ii].to(dev).float()
                z[ii] += 0.5 * (net(T["prep_input"](T["center"](v))).float()
                                + net(T["prep_input"](T["center"](v, flip=True))).float()).cpu().numpy()
                cnt[ii] += 1
        del net
    alt[n["name"]] = z / np.maximum(cnt, 1)
alt = pd.DataFrame(alt)
alt["cnn2d"] = alt[["effb0", "rn26d"]].mean(axis=1)
avg = o.copy()
print(f"{'':>10}  {'original':>9} {'alt-reg':>9} {'averaged':>9}")
for c in ("cnn2d", "gbm", "lr"):
    avg[c] = 0.5 * (o[c].to_numpy() + alt[c].to_numpy())
    print(f"{c:>10}: {train.logloss(y, train.sigmoid(o[c].to_numpy())):9.4f} "
          f"{train.logloss(y, train.sigmoid(alt[c].to_numpy())):9.4f} "
          f"{train.logloss(y, train.sigmoid(avg[c].to_numpy())):9.4f}")
s = [train.fit_stack(d, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
     for d in (o, alt.assign(y=y, fold=folds, group=o.group), avg)]
print(f"{'STACK':>10}: {s[0]:9.4f} {s[1]:9.4f} {s[2]:9.4f}")
