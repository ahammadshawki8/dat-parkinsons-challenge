"""Model soup: average the WEIGHTS of the 3 seeds per fold, compare with averaging their predictions."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch

P = ROOT
sys.path.insert(0, str(P / "solution"))
import datlib, train  # noqa: E402

dev = "cuda"
T = datlib.build_torch()
A = P / "runs/v3_bag/assets"
cfg = json.load(open(A / "config.json"))
o = pd.read_csv(P / "runs/v3_bag/oof.csv", dtype={"uid": str})
R = np.load(P / "cache/reg_crops.npy", mmap_mode="r")
y = o.y.to_numpy()
folds = o.fold.to_numpy()          # split A partition; only use split-A ("a_") checkpoints
store = torch.from_numpy(np.asarray(R)).unsqueeze(1)

for n in cfg["nets"]:
    files = [f for f in n["files"] if f.startswith("a_")]
    if len(files) < 10:
        continue
    soup = np.zeros(len(y))
    for f in range(5):
        fold_files = [fn for fn in files if fn.endswith(f"_f{f}.pt")]
        if len(fold_files) < 2:
            continue
        sds = [torch.load(A / fn, map_location="cpu", weights_only=True) for fn in fold_files]
        avg = {}
        for k in sds[0]:
            if sds[0][k].is_floating_point():
                avg[k] = torch.stack([sd[k].float() for sd in sds]).mean(0)
            else:
                avg[k] = sds[0][k]
        net = T["make_net"](n, pretrained=False)
        net.load_state_dict(avg)
        net = net.to(dev).eval()
        va = np.where(folds == f)[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(va), 64):
                ii = va[b:b + 64]
                v = store[ii].to(dev).float()
                soup[ii] = 0.5 * (net(T["prep_input"](T["center"](v))).float()
                                  + net(T["prep_input"](T["center"](v, flip=True))).float()).cpu().numpy()
        del net
    print(f"[{n['name']}] {len(files)} split-A checkpoints ({len(files)//5} seeds x 5 folds)")
    print(f"    prediction averaging (current): {train.logloss(y, train.sigmoid(o[n['name']].to_numpy())):.4f} "
          f"auc {train.auc(y, o[n['name']]):.4f}")
    print(f"    weight averaging (model soup):  {train.logloss(y, train.sigmoid(soup)):.4f} auc {train.auc(y, soup):.4f}")
