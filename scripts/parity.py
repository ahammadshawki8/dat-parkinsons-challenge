"""End-to-end parity: main.py smoke predictions vs predictions rebuilt from the training cache with the same assets."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json, sys
import numpy as np, pandas as pd, torch, lightgbm as lgb

ASSET_DIR = sys.argv[1]
PRED = sys.argv[2]
sys.path.insert(0, str(ROOT / "solution"))
import datlib  # noqa: E402

P = ROOT_S
cfg = json.load(open(ASSET_DIR + "/config.json"))
tab = pd.read_csv(P + "cache/table.csv", dtype={"uid": str})
F = pd.read_csv(P + "cache/features.csv")
R = np.load(P + "cache/reg_crops.npy", mmap_mode="r")
pred = pd.read_csv(PRED, dtype={"uid": str})
idx = np.array([tab.index[tab.uid == u][0] for u in pred.uid])
X = F.loc[idx, cfg["feat_cols"]].fillna(0).to_numpy(float)
zoom = tab.loc[idx, "zoom_x"].to_numpy(float)
cols = {"gbm": np.average([lgb.Booster(model_file=ASSET_DIR + "/" + f).predict(X, raw_score=True) for f in cfg["gbm"]], axis=0,
                         weights=cfg.get("gbm_weights") or [1.0] * len(cfg["gbm"])),
        "lr": np.average([datlib.lr_apply(p, X, zoom) for p in cfg["lr"]], axis=0,
                         weights=cfg.get("lr_weights") or [1.0] * len(cfg["lr"]))}
T = datlib.build_torch()
v = torch.from_numpy(np.asarray(R[idx])).unsqueeze(1).float().cuda()
for n in cfg["nets"]:
    acc = np.zeros(len(idx))
    fw = list(n.get("weights") or [1.0] * len(n["files"]))
    for fn, wt in zip(n["files"], fw):
        net = T["make_net"](n, pretrained=False)
        sd = torch.load(ASSET_DIR + "/" + fn, map_location="cpu", weights_only=True)
        net.load_state_dict({k: (t.float() if t.is_floating_point() else t) for k, t in sd.items()})
        net = net.cuda().eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            acc += wt * 0.5 * (net(T["prep_input"](T["center"](v))).float() + net(T["prep_input"](T["center"](v, flip=True))).float()).cpu().numpy()
    cols[n["name"]] = acc / float(sum(fw))
for g, m in cfg["groups"].items():
    gw = cfg.get("group_weights", {}).get(g, {})
    cols[g] = np.average([cols[k] for k in m], axis=0, weights=[float(gw.get(k, 1.0)) for k in m])
Z = np.stack([cols[c] for c in cfg["stack"]["cols"]], 1)
p = 1 / (1 + np.exp(-(Z @ np.array(cfg["stack"]["coef"]) + cfg["stack"]["b"]) / float(cfg["stack"].get("temp", 1.0))))
p = np.clip(p, max(cfg["stack"]["eps"], 1e-3), 1 - max(cfg["stack"]["eps"], 1e-3))
d = np.abs(p - pred.is_pathologic.to_numpy())
print(f"parity main.py vs training-cache pipeline: mean |diff| {d.mean():.4f}, max |diff| {d.max():.4f}")
