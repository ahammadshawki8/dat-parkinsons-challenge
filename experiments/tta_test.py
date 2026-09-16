"""Re-predict OOF with existing fold checkpoints under richer TTA (no retraining)."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json, math, sys
sys.path.insert(0, str(ROOT / "solution"))
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import datlib, train

P = ROOT_S
dev = "cuda"
T = datlib.build_torch()
R = np.load(P + "cache/reg_crops.npy", mmap_mode="r")
o = pd.read_csv(P + "runs/final/oof.csv", dtype={"uid": str})
cfg = json.load(open(P + "runs/final/assets/config.json"))
y = o.y.values
folds = o.fold.values
half_n = torch.tensor([datlib.NET[2], datlib.NET[1], datlib.NET[0]], dtype=torch.float32, device=dev) / 2
half_s = torch.tensor([datlib.STORE[2], datlib.STORE[1], datlib.STORE[0]], dtype=torch.float32, device=dev) / 2


def view(v, shift=(0, 0, 0), yaw=0.0, flip=False):
    B = v.shape[0]
    a = math.radians(yaw)
    Rm = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], device=dev)
    if flip:
        Rm = Rm * torch.tensor([-1.0, 1.0, 1.0], device=dev).view(1, 3)
    lin = Rm * half_n.view(1, 3) / half_s.view(3, 1)
    th = torch.cat([lin, (torch.tensor(shift, dtype=torch.float32, device=dev) / half_s).view(3, 1)], 1)
    th = th.unsqueeze(0).expand(B, -1, -1)
    g = F.affine_grid(th, (B, 1) + datlib.NET, align_corners=False)
    return F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False)


VIEWS = {
    "flip only (current)": [dict(), dict(flip=True)],
    "flip + shifts": [dict(flip=f, shift=s) for f in (False, True)
                      for s in ((0, 0, 0), (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 1), (0, 0, -1))],
    "flip + shifts + yaw": [dict(flip=f, shift=s, yaw=a) for f in (False, True)
                            for s in ((0, 0, 0), (2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0))
                            for a in (-4.0, 0.0, 4.0)],
}
res = {k: {n["name"]: np.zeros(len(y)) for n in cfg["nets"]} for k in VIEWS}
for n in cfg["nets"]:
    nf = len(n["files"]) // 5
    for fn in n["files"]:
        f = int(fn.split("_f")[1].split(".")[0])
        va = np.where(folds == f)[0]
        net = T["make_net"](n, pretrained=False)
        sd = torch.load(P + "runs/final/assets/" + fn, map_location="cpu", weights_only=True)
        net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()})
        net = net.to(dev).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for b in range(0, len(va), 48):
                ii = va[b:b + 48]
                v = torch.from_numpy(np.asarray(R[ii])).to(dev).float().unsqueeze(1)
                for k, views in VIEWS.items():
                    z = torch.stack([net(T["prep_input"](view(v, **vw))).float() for vw in views]).mean(0)
                    res[k][n["name"]][ii] += z.cpu().numpy() / nf
        del net
    print("done", n["name"], flush=True)

base = train.fit_stack(o, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
print(f"reference stack from saved OOF: {base:.4f}")
for k in VIEWS:
    oo = o.copy()
    for nm, z in res[k].items():
        oo[nm] = z
    oo["cnn2d"] = oo[["effb0", "rn26d"]].mean(axis=1)
    s = train.fit_stack(oo, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
    print(f"{k:>22}: cnn2d {train.logloss(y, train.sigmoid(oo.cnn2d.values)):.4f} auc {train.auc(y, oo.cnn2d.values):.4f} | stack {s:.4f}", flush=True)
