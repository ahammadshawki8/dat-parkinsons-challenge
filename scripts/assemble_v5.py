"""v5 = 10-fold ensemble (each CNN trained on 90% of the data), optionally combined with v3's 5-fold models."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json, shutil, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import train  # noqa: E402

ADD_V3 = "--with-v3" in sys.argv          # also ship v3's 5-fold checkpoints (bigger zip, more diversity)
runs = ROOT / "runs"
SRC = [runs / "e_10fold", runs / "v5_10f"]
out = runs / "v5_bag"
shutil.rmtree(out, ignore_errors=True)
(out / "assets").mkdir(parents=True)

cfgs = [json.loads((d / "assets" / "config.json").read_text()) for d in SRC]
oofs = [pd.read_csv(d / "oof.csv", dtype={"uid": str}) for d in SRC]
assert all((o.uid == oofs[0].uid).all() and (o.fold == oofs[0].fold).all() for o in oofs), "fold partitions differ"
assert cfgs[0]["feat_cols"] == cfgs[1]["feat_cols"]
shutil.copy(SRC[0] / "assets" / "template.npy", out / "assets" / "template.npy")

cfg = dict(feat_cols=cfgs[0]["feat_cols"], folds=10, gbm=[], lr=[], nets=[],
           groups={"cnn2d": ["effb0", "rn26d"]})
nets = {}
for tag, (d, c) in enumerate(zip(SRC, cfgs)):
    t = f"t{tag}"
    for fn in c["gbm"]:
        shutil.copy(d / "assets" / fn, out / "assets" / f"{t}_{fn}")
        cfg["gbm"].append(f"{t}_{fn}")
    cfg["lr"] += c["lr"]
    for n in c["nets"]:
        e = nets.setdefault(n["name"], {**n, "files": []})
        for fn in n["files"]:
            shutil.copy(d / "assets" / fn, out / "assets" / f"{t}_{fn}")
            e["files"].append(f"{t}_{fn}")

o = oofs[0][["uid", "y", "fold", "group"]].copy()
y = o.y.to_numpy()
seeds = {"effb0": [], "rn26d": [], "gbm": [], "lr": []}
for c, df in zip(cfgs, oofs):
    for n in c["nets"]:
        w = len(n["seeds"])
        seeds[n["name"]] += [df[n["name"]].to_numpy()] * w      # weight by number of seeds averaged in that run
    seeds["gbm"].append(df["gbm"].to_numpy())
    seeds["lr"].append(df["lr"].to_numpy())
for k, v in seeds.items():
    o[k] = np.mean(v, axis=0)

if ADD_V3:
    v3 = pd.read_csv(runs / "v3_bag" / "oof.csv", dtype={"uid": str})
    assert (v3.uid == o.uid).all()
    c3 = json.loads((runs / "v3_bag" / "assets" / "config.json").read_text())
    for n in c3["nets"]:
        e = nets.setdefault(n["name"], {**n, "files": []})
        for fn in n["files"]:
            shutil.copy(runs / "v3_bag" / "assets" / fn, out / "assets" / f"v3_{fn}")
            e["files"].append(f"v3_{fn}")
    for fn in c3["gbm"]:
        shutil.copy(runs / "v3_bag" / "assets" / fn, out / "assets" / f"v3_{fn}")
        cfg["gbm"].append(f"v3_{fn}")
    cfg["lr"] += c3["lr"]
    for col in ("effb0", "rn26d", "gbm", "lr"):
        o[col] = 0.5 * (o[col].to_numpy() + v3[col].to_numpy())   # 10-fold and 5-fold OOF are both honest
cfg["nets"] = list(nets.values())

o["cnn2d"] = o[["effb0", "rn26d"]].mean(axis=1)
for c in ("effb0", "rn26d", "cnn2d", "gbm", "lr"):
    print(f"[{c}] OOF {train.logloss(y, train.sigmoid(o[c].to_numpy())):.4f} auc {train.auc(y, o[c]):.4f}")
stack = train.fit_stack(o, ["cnn2d", "gbm", "lr"], verbose=False)
print(f"v5 stack nested OOF {stack['nested_logloss']:.4f} auc {stack.get('auc', float('nan'))}  (v3 = 0.2035)  with_v3={ADD_V3}")
cfg["stack"] = stack
(out / "assets" / "config.json").write_text(json.dumps(cfg, indent=1))
o.to_csv(out / "oof.csv", index=False)

sub = ROOT / "submission_v5.zip"
with zipfile.ZipFile(sub, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(ROOT / "solution" / "main.py", "main.py")
    zf.write(ROOT / "solution" / "datlib.py", "datlib.py")
    for f in sorted((out / "assets").iterdir()):
        zf.write(f, f"assets/{f.name}")
print(sub, f"{sub.stat().st_size / 1e6:.1f} MB", len(zipfile.ZipFile(sub).namelist()), "entries")
