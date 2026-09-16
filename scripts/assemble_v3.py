"""Bag v3 trainings from two fold splits into submission_v3.zip."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import train  # noqa: E402

RN_BOTH = "--rn-both" in sys.argv
runs = ROOT / "runs"
A, B = runs / "v3_s2026", runs / "v3_s7"
out = runs / "v3_bag"
shutil.rmtree(out, ignore_errors=True)
(out / "assets").mkdir(parents=True)
ca = json.loads((A / "assets" / "config.json").read_text())
cb = json.loads((B / "assets" / "config.json").read_text())
assert ca["feat_cols"] == cb["feat_cols"]
assert np.allclose(np.load(A / "assets" / "template.npy"), np.load(B / "assets" / "template.npy"))
assert all(p.get("adj") is not None for p in ca["lr"] + cb["lr"]), "expected voxel-size adjusted LR"
shutil.copy(A / "assets" / "template.npy", out / "assets" / "template.npy")
oa = pd.read_csv(A / "oof.csv", dtype={"uid": str})
ob = pd.read_csv(B / "oof.csv", dtype={"uid": str})
assert (oa.uid == ob.uid).all()

cfg = dict(feat_cols=ca["feat_cols"], folds=5, gbm=[], lr=ca["lr"] + cb["lr"], nets=[],
           groups={"cnn2d": ["effb0", "rn26d"]})
for tag, run, c in (("a", A, ca), ("b", B, cb)):
    for fn in c["gbm"]:
        shutil.copy(run / "assets" / fn, out / "assets" / f"{tag}_{fn}")
        cfg["gbm"].append(f"{tag}_{fn}")
nets = {}
for tag, run, c in (("a", A, ca), ("b", B, cb)):
    for n in c["nets"]:
        if tag == "b" and n["name"] == "rn26d" and not RN_BOTH:
            continue
        e = nets.setdefault(n["name"], {**n, "files": []})
        for fn in n["files"]:
            shutil.copy(run / "assets" / fn, out / "assets" / f"{tag}_{fn}")
            e["files"].append(f"{tag}_{fn}")
cfg["nets"] = list(nets.values())

o = oa[["uid", "y", "fold", "group"]].copy()
y = o.y.to_numpy()
for c in ("effb0", "rn26d", "gbm", "lr"):
    o[c] = oa[c] if (c == "rn26d" and not RN_BOTH) else 0.5 * (oa[c] + ob[c])
o["cnn2d"] = o[["effb0", "rn26d"]].mean(axis=1)
for d_ in (oa, ob):
    d_["cnn2d"] = d_[["effb0", "rn26d"]].mean(axis=1)
for c in ("effb0", "rn26d", "cnn2d", "gbm", "lr"):
    print(f"[{c}] split A {train.logloss(y, train.sigmoid(oa[c].values)):.4f} | split B "
          f"{train.logloss(y, train.sigmoid(ob[c].values)):.4f} | bagged {train.logloss(y, train.sigmoid(o[c].values)):.4f}"
          f" auc {train.auc(y, o[c].values):.4f}")
sa = train.fit_stack(oa, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
sb = train.fit_stack(ob, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
stack = train.fit_stack(o, ["cnn2d", "gbm", "lr"])
print(f"STACK nested OOF: split A {sa:.4f} | split B {sb:.4f} | bagged {stack['nested_logloss']:.4f} (rn26d both splits: {RN_BOTH})")
cfg["stack"] = stack
(out / "assets" / "config.json").write_text(json.dumps(cfg, indent=1))
o.to_csv(out / "oof.csv", index=False)

sub = ROOT / "submission_v3.zip"
with zipfile.ZipFile(sub, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(ROOT / "solution" / "main.py", "main.py")
    zf.write(ROOT / "solution" / "datlib.py", "datlib.py")
    for f in sorted((out / "assets").iterdir()):
        zf.write(f, f"assets/{f.name}")
print(sub, f"{sub.stat().st_size / 1e6:.1f} MB", len(zipfile.ZipFile(sub).namelist()), "entries")
