"""v6 = v5 ensemble + dual-normalised models (10-fold x3 seeds, plus a second split x3 seeds)."""
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

runs = ROOT / "runs"
BASE = runs / "v5_bag"
DUALS = [runs / "v6_dual10", runs / "v6_dual5b"]
out = runs / "v6_bag"
shutil.rmtree(out, ignore_errors=True)
(out / "assets").mkdir(parents=True)

cb = json.loads((BASE / "assets" / "config.json").read_text())
cds = [json.loads((d / "assets" / "config.json").read_text()) for d in DUALS]
assert all(c["feat_cols"] == cb["feat_cols"] for c in cds)
for f in (BASE / "assets").iterdir():          # keep v5 assets under their existing names
    shutil.copy(f, out / "assets" / f.name)
cfg = dict(feat_cols=cb["feat_cols"], folds=cb["folds"], gbm=list(cb["gbm"]), lr=list(cb["lr"]),
           nets=[dict(n) for n in cb["nets"]], groups={"cnn2d": ["effb0", "rn26d", "dual"]})
nets = {n["name"]: n for n in cfg["nets"]}
for tag, (d, c) in enumerate(zip(DUALS, cds)):
    t = f"d{tag}"
    for fn in c["gbm"]:
        shutil.copy(d / "assets" / fn, out / "assets" / f"{t}_{fn}")
        cfg["gbm"].append(f"{t}_{fn}")
    cfg["lr"] += c["lr"]
    for n in c["nets"]:
        e = nets.setdefault(n["name"], {**n, "files": []})
        if e is not n and "files" in e and e.get("_new") is None and n["name"] not in [x["name"] for x in cfg["nets"]]:
            e["files"] = []
            e["_new"] = True
            cfg["nets"].append(e)
        for fn in n["files"]:
            shutil.copy(d / "assets" / fn, out / "assets" / f"{t}_{fn}")
            e["files"].append(f"{t}_{fn}")
for n in cfg["nets"]:
    n.pop("_new", None)

ob = pd.read_csv(BASE / "oof.csv", dtype={"uid": str})
ods = [pd.read_csv(d / "oof.csv", dtype={"uid": str}) for d in DUALS]
assert all((d.uid == ob.uid).all() for d in ods)
o = ob[["uid", "y", "fold", "group"]].copy()
y = o.y.to_numpy()
o["effb0"], o["rn26d"] = ob["effb0"], ob["rn26d"]
o["dual"] = np.mean([d["dual"].to_numpy() for d in ods], axis=0)
o["gbm"] = np.mean([ob["gbm"].to_numpy()] + [d["gbm"].to_numpy() for d in ods], axis=0)
o["lr"] = np.mean([ob["lr"].to_numpy()] + [d["lr"].to_numpy() for d in ods], axis=0)
o["cnn2d"] = o[["effb0", "rn26d", "dual"]].mean(axis=1)
for c in ("effb0", "rn26d", "dual", "cnn2d", "gbm", "lr"):
    print(f"[{c}] OOF {train.logloss(y, train.sigmoid(o[c].to_numpy())):.4f} auc {train.auc(y, o[c]):.4f}")
best = None
for w in ([1, 1, 1], [1, 1, 2], [1, 0, 1], [2, 1, 2]):
    oo = o.copy()
    oo["cnn2d"] = (o[["effb0", "rn26d", "dual"]].to_numpy() * np.array(w)).sum(1) / sum(w)
    s = train.fit_stack(oo, ["cnn2d", "gbm", "lr"], verbose=False)
    print(f"  cnn weights effb0/rn26d/dual = {w}: stack {s['nested_logloss']:.4f}")
    if best is None or s["nested_logloss"] < best[0]:
        best = (s["nested_logloss"], w, s, oo)
ll, w, stack, o = best
print(f"v6 stack nested OOF {ll:.4f} with weights {w}  (v5 = 0.2025, v3 = 0.2035)")
cfg["group_weights"] = {"cnn2d": {"effb0": w[0], "rn26d": w[1], "dual": w[2]}}
cfg["stack"] = stack
(out / "assets" / "config.json").write_text(json.dumps(cfg, indent=1))
o.to_csv(out / "oof.csv", index=False)

sub = ROOT / "submission_v6.zip"
with zipfile.ZipFile(sub, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(ROOT / "solution" / "main.py", "main.py")
    zf.write(ROOT / "solution" / "datlib.py", "datlib.py")
    for f in sorted((out / "assets").iterdir()):
        zf.write(f, f"assets/{f.name}")
print(sub, f"{sub.stat().st_size / 1e6:.1f} MB", len(zipfile.ZipFile(sub).namelist()), "entries")
