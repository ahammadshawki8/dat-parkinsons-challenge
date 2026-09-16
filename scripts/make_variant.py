"""Build a v8-family variant with different CNN-group weights (weight 0 drops the family and its files).

usage: python make_variant.py OUT_NAME effb0=W dual=W seq=W
"""
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

OUT = sys.argv[1]
W = {k: float(v) for k, v in (a.split("=") for a in sys.argv[2:])}
V8 = ROOT / "runs/v8"
cfg = json.loads((V8 / "assets/config.json").read_text())
o = pd.read_csv(V8 / "oof.csv", dtype={"uid": str})
y = o.y.to_numpy()
keep = [m for m in cfg["groups"]["cnn2d"] if W.get(m, cfg["group_weights"]["cnn2d"][m]) > 0]
gw = {m: W.get(m, cfg["group_weights"]["cnn2d"][m]) for m in keep}
drop_files = {f for n in cfg["nets"] if n["name"] not in keep for f in n["files"]}

out = ROOT / "runs" / OUT
shutil.rmtree(out, ignore_errors=True)
(out / "assets").mkdir(parents=True)
for f in (V8 / "assets").iterdir():
    if f.name != "config.json" and f.name not in drop_files:
        shutil.copy(f, out / "assets" / f.name)
cfg["nets"] = [n for n in cfg["nets"] if n["name"] in keep]
cfg["groups"] = {"cnn2d": keep}
cfg["group_weights"] = {"cnn2d": gw}
o["cnn2d"] = sum(w * o[m] for m, w in gw.items()) / sum(gw.values())
stack = train.fit_stack(o, ["cnn2d", "gbm", "lr"], verbose=False)
cfg["stack"] = stack
(out / "assets/config.json").write_text(json.dumps(cfg, indent=1))
o.to_csv(out / "oof.csv", index=False)
missing = [f for n in cfg["nets"] for f in n["files"] if not (out / "assets" / f).exists()]
assert not missing, f"missing files: {missing[:3]}"
sub = ROOT / f"submission_{OUT}.zip"
with zipfile.ZipFile(sub, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(ROOT / "solution/main.py", "main.py")
    zf.write(ROOT / "solution/datlib.py", "datlib.py")
    for f in sorted((out / "assets").iterdir()):
        zf.write(f, f"assets/{f.name}")
print(f"{OUT}: families {gw} | nested OOF {stack['nested_logloss']:.4f} (v8 0.1990) | "
      f"{sub.stat().st_size / 1e6:.0f} MB, {len(zipfile.ZipFile(sub).namelist())} entries")
