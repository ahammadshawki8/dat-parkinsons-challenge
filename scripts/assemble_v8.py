"""v8/v9 = v6 (minus ResNet26d) + slice-sequence models bagged over two splits (+ optional wide variant).
One explicit weight table drives the OOF columns AND the deployed config; consistency is asserted.

usage: python assemble_v8.py OUT_NAME SEQ_WEIGHT [--wide WIDE_WEIGHT] [--scan]
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "solution"))
import train  # noqa: E402
import ensemble_weights as A  # noqa: E402

args = sys.argv[1:]
OUT_NAME = args[0]
SEQ_W = float(args[1])
WIDE_W = float(args[args.index("--wide") + 1]) if "--wide" in args else 0.0
SCAN = "--scan" in args
runs = ROOT / "runs"
V6 = runs / "v6_bag"
SEQ_A = [runs / "a08_seq", runs / "a08_seq_s12"]          # split 2026: seeds 0 | 1,2
SEQ_B = [runs / "v8_seq_b"]                                # split 7: seeds 0,1,2
WIDE = [runs / "v9_seqwide"] if WIDE_W > 0 else []

o = A.build(A.DEPLOY)                                      # effb0/dual/gbm/lr exactly as deployed from v6 files
y = o.y.to_numpy()


def seq_col(run_dirs, name):
    parts = []
    for d in run_dirs:
        df = pd.read_csv(d / "oof.csv", dtype={"uid": str})
        assert (df.uid == o.uid).all()
        c = json.loads((d / "assets" / "config.json").read_text())
        n = [x for x in c["nets"] if x["name"] == name][0]
        parts += [df[name].to_numpy()] * len(n["seeds"])    # seed-weighted within a split
    return np.mean(parts, axis=0)


o["seq"] = 0.5 * seq_col(SEQ_A, "seq") + 0.5 * seq_col(SEQ_B, "seq")   # equal split weights
if WIDE:
    o["seqw"] = seq_col(WIDE, "seqw")

if SCAN:
    print("group-weight scan (nested stack):")
    for sw in (1, 2, 3, 4, 6):
        for ww in ((0,) if not WIDE else (0, 1, 2)):
            gw = {"effb0": 1, "dual": 1, "seq": sw}
            if ww:
                gw["seqw"] = ww
            oo = o.copy()
            oo["cnn2d"] = sum(v * o[k] for k, v in gw.items()) / sum(gw.values())
            s = train.fit_stack(oo, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
            print(f"   seq {sw} wide {ww}: {s:.4f}")
    oo = o.copy()
    oo["cnn2d"] = (o["dual"] + o["seq"]) / 2
    print(f"   dual+seq only (no plain effb0): {train.fit_stack(oo, ['cnn2d','gbm','lr'], verbose=False)['nested_logloss']:.4f}")

out = runs / OUT_NAME
shutil.rmtree(out, ignore_errors=True)
(out / "assets").mkdir(parents=True)
c6 = json.loads((V6 / "assets" / "config.json").read_text())
cfg = {k: v for k, v in c6.items() if k != "stack"}
for f in (V6 / "assets").iterdir():
    if f.name != "config.json" and "rn26d" not in f.name:
        shutil.copy(f, out / "assets" / f.name)
cfg["nets"] = [dict(n) for n in c6["nets"] if n["name"] != "rn26d"]


def add_family(name, split_weights):
    """copy checkpoints; per-file weight = split weight / (#files in that split)"""
    files, weights, ncfg = [], [], None
    for (group, sw) in split_weights:
        group_files = []
        for d in group:
            c = json.loads((d / "assets" / "config.json").read_text())
            n = [x for x in c["nets"] if x["name"] == name][0]
            ncfg = n
            for fn in n["files"]:
                shutil.copy(d / "assets" / fn, out / "assets" / f"{d.name}_{fn}")
                group_files.append(f"{d.name}_{fn}")
        files += group_files
        weights += [sw / len(group_files)] * len(group_files)
    cfg["nets"].append({**ncfg, "files": files, "weights": weights})
    return np.array(weights)


w_seq = add_family("seq", [(SEQ_A, 0.5), (SEQ_B, 0.5)])
members = ["effb0", "dual", "seq"]
gw = {"effb0": 1.0, "dual": 1.0, "seq": SEQ_W}
if WIDE:
    add_family("seqw", [(WIDE, 1.0)])
    members.append("seqw")
    gw["seqw"] = WIDE_W
cfg["groups"] = {"cnn2d": members}
cfg["group_weights"] = {"cnn2d": gw}

# consistency: each split's files carry exactly its split weight, and seeds within a split are equal
assert abs(w_seq.sum() - 1.0) < 1e-9 and len(w_seq) == 30, "expected 30 seq checkpoints (2 splits x 3 seeds x 5 folds)"
assert np.allclose(w_seq[:15], 0.5 / 15) and np.allclose(w_seq[15:], 0.5 / 15)
print("A01 consistency check passed: config weights == OOF construction")

o["cnn2d"] = sum(gw[m] * o[m] for m in members) / sum(gw.values())
for c in members + ["cnn2d", "gbm", "lr"]:
    print(f"[{c}] OOF {train.logloss(y, train.sigmoid(o[c].to_numpy())):.4f} auc {train.auc(y, o[c]):.4f}")
stack = train.fit_stack(o, ["cnn2d", "gbm", "lr"], verbose=False)
print(f"{OUT_NAME}: nested OOF {stack['nested_logloss']:.4f}  (v7 0.1988, v6 0.2014)  weights {gw}")
cfg["stack"] = stack
(out / "assets" / "config.json").write_text(json.dumps(cfg, indent=1))
o.to_csv(out / "oof.csv", index=False)

sub = ROOT / f"submission_{OUT_NAME}.zip"
with zipfile.ZipFile(sub, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(ROOT / "solution" / "main.py", "main.py")
    zf.write(ROOT / "solution" / "datlib.py", "datlib.py")
    for f in sorted((out / "assets").iterdir()):
        zf.write(f, f"assets/{f.name}")
print(sub, f"{sub.stat().st_size / 1e6:.1f} MB", len(zipfile.ZipFile(sub).namelist()), "entries")
