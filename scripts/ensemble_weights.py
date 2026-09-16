"""A01: evaluated (assembly) vs deployed (equal-per-checkpoint) ensemble weights for v6.

Every column is a genuine out-of-fold prediction from its own run, so both definitions are valid
held-out ensembles. Only whole-set scalar summaries are printed.
"""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import json, sys
from pathlib import Path

import numpy as np, pandas as pd

P = ROOT
sys.path.insert(0, str(P / "solution"))
import train  # noqa: E402

R = {k: pd.read_csv(P / f"runs/{k}/oof.csv", dtype={"uid": str})
     for k in ("e_10fold", "v5_10f", "v3_s2026", "v3_s7", "v6_dual10", "v6_dual5b")}
base = R["v3_s2026"]
assert all((d.uid == base.uid).all() for d in R.values())
y = base.y.to_numpy()

# (run, column, weight) as the assembly scripts combined them
EVAL = {
    "effb0": [("e_10fold", 1/6), ("v5_10f", 1/3), ("v3_s2026", 1/4), ("v3_s7", 1/4)],
    "rn26d": [("v5_10f", 1/2), ("v3_s2026", 1/2)],
    "dual":  [("v6_dual10", 1/2), ("v6_dual5b", 1/2)],
    "gbm":   [("e_10fold", 1/12), ("v5_10f", 1/12), ("v3_s2026", 1/12), ("v3_s7", 1/12), ("v6_dual10", 1/3), ("v6_dual5b", 1/3)],
}
EVAL["lr"] = EVAL["gbm"]
# as main.py deploys them: equal weight per checkpoint / model file
N_FILES = {("effb0", "e_10fold"): 10, ("effb0", "v5_10f"): 20, ("effb0", "v3_s2026"): 15, ("effb0", "v3_s7"): 15,
           ("rn26d", "v5_10f"): 10, ("rn26d", "v3_s2026"): 15,
           ("dual", "v6_dual10"): 30, ("dual", "v6_dual5b"): 15}
TAB_FILES = {"e_10fold": 10, "v5_10f": 10, "v3_s2026": 5, "v3_s7": 5, "v6_dual10": 10, "v6_dual5b": 5}
DEPLOY = {}
for fam in ("effb0", "rn26d", "dual"):
    runs = [r for r, _ in EVAL[fam]]
    tot = sum(N_FILES[(fam, r)] for r in runs)
    DEPLOY[fam] = [(r, N_FILES[(fam, r)] / tot) for r in runs]
tot = sum(TAB_FILES.values())
DEPLOY["gbm"] = [(r, n / tot) for r, n in TAB_FILES.items()]
DEPLOY["lr"] = DEPLOY["gbm"]


def build(defn):
    o = base[["uid", "y", "fold", "group"]].copy()
    for fam, parts in defn.items():
        assert abs(sum(w for _, w in parts) - 1) < 1e-9
        o[fam] = sum(w * R[r][fam].to_numpy() for r, w in parts)
    o["cnn2d"] = o[["effb0", "rn26d", "dual"]].mean(axis=1)
    return o


print(f"{'family':>6} | {'max |weight diff| evaluated vs deployed':>40}")
for fam in EVAL:
    d = {r: w for r, w in EVAL[fam]}
    e = {r: w for r, w in DEPLOY[fam]}
    print(f"{fam:>6} | {max(abs(d[r] - e[r]) for r in d):40.3f}")
for name, defn in (("as evaluated (assembly)", EVAL), ("as deployed (main.py)", DEPLOY)):
    o = build(defn)
    s = train.fit_stack(o, ["cnn2d", "gbm", "lr"], verbose=False)
    print(f"{name:>26}: stack nested {s['nested_logloss']:.4f}")
oe, od = build(EVAL), build(DEPLOY)
diff = np.abs(train.sigmoid(oe[['cnn2d', 'gbm', 'lr']].to_numpy().mean(1)) - train.sigmoid(od[['cnn2d', 'gbm', 'lr']].to_numpy().mean(1)))
print(f"whole-set mean |prob difference| between the two definitions: {diff.mean():.4f}")
