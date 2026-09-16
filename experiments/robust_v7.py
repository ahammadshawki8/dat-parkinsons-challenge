"""Does the slice-sequence model improve robustness to unseen acquisition families? (v6-like vs v7-like)"""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

P = ROOT
sys.path.insert(0, str(P / "solution"))
import train  # noqa: E402

so_eff = pd.read_csv(P / "runs/so_strong/oof.csv", dtype={"uid": str})
so_dual = pd.read_csv(P / "runs/so_dual/oof.csv", dtype={"uid": str})
so_seq = pd.read_csv(P / "runs/so_seq/oof.csv", dtype={"uid": str})
rd = pd.read_csv(P / "runs/v7_bag/oof.csv", dtype={"uid": str})        # random-fold, deployed definitions
assert (so_eff.uid == rd.uid).all() and (so_dual.uid == rd.uid).all() and (so_seq.uid == rd.uid).all()
assert (so_dual.fold == so_seq.fold).all() and (so_eff.fold == so_seq.fold).all(), "scanner folds must match"
y = rd.y.to_numpy()

so = so_seq[["uid", "y", "fold", "group"]].copy()
so["effb0"], so["dual"], so["seq"] = so_eff["effb0"], so_dual["dual"], so_seq["seq"]
so["gbm"], so["lr"] = so_dual["gbm"], so_dual["lr"]

for tag, w in (("v6-like  (effb0 + dual)", {"effb0": 1, "dual": 1}),
               ("v7-like  (effb0 + dual + 2*seq)", {"effb0": 1, "dual": 1, "seq": 2})):
    s = so.copy()
    s["cnn2d"] = sum(v * so[k] for k, v in w.items()) / sum(w.values())
    nested = train.fit_stack(s, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
    r = rd.copy()
    r["cnn2d"] = sum(v * rd[k] for k, v in w.items()) / sum(w.values())
    m = LogisticRegression(C=0.1, max_iter=5000).fit(r[["cnn2d", "gbm", "lr"]].to_numpy(), y)
    proxy = train.logloss(y, m.predict_proba(s[["cnn2d", "gbm", "lr"]].to_numpy())[:, 1])
    print(f"{tag:>34}: held-out-scanner stack {nested:.4f} | LB proxy {proxy:.4f} | "
          f"CNN column {train.logloss(y, train.sigmoid(s['cnn2d'].to_numpy())):.4f}")
for k in ("effb0", "dual", "seq"):
    print(f"   single model on held-out scanners: {k:>5} {train.logloss(y, train.sigmoid(so[k].to_numpy())):.4f} auc {train.auc(y, so[k]):.4f}")
