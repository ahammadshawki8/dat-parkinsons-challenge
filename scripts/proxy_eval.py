"""Leaderboard proxy: stacker fitted on random-fold OOF, applied to scanner-out predictions of the same recipe."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
sys.path.insert(0, str(ROOT / "solution"))
import numpy as np, pandas as pd, train
from sklearn.linear_model import LogisticRegression

P = ROOT_S
pairs = [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]
cols = ["cnn2d", "gbm", "lr"]
for rd_run, so_run in pairs:
    rd = pd.read_csv(P + f"runs/{rd_run}/oof.csv", dtype={"uid": str})
    so = pd.read_csv(P + f"runs/{so_run}/oof.csv", dtype={"uid": str})
    assert (rd.uid == so.uid).all()
    y = rd.y.to_numpy()
    nested = train.fit_stack(rd, cols, verbose=False)
    m = LogisticRegression(C=nested["C"], max_iter=5000).fit(rd[cols].to_numpy(), y)
    for T in (1.0, 1.1, 1.2):
        ps = train.sigmoid(m.decision_function(so[cols].to_numpy()) / T)
        print(f"{rd_run:>12} -> {so_run:<12} T={T}: random nested {nested['nested_logloss']:.4f} | "
              f"LB proxy {train.logloss(y, ps):.4f} auc {train.auc(y, ps):.4f} | weights {np.round(m.coef_[0], 3)}")
