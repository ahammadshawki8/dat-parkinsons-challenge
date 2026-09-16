"""A24: disagreement-conditioned shrinkage toward the training prior, doubly nested. Whole-set scalars only."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
from pathlib import Path

import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

P = ROOT
sys.path.insert(0, str(P / "solution"))
import train  # noqa: E402

o = pd.read_csv(P / "runs/v8/oof.csv", dtype={"uid": str})
y = o.y.to_numpy()
folds = o.fold.to_numpy()
GW = {"effb0": 1.0, "dual": 1.0, "seq": 2.0}
o["cnn2d"] = sum(w * o[k] for k, w in GW.items()) / sum(GW.values())
Z = o[["cnn2d", "gbm", "lr"]].to_numpy()
D = np.column_stack([o[["effb0", "dual", "seq", "gbm", "lr"]].to_numpy().std(1)])   # family disagreement


def ll(yy, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_alpha(pz, d, yt, prior, conditional):
    def alpha(v, dd):
        return sig(v[0] + (v[1] * dd if conditional else 0.0)) * 0.5      # capped at 0.5
    def obj(v):
        a = alpha(v, dd=d)
        return ll(yt, (1 - a) * pz + a * prior) + 1e-3 * np.sum(v ** 2)
    v = minimize(obj, np.array([-4.0, 0.0]), method="Nelder-Mead").x
    return lambda pp, dd: (1 - alpha(v, dd)) * pp + alpha(v, dd) * prior


res = {"stack": np.zeros(len(y)), "constant shrink": np.zeros(len(y)), "disagreement shrink": np.zeros(len(y))}
for f in np.unique(folds):
    tr, va = folds != f, folds == f
    prior = y[tr].mean()
    inner = np.zeros(tr.sum())
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=20 + int(f)).split(Z[tr], y[tr]):
        inner[iva] = LogisticRegression(C=0.1, max_iter=5000).fit(Z[tr][itr], y[tr][itr]).decision_function(Z[tr][iva])
    m = LogisticRegression(C=0.1, max_iter=5000).fit(Z[tr], y[tr])
    pv = sig(m.decision_function(Z[va]))
    res["stack"][va] = pv
    for name, cond in (("constant shrink", False), ("disagreement shrink", True)):
        g = fit_alpha(sig(inner), D[tr][:, 0], y[tr], prior, cond)
        res[name][va] = g(pv, D[va][:, 0])
for k, p in res.items():
    print(f"{k:>20}: {ll(y, p):.4f}")
