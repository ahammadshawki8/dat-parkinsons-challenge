"""A03 probability vs logit pooling, A04 constrained calibration — doubly nested, whole-set scalars only."""
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
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / "solution"))
import train  # noqa: E402
import ensemble_weights as A  # noqa: E402  (reuses the deployed-definition columns)

o = A.build(A.DEPLOY)
y = o.y.to_numpy()
folds = o.fold.to_numpy()
FAM = ["cnn2d", "gbm", "lr"]
Z = o[FAM].to_numpy()
eps = 1e-6


def sig(z):
    return 1 / (1 + np.exp(-z))


def ll(yy, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))


def stack_fit(Zt, yt):
    return LogisticRegression(C=0.1, max_iter=5000).fit(Zt, yt)


def prob_pool_fit(Zt, yt, shrink=0.02):
    """non-negative weights summing to 1 on member PROBABILITIES, shrunk toward equal weights"""
    Pt = sig(Zt)
    k = Pt.shape[1]
    def obj(a):
        w = np.exp(a) / np.exp(a).sum()
        return ll(yt, Pt @ w) + shrink * np.sum((w - 1 / k) ** 2)
    a = minimize(obj, np.zeros(k), method="Nelder-Mead", options={"maxiter": 2000}).x
    return np.exp(a) / np.exp(a).sum()


def calib_fit(kind, s, yt):
    """s: logit of the base prediction on inner held-out data"""
    if kind == "identity":
        return lambda q: q
    if kind == "temperature":
        f = lambda t: ll(yt, sig(s / t[0]))
        t = minimize(f, [1.0], bounds=[(0.5, 3.0)], method="L-BFGS-B").x[0]
        return lambda q: q / t
    if kind == "affine":
        f = lambda ab: ll(yt, sig(ab[0] * s + ab[1])) + 1e-3 * ((ab[0] - 1) ** 2 + ab[1] ** 2)
        a, b = minimize(f, [1.0, 0.0], bounds=[(0.2, 5.0), (-3, 3)], method="L-BFGS-B").x
        return lambda q: a * q + b
    if kind == "beta":
        p = np.clip(sig(s), eps, 1 - eps)
        f = lambda v: ll(yt, sig(v[0] * np.log(p) - v[1] * np.log(1 - p) + v[2])) + 1e-3 * ((v[0] - 1) ** 2 + (v[1] - 1) ** 2 + v[2] ** 2)
        a, b, c = minimize(f, [1.0, 1.0, 0.0], bounds=[(0, 10), (0, 10), (-5, 5)], method="L-BFGS-B").x
        def g(q):
            pq = np.clip(sig(q), eps, 1 - eps)
            return a * np.log(pq) - b * np.log(1 - pq) + c
        return g
    raise ValueError(kind)


KINDS = ["identity", "temperature", "affine", "beta"]
res = {f"stack+{k}": np.zeros(len(y)) for k in KINDS}
res["prob-pool"] = np.zeros(len(y))
res["logit-mean (equal)"] = np.zeros(len(y))
res["prob-mean (equal)"] = np.zeros(len(y))
for f in np.unique(folds):
    tr, va = folds != f, folds == f
    Zt, yt = Z[tr], y[tr]
    # inner OOF stack logits for fitting calibrators (never touches the outer fold)
    inner = np.zeros(tr.sum())
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=10 + int(f)).split(Zt, yt):
        inner[iva] = stack_fit(Zt[itr], yt[itr]).decision_function(Zt[iva])
    m = stack_fit(Zt, yt)
    outer = m.decision_function(Z[va])
    for k in KINDS:
        res[f"stack+{k}"][va] = sig(calib_fit(k, inner, yt)(outer))
    w = prob_pool_fit(Zt, yt)
    res["prob-pool"][va] = sig(Z[va]) @ w
    res["logit-mean (equal)"][va] = sig(Z[va].mean(1))
    res["prob-mean (equal)"][va] = sig(Z[va]).mean(1)

print("doubly-nested held-out log loss (whole training set):")
for k, p in res.items():
    print(f"  {k:>22}: {ll(y, p):.4f}")
base = res["stack+identity"]
print("probability floor applied separately to the stack:")
for fl in (0.0, 0.002, 0.005, 0.01):
    print(f"  floor {fl:<6}: {ll(y, np.clip(base, fl, 1 - fl)):.4f}")
