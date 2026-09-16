"""Simulate a harder test set and find the calibration that is best under it (aggregate numbers only)."""
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
o = pd.read_csv(P + "runs/bag/oof.csv", dtype={"uid": str})
y = o.y.to_numpy()
Z = o[["cnn2d", "gbm", "lr"]].to_numpy()
z = np.zeros(len(y))
for f in range(5):
    tr = o.fold.to_numpy() != f
    z[~tr] = LogisticRegression(C=0.1, max_iter=5000).fit(Z[tr], y[tr]).decision_function(Z[~tr])
print(f"clean OOF: logloss {train.logloss(y, train.sigmoid(z)):.4f} auc {train.auc(y, z):.4f}")
rng = np.random.default_rng(0)
p0 = train.sigmoid(z)


def evaluate(ys, zz):
    out = {}
    for T in (1.0, 1.1, 1.2, 1.3, 1.5):
        for eps in (0.0, 0.005, 0.01, 0.02, 0.03):
            out[(T, eps)] = train.logloss(ys, np.clip(train.sigmoid(zz / T), eps, 1 - eps))
    return out


scen = {
    "uniform random label flips": lambda k: rng.choice(len(y), k, replace=False),
    "flips among uncertain (0.05<p<0.95)": lambda k: rng.choice(np.where((p0 > 0.05) & (p0 < 0.95))[0], k, replace=False),
}
for name, pick in scen.items():
    print(f"\n=== scenario: {name}")
    for k in (0, 5, 10, 15, 20, 30):
        lls, aucs, tabs = [], [], []
        for rep in range(40):
            ys = y.copy()
            if k:
                idx = pick(k)
                ys[idx] = 1 - ys[idx]
            lls.append(train.logloss(ys, p0)); aucs.append(train.auc(ys, z)); tabs.append(evaluate(ys, z))
        mean_tab = {key: np.mean([t[key] for t in tabs]) for key in tabs[0]}
        best = min(mean_tab, key=mean_tab.get)
        print(f"flips={k:3d} ({k / len(y) * 100:.1f}%): auc {np.mean(aucs):.4f} logloss {np.mean(lls):.4f} | "
              f"best T={best[0]} eps={best[1]} -> {mean_tab[best]:.4f} (gain {np.mean(lls) - mean_tab[best]:.4f}) | "
              f"T=1.2,eps=.01 -> {mean_tab[(1.2, 0.01)]:.4f} | T=1.0,eps=.01 -> {mean_tab[(1.0, 0.01)]:.4f}")
