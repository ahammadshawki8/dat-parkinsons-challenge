"""Match the simulated test difficulty to the actual LB score, then read off the best calibration."""
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
LB = {"v3": 0.2405}
o = pd.read_csv(P + "runs/v3_bag/oof.csv", dtype={"uid": str})
y = o.y.to_numpy()
Z = o[["cnn2d", "gbm", "lr"]].to_numpy()
z = np.zeros(len(y))
for f in range(5):
    tr = o.fold.to_numpy() != f
    z[~tr] = LogisticRegression(C=0.1, max_iter=5000).fit(Z[tr], y[tr]).decision_function(Z[~tr])
p0 = train.sigmoid(z)
print(f"v3 OOF: logloss {train.logloss(y, p0):.4f} auc {train.auc(y, z):.4f}  | actual LB {LB['v3']}")
rng = np.random.default_rng(0)
grid = [(T, e) for T in np.arange(1.0, 1.61, 0.05) for e in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03)]
for k in (0, 5, 8, 10, 12, 15, 18, 22):
    ll, au, tab = [], [], []
    for _ in range(60):
        ys = y.copy()
        if k:
            idx = rng.choice(len(y), k, replace=False)
            ys[idx] = 1 - ys[idx]
        ll.append(train.logloss(ys, p0)); au.append(train.auc(ys, z))
        tab.append([train.logloss(ys, np.clip(train.sigmoid(z / T), max(e, 1e-15), 1 - max(e, 1e-15))) for T, e in grid])
    m = np.mean(tab, 0)
    b = int(np.argmin(m))
    star = "  <== matches LB" if abs(np.mean(ll) - LB["v3"]) < 0.004 else ""
    print(f"flips={k:3d}: sim logloss {np.mean(ll):.4f} auc {np.mean(au):.4f} | best T={grid[b][0]:.2f} eps={grid[b][1]} -> {m[b]:.4f} "
          f"(gain {np.mean(ll) - m[b]:.4f}){star}")
# what each candidate correction costs on the CLEAN OOF (the downside if the test is not harder)
print("\ncost on clean OOF:")
for T in (1.0, 1.1, 1.15, 1.2, 1.3):
    for e in (0.0, 0.01):
        print(f"  T={T} eps={e}: {train.logloss(y, np.clip(train.sigmoid(z / T), max(e, 1e-15), 1 - max(e, 1e-15))):.4f}")
