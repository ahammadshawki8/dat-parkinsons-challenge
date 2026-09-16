"""Label-noise sample weights from INNER cross-validation inside each training fold (no validation leakage)."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold

P = ROOT
sys.path.insert(0, str(P / "solution"))
import train  # noqa: E402
import lightgbm as lgb  # noqa: E402

tab = pd.read_csv(P / "cache/table.csv", dtype={"uid": str})
F = pd.read_csv(P / "cache/features.csv")
o = pd.read_csv(P / "runs/v3_bag/oof.csv", dtype={"uid": str})
y = o.y.to_numpy()
folds = o.fold.to_numpy()
cols = [c for c in F.columns if c not in train.NON_FEATURES]
X = F[cols].fillna(0).to_numpy(float)
zoom = tab["zoom_x"].to_numpy(float)

W = np.ones((5, len(y)))          # weights[fold] = weight per scan when that fold is held out
contradict = np.zeros((5, len(y)))
for f in range(5):
    tr = np.where(folds != f)[0]
    inner = np.zeros(len(tr))
    skf = StratifiedKFold(4, shuffle=True, random_state=100 + f)
    for itr, iva in skf.split(tr, y[tr]):
        a, b = tr[itr], tr[iva]
        z1 = train.apply_lr(train.fit_lr(X[a], y[a], zoom=zoom[a]), X[b], zoom[b])
        z2 = lgb.LGBMClassifier(**train.DEFAULT_CFG["gbm"], verbose=-1, random_state=0).fit(X[a], y[a]).booster_.predict(X[b], raw_score=True)
        inner[iva] = 0.5 * (z1 + z2)
    p = train.sigmoid(inner)
    bad = np.abs(p - y[tr]) > 0.8          # image measurements strongly contradict the label
    contradict[f, tr] = bad
    W[f, tr] = np.where(bad, 0.3, 1.0)
    print(f"fold {f}: {int(bad.sum())}/{len(tr)} training scans down-weighted ({bad.mean() * 100:.1f}%)")
np.save(P / "cache/noise_weights.npy", W)
print("saved cache/noise_weights.npy", W.shape)
