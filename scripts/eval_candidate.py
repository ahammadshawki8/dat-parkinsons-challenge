"""Score round-3 candidates: paired single-model comparison and marginal gain over the v8 ensemble."""
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

v8 = pd.read_csv(P / "runs/v8/oof.csv", dtype={"uid": str})
y = v8.y.to_numpy()
ref = {"dual s0": pd.read_csv(P / "runs/e_dual/oof.csv", dtype={"uid": str})["dual"].to_numpy(),
       "seq s0": pd.read_csv(P / "runs/a08_seq/oof.csv", dtype={"uid": str})["seq"].to_numpy()}
GW = {"effb0": 1.0, "dual": 1.0, "seq": 2.0}


def nested_loss(df, cols):
    z = np.zeros(len(y))
    f = df.fold.to_numpy()
    for k in np.unique(f):
        tr = f != k
        z[~tr] = LogisticRegression(C=0.1, max_iter=5000).fit(df[cols].to_numpy()[tr], y[tr]).decision_function(df[cols].to_numpy()[~tr])
    p = train.sigmoid(z)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


base_df = v8.copy()
base_df["cnn2d"] = sum(w * v8[k] for k, w in GW.items()) / sum(GW.values())
base = nested_loss(base_df, ["cnn2d", "gbm", "lr"])
print(f"v8 reference ensemble: {base.mean():.4f}")
for k, v in ref.items():
    print(f"  reference single model {k}: {train.logloss(y, train.sigmoid(v)):.4f} auc {train.auc(y, v):.4f}")
rng = np.random.default_rng(0)
for run, col in [a.split(":") for a in sys.argv[1:]]:
    f = P / f"runs/{run}/oof.csv"
    if not f.exists():
        print(f"{run}: not finished")
        continue
    c = pd.read_csv(f, dtype={"uid": str})[col].to_numpy()
    line = f"{run:>10} ({col}): alone {train.logloss(y, train.sigmoid(c)):.4f} auc {train.auc(y, c):.4f}"
    for w in (1.0, 2.0):
        df = v8.copy()
        df["new"] = c
        gw = dict(GW, new=w)
        df["cnn2d"] = sum(ww * df[k] for k, ww in gw.items()) / sum(gw.values())
        d = nested_loss(df, ["cnn2d", "gbm", "lr"]) - base
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
        line += f" | +w{w:.0f}: {d.mean():+.4f} ({(bs < 0).mean() * 100:.0f}% improve)"
    print(line)
