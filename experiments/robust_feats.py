from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
sys.path.insert(0, str(ROOT / "solution"))
import numpy as np, pandas as pd, train, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold

P = ROOT_S
tab = pd.read_csv(P + "cache/table.csv", dtype={"uid": str})
F = pd.read_csv(P + "cache/features.csv")
lab = pd.read_csv(P + "data/train_labels.csv", dtype={"uid": str}).set_index("uid")
y = lab.loc[tab.uid, "is_pathologic"].values
cols = [c for c in F.columns if c not in train.NON_FEATURES]
zoom = tab["zoom_x"].to_numpy()
fam = tab["zoom_x"].round(1).astype(str)
so = np.zeros(len(y), int)
for f, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, fam)):
    so[va] = f
rnd = pd.read_csv(P + "runs/final/oof.csv", dtype={"uid": str}).fold.values

ratio_cols = [c for c in cols if any(k in c for k in ("asym", "put_caud", "post_ant", "post_caud", "box_ratio", "mid_ant", "elong", "ext", "cen"))]
abs_cols = [c for c in cols if c not in ratio_cols]


def resid(Xtr, ytr, ztr, Xte, zte):
    """remove the voxel-size trend of each feature, fitted on TRAINING normals only (quadratic in zoom)."""
    m = ytr == 0
    A = np.stack([np.ones(m.sum()), ztr[m], ztr[m] ** 2], 1)
    coef, *_ = np.linalg.lstsq(A, Xtr[m], rcond=None)
    def apply(X, z):
        return X - np.stack([np.ones(len(z)), z, z ** 2], 1) @ coef
    return apply(Xtr, ztr), apply(Xte, zte)


def run(names, folds, adjust=False, add_zoom=False):
    X = F[names].fillna(0).to_numpy(float)
    z_lr, z_gb = np.zeros(len(y)), np.zeros(len(y))
    for f in range(5):
        tr, va = folds != f, folds == f
        Xtr, Xva = X[tr], X[va]
        if adjust:
            Xtr, Xva = resid(Xtr, y[tr], zoom[tr], Xva, zoom[va])
        if add_zoom:
            Xtr = np.column_stack([Xtr, zoom[tr]]); Xva = np.column_stack([Xva, zoom[va]])
        z_lr[va] = train.apply_lr(train.fit_lr(Xtr, y[tr]), Xva)
        z_gb[va] = lgb.LGBMClassifier(**train.DEFAULT_CFG["gbm"], verbose=-1, random_state=0).fit(Xtr, y[tr]).booster_.predict(Xva, raw_score=True)
    return z_lr, z_gb


def show(tag, **kw):
    out = []
    for fname, folds in (("random", rnd), ("scanner-out", so)):
        zl, zg = run(**kw, folds=folds)
        out.append(f"{fname}: lr {train.logloss(y, train.sigmoid(zl)):.4f}/{train.auc(y, zl):.3f} gbm {train.logloss(y, train.sigmoid(zg)):.4f}/{train.auc(y, zg):.3f}")
    print(f"{tag:>34} | " + " | ".join(out), flush=True)


print(f"features: all={len(cols)} ratio/shape={len(ratio_cols)} absolute={len(abs_cols)}")
show("all (current)", names=cols)
show("ratio/shape only", names=ratio_cols)
show("absolute only", names=abs_cols)
show("all, voxel-size adjusted", names=cols, adjust=True)
show("all + zoom as feature", names=cols, add_zoom=True)
