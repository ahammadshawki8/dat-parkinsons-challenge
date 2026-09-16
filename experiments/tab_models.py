from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
sys.path.insert(0, str(ROOT / "solution"))
import numpy as np, pandas as pd, train
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold

P = ROOT_S
o = pd.read_csv(P + "runs/final/oof.csv", dtype={"uid": str})
y = o.y.values
folds = o.fold.values
tab = pd.read_csv(P + "cache/table.csv", dtype={"uid": str})
F = pd.read_csv(P + "cache/features.csv")
E = pd.read_csv(P + "cache/extra_feats.csv")
cols = [c for c in F.columns if c not in train.NON_FEATURES]
X = F[cols].fillna(0).values
XE = pd.concat([F[cols], E], axis=1).fillna(0).values
fam = tab["zoom_x"].round(1).astype(str)
grp = np.zeros(len(y), int)
for f, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, fam)):
    grp[va] = f


def clipz(A):
    return np.clip(A, -6, 6)


def cv(make, Xm, fl):
    z = np.zeros(len(y))
    for f in range(5):
        tr, va = fl != f, fl == f
        sc = StandardScaler().fit(Xm[tr])
        m = make().fit(clipz(sc.transform(Xm[tr])), y[tr])
        Zv = clipz(sc.transform(Xm[va]))
        z[va] = m.decision_function(Zv) if hasattr(m, "decision_function") else train.logit(m.predict_proba(Zv)[:, 1])
    return z


models = {
    "lr_extra": (lambda: __import__("sklearn.linear_model", fromlist=["x"]).LogisticRegression(C=0.05, max_iter=5000), XE),
    "svm": (lambda: SVC(C=0.5, gamma="scale"), X),
    "svm_extra": (lambda: SVC(C=0.5, gamma="scale"), XE),
    "mlp": (lambda: MLPClassifier((32,), alpha=1.0, max_iter=2000, random_state=0), X),
    "et": (lambda: ExtraTreesClassifier(500, min_samples_leaf=5, max_features=0.3, n_jobs=6, random_state=0), X),
}
for name, (mk, Xm) in models.items():
    z = cv(mk, Xm, folds)
    zc = train.sigmoid(z)
    # 1-D calibration inside nested CV for fair single-model logloss
    tmp = o[["y", "fold", "group"]].copy()
    tmp[name] = z
    s1 = train.fit_stack(tmp, [name], verbose=False)
    o[name] = z
    s = train.fit_stack(o, ["cnn2d", "gbm", "lr", name], verbose=False)
    zg = cv(mk, Xm, grp)
    tmpg = pd.DataFrame({"y": y, "fold": grp, "group": o.group, name: zg})
    sg = train.fit_stack(tmpg, [name], verbose=False)
    print(f"{name:>10}: alone(calibrated) {s1['nested_logloss']:.4f} auc {train.auc(y, z):.4f} | scanner-out alone {sg['nested_logloss']:.4f} | stack+this {s['nested_logloss']:.4f}", flush=True)
base = train.fit_stack(o, ["cnn2d", "gbm", "lr"], verbose=False)
print("reference stack (cnn2d,gbm,lr): %.4f" % base["nested_logloss"])
o.to_csv(P + "runs/final/oof_tab_extra.csv", index=False)
