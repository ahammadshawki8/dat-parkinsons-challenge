from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys; sys.path.insert(0, 'solution')
import numpy as np, pandas as pd, train, datlib, lightgbm as lgb
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedGroupKFold

R = np.load('cache/reg_crops.npy', mmap_mode='r')
tab = pd.read_csv('cache/table.csv', dtype={'uid': str})
ZO = tab['zoom_x'].to_numpy()
TPL = np.load('cache/template.npy'); ATL = datlib.make_atlas(TPL)
NCC = pd.read_csv('cache/features.csv')['v_ncc'].to_numpy()

def feats(args):
    i, c = args
    v = np.asarray(R[i], dtype=np.float32)
    s_mm = np.sqrt(max(0.0, (c * 4.0) ** 2 - (c * ZO[i]) ** 2))
    if s_mm > 0.1:
        v = ndi.gaussian_filter(v, s_mm / datlib.SP)
    meta = tab.iloc[i].to_dict()
    f = datlib.features(v, meta); f.update(datlib.voi_features(v, ATL, NCC[i]))
    return f

if __name__ == '__main__':
    lab = pd.read_csv('data/train_labels.csv', dtype={'uid': str}).set_index('uid'); y = lab.loc[tab.uid, 'is_pathologic'].values
    rnd = pd.read_csv('runs/final/oof.csv', dtype={'uid': str}).fold.values
    fam = tab['zoom_x'].round(1).astype(str)
    grp = np.zeros(len(y), int)
    for f, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, fam)): grp[va] = f
    for c in (0.0, 0.8, 1.2):
        if c == 0:
            F = pd.read_csv('cache/features.csv')
        else:
            with ProcessPoolExecutor(4) as ex:
                F = pd.DataFrame(list(ex.map(feats, [(i, c) for i in range(len(y))], chunksize=16)))
            F.to_csv(f'cache/features_harm{c}.csv', index=False)
        cols = [k for k in F.columns if k not in train.NON_FEATURES]; X = F[cols].fillna(0).values
        for fname, folds in (('random', rnd), ('scanner-out', grp)):
            z = np.zeros(len(y)); g = np.zeros(len(y))
            for f in range(5):
                tr, va = folds != f, folds == f
                z[va] = train.apply_lr(train.fit_lr(X[tr], y[tr]), X[va])
                g[va] = lgb.LGBMClassifier(**train.DEFAULT_CFG['gbm'], verbose=-1, random_state=0).fit(X[tr], y[tr]).booster_.predict(X[va], raw_score=True)
            print(f'harmonise c={c} {fname:>11}: lr {train.logloss(y, train.sigmoid(z)):.4f}/{train.auc(y, z):.4f}  gbm {train.logloss(y, train.sigmoid(g)):.4f}/{train.auc(y, g):.4f}', flush=True)
