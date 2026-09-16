from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys; sys.path.insert(0,'solution')
import numpy as np, pandas as pd, train, datlib, lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
R = np.load('cache/reg_crops.npy', mmap_mode='r')
tpl = np.load('cache/template.npy')

def make_atlas2(template, n_side):
    lab = datlib.make_atlas(template, n_side)
    return lab

def extra(i):
    v = datlib._center(np.asarray(R[i], dtype=np.float32), datlib.NET)
    out = {}
    for ns, lab in ATL.items():
        ref = float(v[lab == 9].mean()) + 1e-3
        per = {}
        for side, nm in ((0,'a'),(3,'b')):
            d = {}
            put = np.argwhere((lab == side+2) | (lab == side+3))
            vals = v[tuple(put.T)] / ref - 1
            ys = put[:,1]
            qs = np.quantile(ys, [0.25, 0.5, 0.75])
            bins = np.digitize(ys, qs)
            for b in range(4):
                d[f'p{b}'] = float(vals[bins == b].mean())
            cz = np.argwhere(lab == side+1); cv = v[tuple(cz.T)]/ref - 1
            d['c'] = float(cv.mean())
            d['c_top'] = float(np.sort(cv)[::-1][:len(cv)//3].mean())
            d['p_top'] = float(np.sort(vals)[::-1][:len(vals)//3].mean())
            d['slope'] = float(np.polyfit(ys.astype(float), vals, 1)[0])
            per[nm] = d
        for k in per['a']:
            a, b = per['a'][k], per['b'][k]
            out[f'w{ns}_{k}_min'] = min(a,b); out[f'w{ns}_{k}_max'] = max(a,b)
            out[f'w{ns}_{k}_asym'] = abs(a-b)/(abs(a)+abs(b)+1e-3)
        out[f'w{ns}_ref'] = ref
    return out

ATL = {ns: make_atlas2(tpl, ns) for ns in (400, 1100)}
if __name__ == '__main__':
    with ProcessPoolExecutor(12) as ex:
        rows = list(ex.map(extra, range(len(R)), chunksize=20))
    E = pd.DataFrame(rows); E.to_csv('cache/extra_feats.csv', index=False)
    o = pd.read_csv('runs/r2d/oof.csv', dtype={'uid':str}); y=o.y.values; folds=o.fold.values
    F = pd.read_csv('cache/features.csv')
    for name, X in [('base', F.fillna(0).values), ('base+extra', pd.concat([F, E], axis=1).fillna(0).values)]:
        for C in (0.05, 0.1):
            z=np.zeros(len(y))
            for f in range(5):
                tr,va=folds!=f,folds==f; z[va]=train.apply_lr(train.fit_lr(X[tr],y[tr],C),X[va])
            print(name, 'lr C', C, round(train.logloss(y,train.sigmoid(z)),4), round(train.auc(y,z),4))
            o[f'lr_{name}_{C}'] = z
        z=np.zeros(len(y))
        for f in range(5):
            tr,va=folds!=f,folds==f
            z[va]=lgb.LGBMClassifier(**train.DEFAULT_CFG['gbm'],verbose=-1,random_state=f).fit(X[tr],y[tr]).booster_.predict(X[va],raw_score=True)
        print(name, 'gbm', round(train.logloss(y,train.sigmoid(z)),4), round(train.auc(y,z),4)); o[f'gbm_{name}'] = z
    train.fit_stack(o, ['effb0', 'gbm_base+extra', 'lr_base+extra_0.1'])
