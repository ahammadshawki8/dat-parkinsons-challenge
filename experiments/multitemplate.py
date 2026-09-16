"""Multi-template registration (normal / left-reduced / right-reduced / bilateral-reduced), best NCC wins."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
sys.path.insert(0, str(ROOT / "solution"))
import numpy as np, pandas as pd, torch, train, datlib, lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import StratifiedGroupKFold

P = ROOT_S
C = np.load(P + "cache/crops.npy", mmap_mode="r")
R = np.load(P + "cache/reg_crops.npy", mmap_mode="r")
tab = pd.read_csv(P + "cache/table.csv", dtype={"uid": str})
T0 = np.load(P + "cache/template.npy")
ATL = datlib.make_atlas(T0)
d = [(a - b) // 2 for a, b in zip(datlib.STORE, datlib.NET)]


def feats(args):
    i, path = args
    reg = np.load(path, mmap_mode="r")
    meta = tab.iloc[i].to_dict()
    f = datlib.features(np.asarray(reg[i]), meta)
    f.update(datlib.voi_features(np.asarray(reg[i]), ATL, NCCB[i]))
    return f


if __name__ == "__main__":
    lab = pd.read_csv(P + "data/train_labels.csv", dtype={"uid": str}).set_index("uid")
    y = lab.loc[tab.uid, "is_pathologic"].values
    T = datlib.build_torch()
    dev = "cuda"
    # signed side difference of striatal uptake in the existing registration (side a: x < W/2)
    net = R[:, d[0]:d[0] + datlib.NET[0], d[1]:d[1] + datlib.NET[1], d[2]:d[2] + datlib.NET[2]]
    sa = np.array([np.asarray(net[i], np.float32)[(ATL >= 1) & (ATL <= 3)].mean() for i in range(len(y))])
    sb = np.array([np.asarray(net[i], np.float32)[(ATL >= 4) & (ATL <= 6)].mean() for i in range(len(y))])
    asym = (sa - sb) / (sa + sb)
    ab = y == 1
    q = np.quantile(np.abs(asym[ab]), 0.5)
    mean = lambda m: np.asarray(net[np.where(m)[0]], np.float32).mean(0)
    bil = mean(ab & (np.abs(asym) < q)); bil = 0.5 * (bil + bil[:, :, ::-1])
    a_low = mean(ab & (asym < -q))          # side a reduced
    b_low = mean(ab & (asym > q))           # side b reduced
    uni = 0.5 * (a_low + b_low[:, :, ::-1])  # pool both, oriented as "side a reduced"
    temps = {"normal": T0, "bilateral": bil, "a_reduced": uni, "b_reduced": uni[:, :, ::-1].copy()}
    print("templates built from", int(ab.sum()), "abnormal training scans", flush=True)

    best_ncc = np.full(len(y), -np.inf, np.float32)
    best = np.zeros(C.shape, np.float16)
    choice = np.zeros(len(y), int)
    for k, (name, tp) in enumerate(temps.items()):
        reg, ncc = T["register_all"](C, tp.astype(np.float32), dev)
        # score every candidate registration against the NORMAL template space via its own template
        upd = ncc > best_ncc
        best[upd] = reg[upd]
        best_ncc[upd] = ncc[upd]
        choice[upd] = k
        print(f"template {name}: median NCC {np.median(ncc):.3f}, chosen so far {np.bincount(choice, minlength=4)}", flush=True)
        del reg
    np.save(P + "cache/reg_multi.npy", best)
    NCCB = best_ncc
    print("choice by label (normal scans):", np.bincount(choice[y == 0], minlength=4), " (abnormal):", np.bincount(choice[y == 1], minlength=4))

    import multiprocessing as mp
    globals()["NCCB"] = NCCB
    rows = [feats((i, P + "cache/reg_multi.npy")) for i in range(len(y))]
    Fm = pd.DataFrame(rows)
    Fm.to_csv(P + "cache/features_multi.csv", index=False)

    F0 = pd.read_csv(P + "cache/features.csv")
    cols = [c for c in F0.columns if c not in train.NON_FEATURES]
    fam = tab["zoom_x"].round(1).astype(str)
    so = np.zeros(len(y), int)
    for f, (_, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y, y, fam)):
        so[va] = f
    rnd = pd.read_csv(P + "runs/final/oof.csv", dtype={"uid": str}).fold.values
    for tag, FF in (("single normal template", F0), ("multi-template", Fm)):
        X = FF[cols].fillna(0).to_numpy(float)
        msg = []
        for fname, folds in (("random", rnd), ("scanner-out", so)):
            zl, zg = np.zeros(len(y)), np.zeros(len(y))
            for f in range(5):
                tr, va = folds != f, folds == f
                zl[va] = train.apply_lr(train.fit_lr(X[tr], y[tr]), X[va])
                zg[va] = lgb.LGBMClassifier(**train.DEFAULT_CFG["gbm"], verbose=-1, random_state=0).fit(X[tr], y[tr]).booster_.predict(X[va], raw_score=True)
            msg.append(f"{fname}: lr {train.logloss(y, train.sigmoid(zl)):.4f}/{train.auc(y, zl):.3f} gbm {train.logloss(y, train.sigmoid(zg)):.4f}/{train.auc(y, zg):.3f}")
        print(f"{tag:>24} | " + " | ".join(msg), flush=True)
