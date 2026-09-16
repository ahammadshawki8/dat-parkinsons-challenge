"""A07 multi-reference ratios + A17 integrated (partial-volume-aware) uptake; marginal value over v6.
Paired comparison: baseline and augmented tabular models refit on identical folds. Whole-set scalars only."""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, lightgbm as lgb
from scipy import ndimage as ndi

P = ROOT
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / "solution"))
import datlib, train  # noqa: E402

R = np.load(P / "cache/reg_crops.npy", mmap_mode="r")
TPL = np.load(P / "cache/template.npy")
ATL = datlib.make_atlas(TPL)
STRI = (ATL >= 1) & (ATL <= 6)
SIDE_A = (ATL >= 1) & (ATL <= 3)
SIDE_B = (ATL >= 4) & (ATL <= 6)
tsm = ndi.gaussian_filter(TPL, 0.7)
BRAIN_X = (tsm > 0.7) & ~ndi.binary_dilation(STRI, iterations=4)          # extrastriatal brain
D, H, W = TPL.shape
yy = np.arange(H)[None, :, None]
OCC = BRAIN_X & (yy < H * 0.3)                                              # posterior (occipital-like)
DIL = {k: ndi.binary_dilation(STRI, iterations=k) if k else STRI for k in (0, 2, 4)}
XX = np.arange(W)[None, None, :]
DIL_A = {k: m & (XX < W // 2) for k, m in DIL.items()}
DIL_B = {k: m & (XX >= W // 2) for k, m in DIL.items()}
PUTP = {0: ATL == 3, 1: ATL == 6}
PUT = {0: (ATL == 2) | (ATL == 3), 1: (ATL == 5) | (ATL == 6)}
CAUD = {0: ATL == 1, 1: ATL == 4}


def feats(i):
    v = datlib._center(np.asarray(R[i], np.float32), datlib.NET)
    refs = {"bg": 1.0, "occ": float(v[OCC].mean()) + 1e-3, "p75": float(np.percentile(v[BRAIN_X], 75)) + 1e-3}
    out = {"r_occ_p75": refs["occ"] / refs["p75"], "r_occ_bg": refs["occ"], "r_p75_bg": refs["p75"]}
    for rn, rv in refs.items():
        pp = [float(v[PUTP[s]].mean()) / rv for s in (0, 1)]
        pu = [float(v[PUT[s]].mean()) / rv for s in (0, 1)]
        ca = [float(v[CAUD[s]].mean()) / rv for s in (0, 1)]
        out[f"m_{rn}_postput_min"] = min(pp)
        out[f"m_{rn}_put_mean"] = float(np.mean(pu))
        out[f"m_{rn}_caud_mean"] = float(np.mean(ca))
        out[f"m_{rn}_stri_mean"] = float(v[STRI].mean()) / rv
    vox_ml = datlib.SP ** 3 / 1000.0
    for k in DIL:
        ia = vox_ml * float(np.clip(v[DIL_A[k]] - 1.0, 0, None).sum())
        ib = vox_ml * float(np.clip(v[DIL_B[k]] - 1.0, 0, None).sum())
        out[f"i{k}_min"], out[f"i{k}_max"] = min(ia, ib), max(ia, ib)
        out[f"i{k}_asym"] = abs(ia - ib) / (ia + ib + 1e-3)
    return out


if __name__ == "__main__":
    cache = P / "cache/a07_a17_feats.csv"
    if not cache.exists():
        with ProcessPoolExecutor(4) as ex:
            pd.DataFrame(list(ex.map(feats, range(len(R)), chunksize=16))).to_csv(cache, index=False)
    NEW = pd.read_csv(cache)
    import ensemble_weights as A
    o = A.build(A.DEPLOY)
    y = o.y.to_numpy()
    folds = pd.read_csv(P / "runs/v3_s2026/oof.csv", dtype={"uid": str}).fold.to_numpy()
    tab = pd.read_csv(P / "cache/table.csv", dtype={"uid": str})
    Fb = pd.read_csv(P / "cache/features.csv")
    cols = [c for c in Fb.columns if c not in train.NON_FEATURES]
    zoom = tab.zoom_x.to_numpy(float)
    ref_cols = [c for c in NEW.columns if c.startswith(("m_", "r_"))]
    int_cols = [c for c in NEW.columns if c.startswith("i")]

    def tab_oof(X):
        zl, zg = np.zeros(len(y)), np.zeros(len(y))
        for f in range(5):
            tr, va = folds != f, folds == f
            zl[va] = train.apply_lr(train.fit_lr(X[tr], y[tr], zoom=zoom[tr]), X[va], zoom[va])
            zg[va] = lgb.LGBMClassifier(**train.DEFAULT_CFG["gbm"], verbose=-1, random_state=f).fit(X[tr], y[tr]).booster_.predict(X[va], raw_score=True)
        return zl, zg

    print(f"new features: {len(ref_cols)} multi-reference, {len(int_cols)} integrated-uptake")
    for tag, extra in (("baseline features", []), ("+ A07 multi-reference", ref_cols),
                       ("+ A17 integrated uptake", int_cols), ("+ A07 + A17", ref_cols + int_cols)):
        X = pd.concat([Fb[cols], NEW[extra]], axis=1).fillna(0).to_numpy(float)
        zl, zg = tab_oof(X)
        oo = o[["uid", "y", "group", "cnn2d"]].copy()
        oo["fold"], oo["lr"], oo["gbm"] = folds, zl, zg
        s = train.fit_stack(oo, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
        print(f"{tag:>26}: lr {train.logloss(y, train.sigmoid(zl)):.4f} gbm {train.logloss(y, train.sigmoid(zg)):.4f} | stack {s:.4f}")
