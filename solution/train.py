"""Cross-validated training of all models + calibrated stacker; writes submission assets."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import datlib  # noqa: E402

NON_FEATURES = {"uid", "ok", "err", "shape_x", "shape_y", "shape_z", "zoom_x", "zoom_y",
                "zoom_z", "loc_note", "top_to_peak_mm", "peak_off_x_mm", "label", "group", "fold"}

NOISE_W = None
AUX_T = None
AUX_COLS = []
BUDD_DIR = str(Path(__file__).resolve().parent.parent / "external" / "dat_spect_ud" / "dat_spect_ud" / "network_weights")

DEFAULT_CFG = dict(
    folds=5,
    split_seed=2026,
    nets=[
        dict(name="effb0", kind="2d", backbone="efficientnet_b0.ra_in1k", size=192,
             epochs=25, lr=4e-4, wd=1e-2, bs=32, seeds=[0, 1, 2]),
        dict(name="rn26d", kind="2d", backbone="resnet26d.bt_in1k", size=192,
             epochs=25, lr=3e-4, wd=1e-2, bs=32, seeds=[0, 1, 2]),
    ],
    # CNN logits are averaged inside a group before stacking (more stable than separate weights)
    stack_groups={"cnn2d": ["effb0", "rn26d"]},
    gbm=dict(num_leaves=7, learning_rate=0.03, n_estimators=350, min_child_samples=15,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0),
    ema=0.995,
)


def logloss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def make_folds(tab: pd.DataFrame, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold
    g = tab["group"].astype(str)
    counts = g.value_counts()
    g = g.where(g.map(counts) >= 2 * k, "rare")
    key = g + "_" + pd.Series(y.astype(int), index=tab.index).astype(str)
    kc = key.value_counts()
    key = key.where(key.map(kc) >= k, "rare_" + pd.Series(y.astype(int), index=tab.index).astype(str))
    folds = np.zeros(len(y), int)
    for f, (_, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(key, key)):
        folds[va] = f
    return folds


# ---------------------------------------------------------------------------
def train_net(T, store, y, tr, va, ncfg, seed, ema_decay, device, sw=None):
    import torch
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

    torch.manual_seed(seed * 1000 + int(va[0]))
    np.random.seed(seed)
    model = T["make_net"](ncfg, pretrained=True, seed=seed, budd_dir=BUDD_DIR)
    model = model.to(device).to(memory_format=torch.channels_last_3d if ncfg["kind"] == "3d" else torch.contiguous_format)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay), use_buffers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=ncfg["lr"], weight_decay=ncfg["wd"])
    bs, E = ncfg["bs"], ncfg["epochs"]
    steps_per = max(1, len(tr) // bs)
    total = E * steps_per
    warm = steps_per
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda")
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    swt = None if sw is None else torch.tensor(sw, dtype=torch.float32, device=device)
    aux_t = None
    if ncfg.get("aux_cols"):
        A = AUX_T[:, [AUX_COLS.index(c) for c in ncfg["aux_cols"]]].astype(np.float32)
        mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6          # training-fold statistics only
        aux_t = torch.tensor((A - mu) / sd, dtype=torch.float32, device=device)
    lossf = torch.nn.BCEWithLogitsLoss()
    fus_t = None
    if ncfg.get("fusion_cols"):
        Fm = AUX_T[:, [AUX_COLS.index(c) for c in ncfg["fusion_cols"]]].astype(np.float32)
        model.feat_mu.copy_(torch.tensor(Fm[tr].mean(0)))
        model.feat_sd.copy_(torch.tensor(Fm[tr].std(0) + 1e-6))
        fus_t = torch.tensor(Fm, dtype=torch.float32, device=device)
    for ep in range(E):
        model.train()
        if ncfg.get("freeze_bn"):                            # A12: keep pretrained running statistics
            for mod in model.modules():
                if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        perm = np.random.permutation(tr)
        for b in range(steps_per):
            idx = torch.as_tensor(perm[b * bs:(b + 1) * bs], device=device)
            with torch.no_grad():
                x = T["prep_input"](T["augment"](store[idx].float(), **ncfg.get("aug", {})))
            tgt = yt[idx]
            ls = ncfg.get("label_smooth", 0.0)
            if ls:
                tgt = tgt * (1 - ls) + 0.5 * ls
            mix = ncfg.get("mixup", 0.0)
            if mix:
                lam = float(np.random.beta(mix, mix))
                j = torch.randperm(len(idx), device=device)
                x = lam * x + (1 - lam) * x[j]
                tgt = lam * tgt + (1 - lam) * tgt[j]
            aux_loss = 0.0
            with torch.autocast("cuda", dtype=torch.float16):
                if fus_t is not None:
                    out = model(x, feats=fus_t[idx])
                elif aux_t is not None:
                    out, aux_out = model(x, return_aux=True)
                    aux_loss = ncfg.get("aux_w", 0.2) * torch.nn.functional.smooth_l1_loss(aux_out.float(), aux_t[idx])
                else:
                    out = model(x)
            if ncfg.get("sam"):
                # A13: sharpness-aware minimisation (bf16, no scaler; same batch for both passes)
                def closure():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o_ = model(x)
                    return lossf(o_.float(), tgt)
                l1 = closure()
                opt.zero_grad(set_to_none=True)
                l1.backward()
                params = [p_ for p_ in model.parameters() if p_.grad is not None]
                gn = torch.norm(torch.stack([p_.grad.norm() for p_ in params])) + 1e-12
                eps_w = []
                with torch.no_grad():
                    for p_ in params:
                        e = p_.grad * (ncfg["sam"] / gn)
                        p_.add_(e)
                        eps_w.append(e)
                opt.zero_grad(set_to_none=True)
                closure().backward()
                with torch.no_grad():
                    for p_, e in zip(params, eps_w):
                        p_.sub_(e)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                sched.step()
                ema.update_parameters(model)
                continue
            if swt is None:
                loss = lossf(out.float(), tgt)
            else:
                w = swt[idx]
                per = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), tgt, reduction="none")
                loss = (per * w).sum() / w.sum().clamp_min(1e-6)
            loss = loss + aux_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update_parameters(model)
    net = ema.module
    return net, predict_net(T, net, store, va, device, feats=fus_t, wide=ncfg.get("aug", {}).get("wide", False))


def predict_net(T, net, store, idx, device, bs=64, feats=None, wide=False):
    import torch
    net.eval()
    outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(idx), bs):
            ii = torch.as_tensor(idx[b:b + bs], device=device)
            v = store[ii].float()
            kw = {} if feats is None else {"feats": feats[ii]}
            z = 0.5 * (net(T["prep_input"](T["center"](v, wide=wide)), **kw).float() +
                       net(T["prep_input"](T["center"](v, flip=True, wide=wide)), **kw).float())
            outs.append(z.cpu().numpy())
    return np.concatenate(outs)


# ---------------------------------------------------------------------------
def fit_lr(Xtr, ytr, C=0.05, zoom=None):
    from sklearn.linear_model import LogisticRegression
    adj = None
    if zoom is not None:
        m = ytr == 0
        adj, *_ = np.linalg.lstsq(datlib._zoom_basis(zoom[m]), Xtr[m], rcond=None)
        Xtr = Xtr - datlib._zoom_basis(zoom) @ adj
        adj = adj.tolist()
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Z = np.clip((Xtr - mu) / sd, -6, 6)
    m = LogisticRegression(C=C, max_iter=5000).fit(Z, ytr)
    return dict(mu=mu.tolist(), sd=sd.tolist(), coef=m.coef_[0].tolist(), b=float(m.intercept_[0]), adj=adj)


def apply_lr(p, X, zoom=None):
    return datlib.lr_apply(p, X, zoom)


def main(cache_dir, labels_csv, out_dir, cfg=None, only=None):
    import torch
    torch.backends.cudnn.benchmark = True
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    out = Path(out_dir)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    device = "cuda"

    global NOISE_W
    NOISE_W = np.load(cfg["noise_weights"]) if cfg.get("noise_weights") else None
    tab = pd.read_csv(Path(cache_dir) / "table.csv", dtype={"uid": str})
    crops = np.load(Path(cache_dir) / "crops.npy", mmap_mode="r")
    lab = pd.read_csv(labels_csv, dtype={"uid": str}).set_index("uid")
    y = lab.loc[tab["uid"], "is_pathologic"].to_numpy().astype(np.float32)
    tab["group"] = [datlib.group_key(r) if r["ok"] == 1 else "fail" for _, r in tab.iterrows()]
    if cfg.get("fold_mode") == "scanner":  # hold out whole scanner voxel-size families
        from sklearn.model_selection import StratifiedGroupKFold
        fam = tab["zoom_x"].round(1).astype(str).fillna("na")
        folds = np.zeros(len(y), int)
        sgk = StratifiedGroupKFold(cfg["folds"], shuffle=True, random_state=cfg["split_seed"])
        for f, (_, va) in enumerate(sgk.split(y, y, fam)):
            folds[va] = f
    else:
        folds = make_folds(tab, y, cfg["folds"], cfg["split_seed"])
    K = cfg["folds"]

    # ---- template registration + features ------------------------------
    T = datlib.build_torch()
    reg_path = Path(cache_dir) / "reg_crops.npy"
    if reg_path.exists() and (Path(cache_dir) / "template.npy").exists():
        template = np.load(Path(cache_dir) / "template.npy")
        crops = np.load(reg_path)
        feat_tab = pd.read_csv(Path(cache_dir) / "features.csv")
        print("loaded cached registration")
    else:
        t0 = time.time()
        good = (tab["ok"] == 1).to_numpy()
        template = T["build_template"](crops[(y == 0) & good], device, n_iter=cfg.get("tpl_iter", 3))
        np.save(Path(cache_dir) / "template.npy", template)
        atlas = datlib.make_atlas(template)
        metas = tab.to_dict("records")
        crops, rows = datlib.postprocess(crops, metas, template, atlas, T, device)
        feat_tab = pd.DataFrame(rows)
        np.save(reg_path, crops)
        feat_tab.to_csv(Path(cache_dir) / "features.csv", index=False)
        print(f"template + registration + features: {time.time() - t0:.0f}s")
    atlas = datlib.make_atlas(template)
    np.save(out / "assets" / "template.npy", template.astype(np.float32))
    feat_tab.index = tab.index
    tab = pd.concat([tab.drop(columns=[c for c in feat_tab.columns if c in tab.columns]), feat_tab], axis=1)
    feat_cols = [c for c in feat_tab.columns if c not in NON_FEATURES]
    X = tab[feat_cols].fillna(0).to_numpy(np.float64)
    global AUX_T, AUX_COLS
    AUX_COLS = feat_cols
    AUX_T = X
    print(f"n={len(y)} pos_rate={y.mean():.3f} groups={tab['group'].nunique()} features={len(feat_cols)}")

    oof = pd.DataFrame({"uid": tab["uid"], "y": y, "fold": folds, "group": tab["group"]})
    assets = dict(feat_cols=feat_cols, folds=K, nets=[], gbm=[], lr=[])

    # ---- tabular models --------------------------------------------------
    import lightgbm as lgb
    oof["gbm"] = 0.0
    oof["lr"] = 0.0
    for f in range(K):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        m = lgb.LGBMClassifier(**cfg["gbm"], random_state=f, verbose=-1).fit(X[tr], y[tr])
        oof.loc[va, "gbm"] = m.booster_.predict(X[va], raw_score=True)
        fn = f"gbm_f{f}.txt"
        m.booster_.save_model(str(out / "assets" / fn))
        assets["gbm"].append(fn)
        zz = tab["zoom_x"].fillna(2.4).to_numpy(float) if cfg.get("lr_adjust") else None
        p = fit_lr(X[tr], y[tr], zoom=None if zz is None else zz[tr])
        oof.loc[va, "lr"] = apply_lr(p, X[va], None if zz is None else zz[va])
        assets["lr"].append(p)
    for c in ("gbm", "lr"):
        print(f"[{c}] OOF logloss={logloss(y, sigmoid(oof[c].to_numpy())):.4f} auc={auc(y, oof[c]):.4f}")

    # ---- neural nets -----------------------------------------------------
    store = torch.from_numpy(np.ascontiguousarray(crops)).unsqueeze(1).to(device)
    for ncfg in cfg["nets"]:
        if only and ncfg["name"] not in only:
            continue
        name = ncfg["name"]
        col = np.zeros((len(ncfg["seeds"]), len(y)))
        t0 = time.time()
        for si, seed in enumerate(ncfg["seeds"]):
            for f in range(K):
                tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
                sw = None if NOISE_W is None else NOISE_W[f]
                net, z = train_net(T, store, y, tr, va, ncfg, seed, cfg["ema"], device, sw=sw)
                col[si, va] = z
                fn = f"{name}_s{seed}_f{f}.pt"
                sd = {k: v.detach().half().cpu() if v.is_floating_point() else v.cpu()
                      for k, v in net.state_dict().items()}
                torch.save(sd, out / "assets" / fn)
                del net
                torch.cuda.empty_cache()
            zs = col[si]
            print(f"[{name}] seed {seed}: OOF logloss={logloss(y, sigmoid(zs)):.4f} "
                  f"auc={auc(y, zs):.4f}  ({time.time() - t0:.0f}s)", flush=True)
        oof[name] = col.mean(0)
        print(f"[{name}] seed-avg OOF logloss={logloss(y, sigmoid(oof[name].to_numpy())):.4f} "
              f"auc={auc(y, oof[name]):.4f}", flush=True)
        assets["nets"].append(dict(ncfg, files=[f"{name}_s{s}_f{f}.pt" for s in ncfg["seeds"] for f in range(K)]))

    groups = {g: [m for m in members if m in oof.columns]
              for g, members in cfg.get("stack_groups", {}).items()}
    groups = {g: m for g, m in groups.items() if m}
    grouped = {m for ms in groups.values() for m in ms}
    for n in assets["nets"]:
        if n["name"] not in grouped:
            groups[n["name"]] = [n["name"]]
    for g, members in groups.items():
        oof[g] = oof[members].mean(axis=1)
        print(f"[{g}] OOF logloss={logloss(y, sigmoid(oof[g].to_numpy())):.4f} auc={auc(y, oof[g]):.4f}")
    assets["groups"] = groups
    oof.to_csv(out / "oof.csv", index=False)
    stack = fit_stack(oof, list(groups) + ["gbm", "lr"])
    assets["stack"] = stack
    (out / "assets" / "config.json").write_text(json.dumps(assets, indent=1))
    return oof, assets


def fit_stack(oof: pd.DataFrame, cols, verbose=True):
    """Logistic stacker on OOF logits, evaluated with nested CV; picks clip eps."""
    from sklearn.linear_model import LogisticRegression
    y = oof["y"].to_numpy()
    Z = oof[cols].to_numpy()
    folds = oof["fold"].to_numpy()
    best = None
    for C in (0.03, 0.1, 0.3, 1.0):
        nested = np.zeros(len(y))
        for f in np.unique(folds):
            tr, va = folds != f, folds == f
            m = LogisticRegression(C=C, max_iter=5000).fit(Z[tr], y[tr])
            nested[va] = m.decision_function(Z[va])
        ll = logloss(y, sigmoid(nested))
        if verbose:
            print(f"[stack] C={C} nested OOF logloss={ll:.4f} auc={auc(y, nested):.4f}")
        if best is None or ll < best[0] - 1e-5:
            best = (ll, C, nested)
    ll, C, nested = best
    eps_best, ll_best = 0.0, ll
    for eps in (0.002, 0.005, 0.01, 0.015, 0.02, 0.03):
        l2 = logloss(y, np.clip(sigmoid(nested), eps, 1 - eps))
        if l2 < ll_best - 1e-5:
            eps_best, ll_best = eps, l2
    m = LogisticRegression(C=C, max_iter=5000).fit(Z, y)
    if verbose:
        print(f"[stack] chosen C={C} eps={eps_best} nested OOF logloss={ll_best:.4f}; "
              f"weights={dict(zip(cols, np.round(m.coef_[0], 3)))} b={m.intercept_[0]:.3f}")
        g = oof.assign(p=np.clip(sigmoid(nested), eps_best, 1 - eps_best))
        rows = []
        for k, d in g.groupby("group"):
            rows.append((k, len(d), round(logloss(d["y"].to_numpy(), d["p"].to_numpy()), 4)))
        rows.sort(key=lambda r: -r[1])
        print("[stack] per scanner-group (n, logloss):", rows[:12])
    return dict(cols=cols, coef=m.coef_[0].tolist(), b=float(m.intercept_[0]), eps=eps_best,
                C=C, nested_logloss=ll_best, prior=float(y.mean()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], only=sys.argv[4].split(",") if len(sys.argv) > 4 else None)
