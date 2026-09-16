"""A06: frozen DINOv2 ViT-S/14 features + fold-internal PCA/logistic head; marginal value over v6.

Weights: timm 'vit_small_patch14_dinov2.lvd142m' (Apache-2.0, standard DINOv2 model card).
Features stay on disk locally; only whole-set scalars are printed.
"""
from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

P = ROOT
sys.path.insert(0, str(P))
sys.path.insert(0, str(P / "solution"))
import datlib, train  # noqa: E402

FEAT = P / "cache/dino_feats.npy"
dev = "cuda"
if not FEAT.exists():
    import timm
    m = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(dev).eval()
    T = datlib.build_torch()
    R = np.load(P / "cache/reg_crops.npy", mmap_mode="r")
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for b in range(0, len(R), 32):
            v = T["center"](torch.from_numpy(np.asarray(R[b:b + 32])).unsqueeze(1).to(dev).float())
            D = v.shape[2]
            s0 = D // 2 - 6
            img = torch.stack([v[:, 0, s0 + 4 * i:s0 + 4 * i + 4].mean(1) for i in range(3)], 1)   # 3 axial slabs
            img = (img / 6.0).clamp(0, 1)                                                             # bg units -> [0,1]
            img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            feats = []
            for flip in (False, True):
                x = torch.flip(img, dims=[3]) if flip else img
                tok = m.forward_features((x - mean) / std)            # (B, 1+reg+N, C)
                cls = tok[:, 0]
                patches = tok[:, m.num_prefix_tokens:]
                g = int(patches.shape[1] ** 0.5)
                pm = patches.reshape(-1, g, g, patches.shape[-1])
                c0, c1 = g // 4, 3 * g // 4                            # central (striatal) region pooling
                roi = pm[:, c0:c1, c0:c1].mean((1, 2))
                feats.append(torch.cat([cls, patches.mean(1), roi], 1))
            out.append((0.5 * (feats[0] + feats[1])).cpu().numpy())
    np.save(FEAT, np.concatenate(out).astype(np.float32))
    print("extracted DINOv2 features", np.load(FEAT).shape)

X = np.load(FEAT)
import ensemble_weights as A  # noqa: E402
o = A.build(A.DEPLOY)
y = o.y.to_numpy()
folds = o.fold.to_numpy()
z = np.zeros(len(y))
for f in np.unique(folds):
    tr, va = folds != f, folds == f
    sc = StandardScaler().fit(X[tr])
    pca = PCA(64, random_state=0).fit(sc.transform(X[tr]))
    Zt, Zv = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[va]))
    z[va] = LogisticRegression(C=0.05, max_iter=5000).fit(Zt, y[tr]).decision_function(Zv)
print(f"DINOv2 frozen expert alone: logloss {train.logloss(y, train.sigmoid(z)):.4f} auc {train.auc(y, z):.4f}")
base = train.fit_stack(o, ["cnn2d", "gbm", "lr"], verbose=False)["nested_logloss"]
o["dino"] = z
with_d = train.fit_stack(o, ["cnn2d", "gbm", "lr", "dino"], verbose=False)["nested_logloss"]
print(f"v6 stack {base:.4f} -> with DINOv2 column {with_d:.4f} (change {with_d - base:+.4f})")
