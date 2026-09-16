from pathlib import Path as _Path
ROOT = _Path(__file__).resolve().parents[1]
ROOT_S = str(ROOT).replace("\\", "/") + "/"
import sys as _sys
_sys.path.insert(0, str(ROOT / "solution"))
_sys.path.insert(0, str(ROOT / "scripts"))
import sys, itertools
sys.path.insert(0, str(ROOT / "solution"))
sys.path.insert(0, ROOT_S + "external/dat_spect_ud")
import numpy as np, pandas as pd, torch
import datlib, train
from dat_spect_ud.network_architecture import customResNet, customResNetRegional
from sklearn.linear_model import LogisticRegression

W = ROOT_S + "external/dat_spect_ud/dat_spect_ud/network_weights/"
dev = 'cuda'
R = np.load(ROOT_S + "cache/reg_crops.npy", mmap_mode='r')
tab = pd.read_csv(ROOT_S + "cache/table.csv", dtype={'uid': str})
lab = pd.read_csv(ROOT_S + "data/train_labels.csv", dtype={'uid': str}).set_index('uid')
y = lab.loc[tab.uid, 'is_pathologic'].values
tpl = np.load(ROOT_S + "cache/template.npy")
atlas = datlib.make_atlas(tpl)
d = [(a - b) // 2 for a, b in zip(datlib.STORE, datlib.NET)]
ref_mask = np.zeros(datlib.STORE, bool)
ref_mask[d[0]:d[0] + datlib.NET[0], d[1]:d[1] + datlib.NET[1], d[2]:d[2] + datlib.NET[2]] = atlas == 9

vol = torch.from_numpy(np.ascontiguousarray(R)).to(dev).float()  # (N,56,80,80)
# per-scan 75th percentile inside reference region
refv = vol[:, torch.from_numpy(ref_mask).to(dev)]
p75 = torch.quantile(refv, 0.75, dim=1)
print('ref p75 median', p75.median().item())

def nets(mode):
    out = []
    for i in range(5):
        n = customResNet(mode != 'robust')
        n.load_state_dict(torch.load(W + f'network_weights_{mode}_{i}', map_location='cpu', weights_only=True))
        out.append(n.to(dev).eval())
    return out
ENS = {m: nets(m) for m in ('robust', 'acc')}

def slab(dz, dy, flip, k):
    c = datlib.STORE[0] // 2 + dz
    s = vol[:, c - 3:c + 3].mean(1)              # (N, H=y, W=x)
    s = s / (k * p75.view(-1, 1, 1))
    s = torch.nn.functional.pad(s, (0, 0, 8, 8))[:, 12 + dy:84 + dy, 4:76]  # 72x72
    if flip:
        s = torch.flip(s, dims=[2])
    s = s.transpose(1, 2)                         # rows = x, cols = y
    return ((s.clamp(0, 6.5) - 0.6048597782240399) / 0.6051688035793145).unsqueeze(1)

@torch.no_grad()
def score(x, mode):
    zs = []
    for n in ENS[mode]:
        o = []
        for b in range(0, len(x), 256):
            xb = x[b:b + 256]
            if mode == 'robust':
                o.append(n(xb))
            else:
                sm = n(xb.unsqueeze(2)).softmax(1)
                o.append(torch.log(sm[:, 1] + 1e-6) - torch.log(sm[:, 0] + 1e-6))
        zs.append(torch.cat(o))
    return torch.stack(zs).mean(0).cpu().numpy()

def ll1d(z):
    m = LogisticRegression(C=100, max_iter=1000).fit(z[:, None], y)
    return train.logloss(y, m.predict_proba(z[:, None])[:, 1])

res = []
for dz, dy, flip, k in itertools.product((-2, 0, 2, 4), (-4, 0, 4), (True,), (0.8, 1.0, 1.25)):
    z = score(slab(dz, dy, flip, k), 'robust')
    res.append((dz, dy, flip, k, train.auc(y, z), ll1d(z)))
res.sort(key=lambda r: r[5])
for r in res[:8]:
    print('dz=%d dy=%d flip=%s k=%.2f  auc=%.4f  calibrated-logloss=%.4f' % r)
best = res[0]
for flip in (True, False):
    z = score(slab(best[0], best[1], flip, best[3]), 'robust')
    print('flip', flip, 'robust auc %.4f ll %.4f' % (train.auc(y, z), ll1d(z)))
z = score(slab(best[0], best[1], True, best[3]), 'acc')
print('acc ensemble auc %.4f ll %.4f' % (train.auc(y, z), ll1d(z)))
