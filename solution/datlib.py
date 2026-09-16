"""Shared DaT-SPECT library: preprocessing, handcrafted features, models, augmentation.

The exact same file is used for training (Kaggle notebook) and for inference
(inside submission.zip), so preprocessing can never diverge between the two.
Each scan is processed independently (no cross-case statistics).
"""
from __future__ import annotations

import math

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------
# Geometry (arrays are stored as (z=inferior->superior, y=posterior->anterior,
# x=left->right in canonical RAS index order); spacing is isotropic SP mm.
# ----------------------------------------------------------------------------
SP = 2.0                 # working spacing (mm)
COARSE = 4.0             # coarse grid for localisation (mm)
STORE = (56, 80, 80)     # stored crop (D,H,W) = 112 x 160 x 160 mm
NET = (48, 64, 64)       # network crop (D,H,W) =  96 x 128 x 128 mm
FEAT_VERSION = 3


def _otsu(v: np.ndarray, nbins: int = 256) -> float:
    v = v.ravel()
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    hist = hist.astype(np.float64)
    c = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * c)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = (m0[-1] - m0) / w1
        vb = np.nan_to_num(w0 * w1 * (mu0 - mu1) ** 2)
    return float(c[int(np.argmax(vb))])


def _largest_cc(m: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(m)
    if n <= 1:
        return m
    s = np.bincount(lab.ravel())
    s[0] = 0
    return lab == int(np.argmax(s))


def load_canonical(path):
    img = nib.load(str(path))
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    data = np.asanyarray(img.dataobj)
    data = np.asarray(data, dtype=np.float32)
    while data.ndim > 3:
        data = data.mean(axis=-1) if data.shape[-1] > 1 else data[..., 0]
    if data.ndim < 3:
        data = data.reshape(data.shape + (1,) * (3 - data.ndim))
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    if zooms.size < 3:
        zooms = np.concatenate([zooms, np.ones(3 - zooms.size)])
    zooms = np.where(np.isfinite(zooms) & (zooms > 0.3) & (zooms < 20), zooms, 2.0)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.maximum(data, 0.0)
    return data, zooms


def preprocess(path, mode=0):
    """Return (crop float16 (D,H,W) in background units, meta dict).

    mode 0: striatal peak searched in a band 25-125 mm below the vertex (default).
    mode 1: fallback for scans whose geometry does not match that assumption - the peak is
            searched anywhere inside the head mask. Used only when mode 0 registers very poorly.
    """
    data, zooms = load_canonical(path)
    shp = np.array(data.shape, dtype=np.float64)

    # ---- coarse grid (anti-aliased) -------------------------------------
    sig = np.maximum(0.0, 0.5 * COARSE / zooms - 0.35)
    src = ndi.gaussian_filter(data, sig) if np.any(sig > 0.05) else data
    f = zooms / COARSE
    small = ndi.zoom(src, f, order=1, prefilter=False, mode="nearest").astype(np.float32)
    scale = (shp - 1) / np.maximum(np.array(small.shape) - 1, 1)  # small idx -> orig idx

    sm = ndi.gaussian_filter(small, 1.0)
    hi = float(np.percentile(sm, 99.5))
    if not np.isfinite(hi) or hi <= 0:
        raise ValueError("empty volume")
    cl = np.minimum(sm, hi)
    thr = max(_otsu(cl), 0.08 * hi)
    m = cl > thr
    m = ndi.binary_opening(m, iterations=1)
    m = _largest_cc(m)
    m = ndi.binary_fill_holes(ndi.binary_closing(m, iterations=2))
    if m.sum() < 200:
        m = sm > np.percentile(sm, 60)
    note = 0

    zs = np.where(m.any(axis=(0, 1)))[0]
    ztop = int(zs.max())
    zlo = max(0, int(round(ztop - 125.0 / COARSE)))
    zhi = max(zlo + 1, int(round(ztop - 25.0 / COARSE)))
    slab = np.zeros_like(m)
    slab[:, :, zlo:zhi + 1] = m[:, :, zlo:zhi + 1]
    if slab.sum() < 50:
        slab = m
        note = 1
    cx, cy, _ = ndi.center_of_mass(slab)

    sm2 = ndi.gaussian_filter(small, 1.5)
    ii, jj, kk = np.meshgrid(np.arange(small.shape[0]), np.arange(small.shape[1]),
                             np.arange(small.shape[2]), indexing="ij")
    if mode == 1:
        allow = ndi.binary_erosion(m, iterations=2)
        note = 3
        if allow.sum() < 10:
            allow = m
    else:
        allow = slab & (np.abs(ii - cx) * COARSE <= 35.0) & (np.abs(jj - cy) * COARSE <= 50.0)
        if allow.sum() < 10:
            allow = m
            note = 2
    score = np.where(allow, sm2, -np.inf)
    px, py, pz = np.unravel_index(int(np.argmax(score)), score.shape)

    # midline from head mask centroid around the striatal level
    zl2, zh2 = max(0, int(pz - 5)), int(pz + 6)
    band = np.zeros_like(m)
    band[:, :, zl2:zh2] = m[:, :, zl2:zh2]
    if band.sum() > 50:
        cx = ndi.center_of_mass(band)[0]

    # background: trimmed mean inside head, +-50 mm around striatal level
    zb1, zb2 = max(0, int(pz - 12)), int(pz + 13)
    bmask = np.zeros_like(m)
    bmask[:, :, zb1:zb2] = m[:, :, zb1:zb2]
    vals = sm[bmask] if bmask.sum() > 200 else sm[m]
    lo, hi2 = np.percentile(vals, [25.0, 85.0])
    sel = vals[(vals >= lo) & (vals <= hi2)]
    bg = float(sel.mean()) if sel.size else float(np.median(vals))
    if not np.isfinite(bg) or bg <= 1e-6:
        bg = float(np.mean(vals)) + 1e-6

    # ---- 2 mm crop straight from original voxels ------------------------
    center_orig = np.array([cx, py, pz], dtype=np.float64) * scale  # (x,y,z) orig idx
    if np.any(zooms < 0.8 * SP):
        s2 = np.maximum(0.0, 0.5 * SP / zooms - 0.35)
        data = ndi.gaussian_filter(data, s2)
    out_xyz = (STORE[2], STORE[1], STORE[0])
    mat = SP / zooms
    offset = center_orig - mat * (np.array(out_xyz, dtype=np.float64) - 1) / 2.0
    crop = ndi.affine_transform(data, mat, offset=offset, output_shape=out_xyz,
                                order=1, mode="constant", cval=0.0)
    crop = np.clip(crop / bg, 0.0, 40.0).transpose(2, 1, 0)  # -> (z,y,x)
    crop = np.ascontiguousarray(crop).astype(np.float16)

    meta = dict(
        zoom_x=float(zooms[0]), zoom_y=float(zooms[1]), zoom_z=float(zooms[2]),
        shape_x=int(shp[0]), shape_y=int(shp[1]), shape_z=int(shp[2]),
        peak_bg=float(sm2[px, py, pz] / (bg + 1e-6)),
        top_to_peak_mm=float((ztop - pz) * COARSE),
        peak_off_x_mm=float((px - cx) * COARSE),
        loc_note=note,
    )
    return crop, meta


def group_key(meta: dict) -> str:
    return f"{meta['shape_x']}x{meta['shape_y']}x{meta['shape_z']}_{meta['zoom_x']:.2f}"


# ----------------------------------------------------------------------------
# Handcrafted semi-quantitative features (left/right invariant)
# ----------------------------------------------------------------------------
def _center(vol, shape):
    d0 = [(a - b) // 2 for a, b in zip(vol.shape, shape)]
    return vol[d0[0]:d0[0] + shape[0], d0[1]:d0[1] + shape[1], d0[2]:d0[2] + shape[2]]


def features(crop: np.ndarray, meta: dict) -> dict:
    v = _center(crop.astype(np.float32), NET)
    v = ndi.gaussian_filter(v, 1.0)
    D, H, W = v.shape
    out = {}
    flat = v.ravel()
    gmax = float(flat.max()) + 1e-6
    for q in (50, 90, 99, 99.9):
        out[f"g_p{q}"] = float(np.percentile(flat, q))
    out["g_max"] = gmax
    # restrict to a central striatal box (80x80x48 mm)
    sb = v[D // 2 - 12:D // 2 + 12, H // 2 - 20:H // 2 + 20, W // 2 - 20:W // 2 + 20]
    sides = {"R": sb[:, :, :20], "L": sb[:, :, 20:][:, :, ::-1]}  # both lateral->medial
    per = {}
    zz, yy, xx = np.meshgrid(np.arange(sb.shape[0]), np.arange(sb.shape[1]),
                             np.arange(20), indexing="ij")
    for s, h in sides.items():
        hv = h.ravel()
        srt = np.sort(hv)[::-1]
        d = {}
        d["max"] = float(srt[0])
        for k in (30, 125, 400, 1000):
            d[f"top{k}"] = float(srt[:k].mean())
        for t in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
            d[f"vol{t}"] = float((hv > t).sum()) * SP ** 3 / 1000.0
        for frac in (0.5, 0.7):
            hot = h > frac * gmax
            n = int(hot.sum())
            d[f"n{frac}"] = n * SP ** 3 / 1000.0
            if n >= 3:
                y = yy[hot].astype(np.float64)
                z = zz[hot].astype(np.float64)
                x = xx[hot].astype(np.float64)
                w = h[hot].astype(np.float64)
                d[f"yext{frac}"] = float(np.percentile(y, 95) - np.percentile(y, 5)) * SP
                d[f"zext{frac}"] = float(np.percentile(z, 95) - np.percentile(z, 5)) * SP
                d[f"xext{frac}"] = float(np.percentile(x, 95) - np.percentile(x, 5)) * SP
                d[f"ycen{frac}"] = float(np.average(y, weights=w) - sb.shape[1] / 2) * SP
                d[f"xcen{frac}"] = float(np.average(x, weights=w)) * SP
                c = np.cov(np.stack([x, y, z]))
                ev = np.sort(np.linalg.eigvalsh(c + 1e-6 * np.eye(3)))[::-1]
                d[f"elong{frac}"] = float(math.sqrt(ev[0] / max(ev[1], 1e-6)))
            else:
                for nm in ("yext", "zext", "xext", "ycen", "xcen", "elong"):
                    d[f"{nm}{frac}"] = 0.0
        # antero-posterior gradient: uptake profile along y of the hot band
        hot = h > 0.4 * gmax
        prof = np.where(hot, h, 0).sum(axis=(0, 2))
        cnt = hot.sum(axis=(0, 2))
        if cnt.sum() > 5:
            ys = np.where(cnt > 0)[0]
            y0, y1 = ys.min(), ys.max()
            thirds = np.array_split(np.arange(y0, y1 + 1), 3)
            mean_t = [prof[t].sum() / max(cnt[t].sum(), 1) for t in thirds]
            # canonical y increases anteriorly: thirds[0] posterior (putamen)
            d["post_ant"] = float(mean_t[0] / (mean_t[2] + 1e-6))
            d["mid_ant"] = float(mean_t[1] / (mean_t[2] + 1e-6))
        else:
            d["post_ant"] = 0.0
            d["mid_ant"] = 0.0
        # mean uptake in fixed posterior / anterior sub-boxes around the peak
        zc, yc, xc = np.unravel_index(int(np.argmax(h)), h.shape)
        ant = h[max(0, zc - 3):zc + 4, min(yc + 2, H):min(yc + 9, sb.shape[1]), :]
        pos = h[max(0, zc - 3):zc + 4, max(0, yc - 12):max(1, yc - 3), :]
        d["box_post"] = float(np.sort(pos.ravel())[::-1][:60].mean()) if pos.size else 0.0
        d["box_ant"] = float(np.sort(ant.ravel())[::-1][:60].mean()) if ant.size else 0.0
        d["box_ratio"] = d["box_post"] / (d["box_ant"] + 1e-6)
        per[s] = d
    for k in per["R"]:
        a, b = per["R"][k], per["L"][k]
        out[f"{k}_min"] = min(a, b)
        out[f"{k}_max"] = max(a, b)
        out[f"{k}_mean"] = 0.5 * (a + b)
        out[f"{k}_asym"] = abs(a - b) / (abs(a) + abs(b) + 1e-6)
    out["zoom"] = float(np.mean([meta["zoom_x"], meta["zoom_y"], meta["zoom_z"]]))
    out["peak_bg"] = float(meta["peak_bg"])
    return out


# ----------------------------------------------------------------------------
# Torch parts (imported lazily so preprocessing workers stay light)
# ----------------------------------------------------------------------------
def build_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block3d(nn.Module):
        def __init__(self, cin, cout, stride):
            super().__init__()
            self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm3d(cout)
            self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm3d(cout)
            self.sc = None
            if stride != 1 or cin != cout:
                self.sc = nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False),
                                        nn.BatchNorm3d(cout))

        def forward(self, x):
            y = F.relu(self.b1(self.c1(x)), inplace=True)
            y = self.b2(self.c2(y))
            return F.relu(y + (x if self.sc is None else self.sc(x)), inplace=True)

    class Net3D(nn.Module):
        def __init__(self, widths=(24, 48, 96, 160), drop=0.3):
            super().__init__()
            w0 = widths[0]
            self.stem = nn.Sequential(nn.Conv3d(1, w0, 3, 1, 1, bias=False),
                                      nn.BatchNorm3d(w0), nn.ReLU(inplace=True))
            layers, cin = [], w0
            for i, w in enumerate(widths):
                layers.append(Block3d(cin, w, 2))
                if i >= 1:
                    layers.append(Block3d(w, w, 1))
                cin = w
            self.body = nn.Sequential(*layers)
            self.drop = nn.Dropout(drop)
            self.fc = nn.Linear(cin, 1)

        def forward(self, x):
            x = self.body(self.stem(x))
            x = torch.cat([x.mean(dim=(2, 3, 4))], 1)
            return self.fc(self.drop(x)).squeeze(1)

    class Net2D(nn.Module):
        def __init__(self, backbone="efficientnet_b0", pretrained=False, size=192, drop=0.3,
                     nslab=3, slab=4, views="axial", seq_half=8, aux_dim=0, fusion_dim=0):
            super().__init__()
            import timm
            self.size, self.nslab, self.slab, self.views = size, nslab, slab, views
            nch = 3 if views == "triplane" else (2 * nslab if views in ("dual", "dualraw", "bilat") else nslab)
            self.bb = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                        in_chans=nch, drop_rate=0.0)
            self.drop = nn.Dropout(drop)
            self.fc = nn.Linear(self.bb.num_features, 1)
            self.aux = nn.Linear(self.bb.num_features, aux_dim) if aux_dim else None
            self.fusion_dim = fusion_dim
            if fusion_dim:
                # A09: per-fold standardisation stored in the checkpoint (fitted on the training fold only)
                self.register_buffer("feat_mu", torch.zeros(fusion_dim))
                self.register_buffer("feat_sd", torch.ones(fusion_dim))
                self.fmlp = nn.Sequential(nn.Linear(fusion_dim, 64), nn.ReLU(inplace=True), nn.Dropout(drop),
                                          nn.Linear(64, 64), nn.ReLU(inplace=True))
                self.ffc = nn.Linear(self.bb.num_features + 64, 1)
            if views == "bilat":
                nf = self.bb.num_features
                self.bil = nn.Sequential(nn.Linear(2 * nf, 256), nn.ReLU(inplace=True), nn.Dropout(drop),
                                         nn.Linear(256, 1))
            if views == "seq":
                self.offs = list(range(-seq_half, seq_half + 1, 2))   # default 9 slices, 4 mm apart, +-16 mm
                k, hd = len(self.offs), 128
                self.proj = nn.Linear(self.bb.num_features, 2 * hd)
                self.pos = nn.Parameter(torch.zeros(1, k, 2 * hd))
                self.gru = nn.GRU(2 * hd, hd, batch_first=True, bidirectional=True)
                self.att_v = nn.Linear(2 * hd, 64)
                self.att_u = nn.Linear(2 * hd, 64)
                self.att_w = nn.Linear(64, 1)
                self.fc_seq = nn.Linear(2 * hd, 1)

        def _seq(self, x):
            B, D = x.shape[0], x.shape[2]
            c = D // 2
            raw = (x[:, 0] * 2.0 + 1.0).clamp_min(0.0)
            q = torch.quantile(raw[:, c - 10:c + 10].flatten(1).float(), 0.99, dim=1).clamp_min(1e-3).view(-1, 1, 1)
            # (quantile window fixed at +-20 mm so the default model is unchanged)
            imgs = []
            for o in self.offs:
                s = x[:, 0, c + o - 1:c + o + 1].mean(1)                     # 4 mm slice, bg units (centred)
                r = raw[:, c + o - 1:c + o + 1].mean(1) / q * 2.0 - 1.0     # self-normalised
                ctx = x[:, 0, c + o - 3:c + o + 3].mean(1)                   # 12 mm local context
                imgs.append(torch.stack([s, r, ctx], 1))
            seq = torch.stack(imgs, 1)                                        # B,K,3,H,W
            K = seq.shape[1]
            flat = F.interpolate(seq.flatten(0, 1), size=(self.size, self.size), mode="bilinear", align_corners=False)
            e = self.proj(self.bb(flat)).view(B, K, -1) + self.pos
            h, _ = self.gru(e)
            a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h)))   # gated attention
            pooled = (torch.softmax(a, 1) * h).sum(1)
            return self.fc_seq(self.drop(pooled)).squeeze(1)

        def forward(self, x, return_aux=False, feats=None):  # x: (B,1,D,H,W) network crop
            if self.views == "seq":
                return self._seq(x)
            D, H, W = x.shape[2], x.shape[3], x.shape[4]
            if self.views == "triplane":
                ax = x[:, 0, D // 2 - 6:D // 2 + 6].mean(1)              # (B,H,W) axial slab
                co = x[:, 0, :, H // 2 - 6:H // 2 + 6].mean(2)           # (B,D,W) coronal slab
                sa = x[:, 0, :, :, W // 2 - 6:W // 2 + 6].mean(3)        # (B,D,H) sagittal slab
                sz = (self.size, self.size)
                img = torch.cat([F.interpolate(v.unsqueeze(1), size=sz, mode="bilinear", align_corners=False)
                                 for v in (ax, co, sa)], 1)
                return self.fc(self.drop(self.bb(img))).squeeze(1)
            s0 = D // 2 - self.nslab * self.slab // 2
            sl = [x[:, 0, s0 + i * self.slab:s0 + (i + 1) * self.slab].mean(1) for i in range(self.nslab)]
            img = torch.stack(sl, 1)
            if self.views == "dualraw":
                # A05: truly background-independent branch. prep_input centred the image as (I-1)/2,
                # so recover the positive image first, divide by ITS OWN p99, and only then centre.
                raw = (img * 2.0 + 1.0).clamp_min(0.0)
                q = torch.quantile(raw.flatten(1).float(), 0.99, dim=1).clamp_min(1e-3).view(-1, 1, 1, 1)
                img = torch.cat([img, (raw / q) * 2.0 - 1.0], 1)
            if self.views in ("dual", "bilat"):
                # second channel set divided by the scan's own p99. NOTE: this runs after prep_input's centring,
                # so it is effectively (I-B)/(Q-B) and still depends on the background estimate B. A truly
                # scale-invariant I/Q branch ('dualraw') was tested and scored worse (paired, 2 seeds).
                p = torch.quantile(img.flatten(1).float(), 0.99, dim=1).clamp_min(1e-3).view(-1, 1, 1, 1)
                img = torch.cat([img, img / p], 1)
            if self.views == "bilat":
                # A16: shared encoder per striatum (right mirrored onto left), symmetric fusion
                half = img.shape[3] // 2
                a = img[:, :, :, :half]
                b = torch.flip(img[:, :, :, half:], dims=[3])
                two = F.interpolate(torch.cat([a, b], 0), size=(self.size, self.size), mode="bilinear",
                                    align_corners=False)
                h = self.bb(two)
                ha, hb = h[:img.shape[0]], h[img.shape[0]:]
                return self.bil(torch.cat([ha + hb, (ha - hb).abs()], 1)).squeeze(1)
            if self.views == "hemi":
                # score each striatum separately (left mirrored onto right), combine with a noisy-OR:
                # the scan is abnormal if EITHER hemisphere is abnormal.
                half = img.shape[3] // 2
                a = img[:, :, :, :half]
                b = torch.flip(img[:, :, :, half:], dims=[3])
                two = torch.cat([a, b], 0)
                two = F.interpolate(two, size=(self.size, self.size), mode="bilinear", align_corners=False)
                l = self.fc(self.drop(self.bb(two))).squeeze(1)
                l1, l2 = l[:img.shape[0]], l[img.shape[0]:]
                log_pn = F.logsigmoid(-l1) + F.logsigmoid(-l2)          # log P(both hemispheres normal)
                log_pn = log_pn.clamp(min=-30.0, max=-1e-6)
                return torch.log1p(-torch.exp(log_pn)) - log_pn          # logit P(abnormal)
            img = F.interpolate(img, size=(self.size, self.size), mode="bilinear",
                                align_corners=False)
            feat = self.bb(img)
            if self.fusion_dim:
                z = ((feats.float() - self.feat_mu) / self.feat_sd).clamp(-6, 6)
                g = self.fmlp(z)
                if self.training:                               # modality dropout
                    keep_img = (torch.rand(feat.shape[0], 1, device=feat.device) > 0.15).float()
                    keep_tab = (torch.rand(feat.shape[0], 1, device=feat.device) > 0.15).float()
                    feat, g = feat * keep_img, g * keep_tab
                return self.ffc(self.drop(torch.cat([feat, g], 1))).squeeze(1)
            logit = self.fc(self.drop(feat)).squeeze(1)
            if return_aux and self.aux is not None:
                return logit, self.aux(feat)
            return logit

    # Architecture copied from github.com/ThomasBudd/dat_spect_ud (MIT License, (c) 2023 ThomasBudd)
    # so that its public DaT-SPECT weights can be fine-tuned; state-dict keys are identical.
    def _cnn(cin, cout):
        return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                nn.LeakyReLU(0.01, inplace=True)]

    class _Stacked(nn.Module):
        def __init__(self, cin, cout, nblocks, first_stride):
            super().__init__()
            self.first_conv = nn.Conv2d(cin, cout, kernel_size=first_stride, stride=first_stride, bias=False)
            self.blocks = nn.ModuleList([nn.Sequential(*(_cnn(cout, cout) + _cnn(cout, cout)))
                                         for _ in range(nblocks)])
            self.skip_inits = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(nblocks)])

        def forward(self, x):
            x = self.first_conv(x)
            for s, b in zip(self.skip_inits, self.blocks):
                x = x + s * b(x)
            return x

    class BuddNet(nn.Module):
        """DVR slab (6 x 2 mm axial slices, 72x72 px) -> max-pooled logit map."""
        def __init__(self, p75=1.27, dz=-2):
            super().__init__()
            self.body = nn.ModuleList([_Stacked(1, 16, 1, 1), _Stacked(16, 32, 1, (3, 3)),
                                       _Stacked(32, 64, 2, (3, 3))])
            self.logits = nn.Conv2d(64, 1, kernel_size=1)
            self.p75, self.dz = p75, dz

        def forward(self, x):  # x: prep_input-scaled (B,1,48,64,64)
            raw = x[:, 0] * 2.0 + 1.0
            c = raw.shape[1] // 2 + self.dz
            s = raw[:, c - 3:c + 3].mean(1) / self.p75            # (B, y, x)
            s = F.pad(s, (4, 4, 8, 0))[:, :72, :72]                 # posterior shift found in zero-shot fit
            s = ((s.clamp(0, 6.5) - 0.6048597782240399) / 0.6051688035793145).transpose(1, 2)
            h = s.unsqueeze(1)
            for m in self.body:
                h = m(h)
            return self.logits(h).flatten(1).max(1)[0]

    def make_net(ncfg, pretrained=False, seed=0, budd_dir=None):
        kind = ncfg["kind"]
        if kind == "3d":
            return Net3D(widths=tuple(ncfg.get("widths", (24, 48, 96, 160))))
        if kind == "budd":
            net = BuddNet()
            init = ncfg.get("init", "robust")
            if pretrained and init != "scratch":
                sd = torch.load(f"{budd_dir}/network_weights_{init}_{seed % 5}", map_location="cpu",
                                weights_only=True)
                net.load_state_dict(sd, strict=True)
            return net
        return Net2D(ncfg["backbone"], pretrained=pretrained, size=ncfg["size"],
                     nslab=ncfg.get("nslab", 3), slab=ncfg.get("slab", 4),
                     views=ncfg.get("views", "axial"), seq_half=ncfg.get("seq_half", 8),
                     aux_dim=len(ncfg.get("aux_cols", [])), fusion_dim=len(ncfg.get("fusion_cols", [])))

    def prep_input(x):
        return (x - 1.0) / 2.0

    def make_theta(B, device, train, gen=None, out=NET):
        """Affine theta mapping the output grid (NET, or STORE for wide inputs) into the STORE volume."""
        half_n = torch.tensor([out[2], out[1], out[0]], dtype=torch.float32, device=device) / 2
        half_s = torch.tensor([STORE[2], STORE[1], STORE[0]], dtype=torch.float32, device=device) / 2
        R = torch.eye(3, device=device).repeat(B, 1, 1)
        t = torch.zeros(B, 3, device=device)
        if train:
            def U(lo, hi, n=B):
                return lo + (hi - lo) * torch.rand(n, device=device)
            ax = [U(-8, 8), U(-8, 8), U(-12, 12)]  # about x (pitch), y (roll), z (yaw) deg
            ax = [a * math.pi / 180 for a in ax]
            cx, sx = torch.cos(ax[0]), torch.sin(ax[0])
            cy, sy = torch.cos(ax[1]), torch.sin(ax[1])
            cz, sz = torch.cos(ax[2]), torch.sin(ax[2])
            o, z = torch.ones(B, device=device), torch.zeros(B, device=device)
            Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
            Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
            Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
            s = U(0.9, 1.1).view(B, 1, 1)
            R = Rz @ Ry @ Rx * s
            t = torch.stack([U(-4, 4), U(-5, 5), U(-4, 4)], 1)  # voxels (x,y,z)
            flip = (torch.rand(B, device=device) < 0.5).float() * -2 + 1
            R = R * torch.stack([flip, o, o], 1).view(B, 1, 3)  # flip output x
        lin = R * half_n.view(1, 1, 3) / half_s.view(1, 3, 1)
        return torch.cat([lin, (t / half_s).unsqueeze(2)], 2)

    def sample(vol, theta, out=NET):
        import torch.nn.functional as F
        grid = F.affine_grid(theta, (vol.shape[0], 1) + tuple(out), align_corners=False)
        return F.grid_sample(vol, grid, mode="bilinear", padding_mode="zeros",
                             align_corners=False)

    def blur3d(x, sigma):
        if sigma <= 0.05:
            return x
        r = int(math.ceil(2.5 * sigma))
        k = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
        k = torch.exp(-0.5 * (k / sigma) ** 2)
        k = k / k.sum()
        C = x.shape[1]
        x = F.conv3d(F.pad(x, (0, 0, 0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1, 1).repeat(C, 1, 1, 1, 1), groups=C)
        x = F.conv3d(F.pad(x, (0, 0, r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1, 1).repeat(C, 1, 1, 1, 1), groups=C)
        x = F.conv3d(F.pad(x, (r, r, 0, 0, 0, 0), mode="replicate"), k.view(1, 1, 1, 1, -1).repeat(C, 1, 1, 1, 1), groups=C)
        return x

    def blur_aniso(x, sz, sy, sx):
        out = x
        for axis, s in ((2, sz), (3, sy), (4, sx)):
            if s <= 0.05:
                continue
            r = int(math.ceil(2.5 * s))
            k = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
            k = torch.exp(-0.5 * (k / s) ** 2)
            k = k / k.sum()
            shape = [1, 1, 1, 1, 1]
            shape[axis] = -1
            pad = [0, 0, 0, 0, 0, 0]
            pad[{2: 4, 3: 2, 4: 0}[axis]] = r
            pad[{2: 5, 3: 3, 4: 1}[axis]] = r
            out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(*shape))
        return out

    def augment(vol, blur_p=0.5, blur_max=1.3, noise=0.05, noise_p=0.3, per_sample=False, aniso=False,
                wide=False):
        """vol: (B,1,STORE) float on GPU -> (B,1,NET) augmented.
        Large blur_max / noise = 'unrealistic' augmentation (Buddenkotte & Buchert, JNM 2024)."""
        B = vol.shape[0]
        out = STORE if wide else NET
        x = sample(vol, make_theta(B, vol.device, True, out=out), out=out)
        if per_sample:
            # A11: each scan gets its own blur (optionally different along z), grouped into 4 levels
            do = torch.rand(B) < blur_p
            lvl = torch.randint(0, 4, (B,))
            edges = torch.linspace(0.3, blur_max, 4)
            outs = x.clone()
            for L in range(4):
                idx = torch.where(do & (lvl == L))[0]
                if len(idx) == 0:
                    continue
                s = float(edges[L])
                sz = s * float(torch.empty(1).uniform_(0.6, 1.6)) if aniso else s
                outs[idx] = blur_aniso(x[idx], sz, s, s)
            x = outs
        elif torch.rand(1).item() < blur_p:
            x = blur3d(x, float(torch.empty(1).uniform_(0.3, blur_max)))
        gain = 1 + 0.08 * torch.randn(B, 1, 1, 1, 1, device=vol.device).clamp(-2, 2)
        x = 1 + (x - 1) * gain
        nsd = noise * torch.rand(B, 1, 1, 1, 1, device=x.device) * 2
        x = x + nsd * torch.randn_like(x) * (torch.rand(B, 1, 1, 1, 1, device=x.device) < noise_p)
        return x

    def center(vol, flip=False, wide=False):
        if wide:
            return torch.flip(vol, dims=[4]) if flip else vol
        B = vol.shape[0]
        d = [(a - b) // 2 for a, b in zip(STORE, NET)]
        x = vol[:, :, d[0]:d[0] + NET[0], d[1]:d[1] + NET[1], d[2]:d[2] + NET[2]]
        return torch.flip(x, dims=[4]) if flip else x

    # ---------------- registration to a symmetric normal template ---------
    half_n = torch.tensor([NET[2], NET[1], NET[0]], dtype=torch.float32) / 2
    half_s = torch.tensor([STORE[2], STORE[1], STORE[0]], dtype=torch.float32) / 2

    def _rot(a):  # a: (B,3) radians about x,y,z
        B = a.shape[0]
        o, z = torch.ones(B, device=a.device), torch.zeros(B, device=a.device)
        cx, sx, cy, sy, cz, sz = (torch.cos(a[:, 0]), torch.sin(a[:, 0]), torch.cos(a[:, 1]),
                                  torch.sin(a[:, 1]), torch.cos(a[:, 2]), torch.sin(a[:, 2]))
        Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
        Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
        Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
        return Rz @ Ry @ Rx

    def _theta(p, out_half):
        """params (B,9) -> theta mapping an output grid (half size out_half) into STORE."""
        dev = p.device
        hs = half_s.to(dev)
        rot = 0.5 * torch.tanh(p[:, 0:3])
        tr = 12.0 * torch.tanh(p[:, 3:6])
        sc = torch.exp(0.2 * torch.tanh(p[:, 6:9]))
        L = _rot(rot) * sc.view(-1, 1, 3)
        lin = L * out_half.to(dev).view(1, 1, 3) / hs.view(1, 3, 1)
        return torch.cat([lin, (tr / hs).unsqueeze(2)], 2)

    def _ncc(a, b):
        a = a.flatten(1)
        b = b.flatten(1)
        a = a - a.mean(1, keepdim=True)
        b = b - b.mean(1, keepdim=True)
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)

    def register(vol, template, iters=(50, 50), sigmas=(2.0, 1.0), rigid=False):
        """vol (B,1,STORE), template (1,1,NET) -> params (B,9). Samples are independent.
        rigid=True freezes the three scale parameters (A15: preserves head size/shape)."""
        B = vol.shape[0]
        p = torch.zeros(B, 9, device=vol.device, requires_grad=True)
        if rigid:
            p.register_hook(lambda g: torch.cat([g[:, :6], torch.zeros_like(g[:, 6:])], 1))
        opt = torch.optim.Adam([p], lr=0.03)
        for n_it, sg in zip(iters, sigmas):
            mv = torch.clamp(blur3d(vol, sg), 0, 10)
            tp = torch.clamp(blur3d(template, sg), 0, 10).expand(B, -1, -1, -1, -1)
            for _ in range(n_it):
                w = F.grid_sample(mv, F.affine_grid(_theta(p, half_n), (B, 1) + NET, align_corners=False),
                                  mode="bilinear", padding_mode="zeros", align_corners=False)
                loss = (1 - _ncc(w, tp)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return p.detach()

    def warp_store(vol, p):
        th = _theta(p, half_s)
        return F.grid_sample(vol, F.affine_grid(th, (vol.shape[0], 1) + STORE, align_corners=False),
                             mode="bilinear", padding_mode="zeros", align_corners=False)

    def register_all(crops, template, device, bs=64, rigid=False):
        """crops: numpy (N,STORE) float16 -> registered numpy (N,STORE) float16, final NCC."""
        tpl = torch.as_tensor(template, dtype=torch.float32, device=device)[None, None]
        out = np.zeros_like(crops)
        nccs = np.zeros(len(crops), np.float32)
        for b in range(0, len(crops), bs):
            v = torch.as_tensor(np.asarray(crops[b:b + bs]), device=device).float().unsqueeze(1)
            with torch.enable_grad():
                p = register(v, tpl, rigid=rigid)
            with torch.no_grad():
                r = warp_store(v, p)
                nccs[b:b + bs] = _ncc(center(r), tpl.expand(len(v), -1, -1, -1, -1)).cpu().numpy()
            out[b:b + bs] = r[:, 0].cpu().numpy().astype(np.float16)
        return out, nccs

    def build_template(crops_normal, device, n_iter=3):
        v = torch.as_tensor(np.asarray(crops_normal), device=device).float().unsqueeze(1)
        with torch.no_grad():
            t = center(v).mean(0, keepdim=True)
            t = 0.5 * (t + torch.flip(t, dims=[4]))
        for _ in range(n_iter):
            regs = []
            for b in range(0, len(v), 64):
                with torch.enable_grad():
                    p = register(v[b:b + 64], t)
                with torch.no_grad():
                    regs.append(center(warp_store(v[b:b + 64], p)).sum(0, keepdim=True))
            with torch.no_grad():
                t = torch.cat(regs).sum(0, keepdim=True) / len(v)
                t = 0.5 * (t + torch.flip(t, dims=[4]))
        return t[0, 0].cpu().numpy()

    return dict(Net3D=Net3D, Net2D=Net2D, BuddNet=BuddNet, make_net=make_net, prep_input=prep_input, augment=augment,
                center=center, blur3d=blur3d, register_all=register_all,
                build_template=build_template)


# ----------------------------------------------------------------------------
# VOI atlas on the template + VOI binding-ratio features
# ----------------------------------------------------------------------------
VOI_NAMES = {1: "caud", 2: "put_ant", 3: "put_post"}


def make_atlas(template: np.ndarray, n_side: int = 700) -> np.ndarray:
    """Label volume (NET) : side offset 0 (x<W/2) / 3 (x>=W/2) + {1 caud, 2 put ant, 3 put post};
    9 = reference (posterior non-striatal brain)."""
    t = ndi.gaussian_filter(template.astype(np.float64), 0.7)
    D, H, W = t.shape
    lab = np.zeros(t.shape, np.int8)
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    box = (np.abs(zz - D / 2) < 12) & (np.abs(yy - H / 2) < 20) & (np.abs(xx - W / 2) < 22)
    for side, sel in ((0, xx < W // 2), (3, xx >= W // 2)):
        cand = np.where(box & sel, t, -np.inf).ravel()
        idx = np.argsort(cand)[::-1][:n_side]
        m = np.zeros(t.size, bool)
        m[idx] = True
        m = m.reshape(t.shape)
        pts = np.stack([np.abs(xx[m] - (W - 1) / 2.0), yy[m].astype(float)], 1)
        # 2-means on (medial distance, AP): caudate = medial+anterior cluster
        c = np.array([pts[pts[:, 0] < np.median(pts[:, 0])].mean(0),
                      pts[pts[:, 0] >= np.median(pts[:, 0])].mean(0)])
        for _ in range(20):
            a = np.argmin(((pts[:, None, :] - c[None]) ** 2).sum(-1), 1)
            c = np.array([pts[a == k].mean(0) if np.any(a == k) else c[k] for k in range(2)])
        caud_k = int(np.argmin(c[:, 0] - 0.5 * c[:, 1]))
        is_caud = a == caud_k
        coords = np.argwhere(m)
        put = coords[~is_caud]
        ymed = np.median(put[:, 1]) if len(put) else H / 2
        for (z, y, x), ic in zip(coords, is_caud):
            if ic:
                lab[z, y, x] = side + 1
            else:
                lab[z, y, x] = side + (2 if y >= ymed else 3)
    stri = ndi.binary_dilation(lab > 0, iterations=4)
    ref = (~stri) & (yy < H * 0.3) & (np.abs(zz - D / 2) < 14) & (t > 0.7) & (t < np.percentile(t, 90))
    if ref.sum() < 100:
        ref = (~stri) & (t > 0.7) & (t < np.percentile(t, 90))
    lab[ref] = 9
    return lab


def voi_features(reg_crop: np.ndarray, atlas: np.ndarray, ncc: float) -> dict:
    v = _center(reg_crop.astype(np.float32), NET)
    ref = float(np.mean(v[atlas == 9])) + 1e-3
    out = {"v_ref": ref, "v_ncc": float(ncc)}
    per = {}
    for side, nm in ((0, "a"), (3, "b")):
        d = {}
        vals = {}
        for k, name in VOI_NAMES.items():
            x = v[atlas == side + k]
            vals[name] = x
            d[f"{name}_sbr"] = float(x.mean()) / ref - 1
            d[f"{name}_top"] = float(np.sort(x)[::-1][:max(5, len(x) // 4)].mean()) / ref - 1
        put = np.concatenate([vals["put_ant"], vals["put_post"]])
        allv = np.concatenate([put, vals["caud"]])
        d["put_sbr"] = float(put.mean()) / ref - 1
        d["str_sbr"] = float(allv.mean()) / ref - 1
        d["put_caud"] = (d["put_sbr"] + 0.05) / (d["caud_sbr"] + 0.05)
        d["post_ant"] = (d["put_post_sbr"] + 0.05) / (d["put_ant_sbr"] + 0.05)
        d["post_caud"] = (d["put_post_sbr"] + 0.05) / (d["caud_sbr"] + 0.05)
        per[nm] = d
    for k in per["a"]:
        a, b = per["a"][k], per["b"][k]
        out[f"v_{k}_min"] = min(a, b)
        out[f"v_{k}_max"] = max(a, b)
        out[f"v_{k}_mean"] = 0.5 * (a + b)
        out[f"v_{k}_asym"] = abs(a - b) / (abs(a) + abs(b) + 1e-3)
    return out


NCC_RETRY = 0.45   # every training scan registers above this; below it the crop is almost certainly wrong


def postprocess(crops: np.ndarray, metas: list, template: np.ndarray, atlas: np.ndarray, T: dict,
                device: str, paths=None):
    """Register crops to the template and compute all tabular features (per scan).

    If a scan registers far worse than anything seen in training, its striatum was probably
    mis-located: redo that scan with the fallback localisation and keep whichever registers better.
    """
    reg, ncc = T["register_all"](crops, template, device)
    if paths is not None:
        bad = [i for i, c in enumerate(ncc) if c < NCC_RETRY]
        if bad:
            alt_crops, keep = [], []
            for i in bad:
                try:
                    c2, m2 = preprocess(paths[i], mode=1)
                    alt_crops.append(c2)
                    keep.append(i)
                except Exception:
                    pass
            if keep:
                alt_reg, alt_ncc = T["register_all"](np.stack(alt_crops), template, device)
                for j, i in enumerate(keep):
                    if alt_ncc[j] > ncc[i]:
                        reg[i], ncc[i] = alt_reg[j], alt_ncc[j]
    rows = []
    for i in range(len(crops)):
        try:
            f = features(reg[i], metas[i])
            f.update(voi_features(reg[i], atlas, ncc[i]))
        except Exception:
            f = {}
        rows.append(f)
    return reg, rows


# ----------------------------------------------------------------------------
# Logistic model on features, optionally with the voxel-size trend removed
# (trend fitted on training normals only; applied per scan from its own voxel size)
# ----------------------------------------------------------------------------
def _zoom_basis(z):
    z = np.asarray(z, dtype=np.float64)
    return np.stack([np.ones(len(z)), z, z ** 2], 1)


def lr_apply(p, X, zoom=None):
    X = np.asarray(X, dtype=np.float64)
    if p.get("adj") is not None:
        z = np.full(len(X), 2.4) if zoom is None else np.asarray(zoom, dtype=np.float64)
        X = X - _zoom_basis(z) @ np.array(p["adj"])
    Z = np.clip((X - np.array(p["mu"])) / np.array(p["sd"]), -6, 6)
    return Z @ np.array(p["coef"]) + p["b"]
