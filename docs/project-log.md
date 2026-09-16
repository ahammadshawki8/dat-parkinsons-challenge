# DaT Parkinson's Challenge — project record

DrivenData competition: classify dopamine transporter (DaT) SPECT scans as normal (0) or abnormal (1).
Metric: **log loss** (lower better), AUROC reported for reference. Code-execution submission: a `submission.zip`
with `main.py` at the root, run offline on one A100, 24 vCPU, 220 GB RAM, 3 h limit, Python 3.12.
Deadline 2026-09-16 23:59 UTC. Training data: 1362 scans (747 abnormal / 615 normal). Test set ≈ 640 scans
(≈2000 total per the organisers' press material), split into public/private; sizes not disclosed.

---

## 1. Submission history (the only ground truth)

| Submission | What changed | Local nested CV | Public LB | AUROC (LB) | Rank |
|---|---|---|---|---|---|
| v1 | registration + VOI features + EfficientNet-B0 ×3 + ResNet26d ×3 + LightGBM/logistic stack | 0.2120 | 0.2463 | 0.962 | #20 |
| v2 | v1 bagged over 2 fold splits | 0.2111 | 0.2446 | — | #19 |
| v3 | **strong blur/noise augmentation** + voxel-size-adjusted logistic | 0.2035 | **0.2405** | 0.9643 | #16 |
| v3cal | v3 + temperature 1.05 + floor 0.01 | 0.2066 | 0.2408 | 0.9643 | #16 |
| v6 | **dual-normalised CNNs** + 10-fold + everything bagged (179 files, 1.44 GB) | **0.2020** | **0.2403** | — | #15 |
| v7 | v6 minus ResNet26d + **slice-sequence CNN** (weight 2) | 0.1988 | **0.2364** | — | #14 |
| v8 | v7 with the sequence model bagged over 2 splits (6 seeds) | 0.1990 | **0.2356** | 0.9649 | #14 |
| final A | v8 with sequence weight 4 | 0.1991 | pending | | |
| final B | dual + sequence only (drops plain EfficientNet), 580 MB | 0.1991 | pending | | |

Leader: 0.2154 (AUROC 0.9721). Reference: South-Wing 0.2263 at AUROC 0.9645 — same discrimination as v3,
0.014 better log loss, and v3cal proved that difference is **not** calibration.

**The key pattern:** early structural gains transferred (0.2463 → 0.2405), later ensemble gains did not
(0.0015 locally → 0.0002 on the board). The pipeline saturated against this test set.

---

## 2. Pipeline (what the final submission does)

`solution/datlib.py` is shared by training and inference, so preprocessing can never diverge.
Each scan is processed independently (competition rule).

1. **Load** — `nibabel`, `as_closest_canonical` (RAS), 4-D averaged, NaN/negatives cleaned.
2. **Localise the striatum** (no atlas): 4 mm coarse grid → Otsu head mask → find vertex → search the
   brightest smoothed voxel 25–125 mm below it within ±35 mm lateral / ±50 mm AP of the head centroid;
   midline from the head-mask centroid (robust to one-sided uptake loss).
   **Fallback**: if a scan later registers below NCC 0.45 (no training scan does), it is re-processed with
   an unconstrained search and the better registration kept — insurance for odd test files.
3. **Intensity** — divide by a trimmed mean (p25–p85) of head voxels ±50 mm around the striatal plane
   ⇒ "background = 1.0" units, removing dose/scanner sensitivity.
4. **Crop** — one affine resampling from the original voxels to 2 mm isotropic, `STORE = 56×80×80`
   (112×160×160 mm) as float16; network input `NET = 48×64×64`.
5. **Registration** — symmetric template built from training normals (register → average → mirror, ×3).
   Each scan affinely registered (9 params, Adam on GPU, 2 scales, NCC loss). Median NCC 0.77 → 0.85.
6. **VOI features** (~173) — atlas derived from the template (threshold → hemisphere split → k-means
   caudate vs putamen, anterior/posterior halves). Binding ratios, extents, shape, asymmetry, all reduced
   to min/max/mean/asymmetry so the affected side doesn't matter.
7. **Models**
   - **CNNs (2D, ImageNet-pretrained)** on three 8 mm axial slabs through the striatum, trained with
     **strong "unrealistic" augmentation**; two input styles: standard (background-normalised) and
     **dual-normalised** (adds channels divided by the scan's own 99th percentile).
   - **Tabular**: LightGBM + logistic regression (voxel-size-adjusted).
   - **Stack**: logistic regression on out-of-fold logits (CNNs averaged into one column), nested-CV checked.
   - **Bagging**: two independent fold splits, plus 10-fold variants.
8. **Inference** (`main.py`) — pooled preprocessing (worker count capped by free RAM), sequential retry for
   failures, GPU registration + features, all checkpoints averaged with flip TTA, writes
   `/code_execution/submission.csv`. Logs contain only counts and timings.

---

## 3. What worked

| Change | Effect | Notes |
|---|---|---|
| **Template registration + VOI binding ratios** | 0.2319 → 0.2153 | Biggest win; yields clinical caudate/putamen SBRs |
| **Strong "unrealistic" augmentation** (blur σ ≤ 8 mm, noise 0.3, p=0.8/0.7) | CNNs 0.2568 → 0.2335; held-out-scanner CNNs 0.3529 → 0.2716; stack 0.2120 → 0.2075 | Buddenkotte & Buchert, *JNM* 2024. **The only change with a clear LB effect** (0.2446 → 0.2405) |
| **Dual normalisation** (extra channels ÷ scan's own p99) | single CNN 0.2372 → 0.2249; best CNN column 0.2197 | Removes dependence on our background estimate; 3× run-to-run noise |
| 10-fold training (90% data/model) | single seed 0.2372 → 0.2316 | Real for one seed; **vanishes once 3 seeds are averaged** |
| Bagging a second fold split | 0.2120 → 0.2111; 0.2057 → 0.2035 | Small, free |
| Multi-seed ensembling (3/backbone) | ~ −0.005 | Diminishing after 3 |
| Voxel-size-adjusted logistic | held-out-scanner 0.2587 → 0.2558 | Trend fitted on training normals only |

## 4. What did NOT work (all measured, then dropped)

| Idea | Result |
|---|---|
| Calibration: temperature + probability floor | Simulation predicted −0.009; **real LB +0.0003** (v3cal). Refuted |
| Label smoothing 0.06 | +0.0008 with 1 seed = noise; 3 seeds → 0.2059 vs 0.2057 |
| Hemisphere MIL (noisy-OR over striata) | 0.2487 vs 0.2372; no ensemble gain |
| 3D CNN (weak and strong augmentation) | 0.2757; stack 0.2069 vs 0.2057 |
| Tri-plane views | 0.2711 |
| ConvNeXt-Tiny @224 | 0.2486, no stack gain |
| Dual @256 px | 0.2428 (SPECT resolution is ~8–10 mm; more pixels add nothing) |
| Dual + thicker slabs (12 mm) | stack unchanged |
| Dual ResNet26d | 0.2611, *hurts* ensemble (0.2037) |
| 50-epoch training | 0.3328 — overfits |
| Multi-registration TTA | −0.0003 (first evaluation showed −0.05 but that was **my leak**, see §6) |
| Richer TTA (shifts, yaw) | +0.0004 |
| Model soup (weight averaging) | 0.3585 vs 0.2261 — fails across seeds |
| Label-noise sample weighting (inner-CV derived) | −0.002 = noise |
| Whole-head context features (brain size, ventricle proxy, cortical uptake) | 0.2046 vs 0.2047 |
| Multi-template registration (normal/left/right/bilateral) | Worse — chosen template correlates with the label |
| Resolution harmonisation (blur to common resolution) | Clearly worse |
| External pretrained DaT nets (ThomasBudd/dat_spect_ud, MIT) zero-shot + fine-tuned | Zero-shot AUC 0.947 but 0.93 correlated with ours; fine-tuned 0.2975 |
| PCA voxel model, SVM, MLP, ExtraTrees | ≤0.0005 stack gain |
| Non-linear stackers (quadratic, LightGBM, isotonic) | 0.2073–0.2100 vs linear 0.2035 |
| Extra VOI profile features, scanner-aware stacker | ≤0.0004 |

## 5. Validation methodology

- **Random 5-fold** stratified by label × scanner group; **nested CV** for the stacker.
- **Held-out-scanner CV** (`StratifiedGroupKFold` on voxel size) as a leaderboard proxy. Fitting the stack on
  random folds and applying it to held-out-scanner predictions gave **0.2444** when v2 scored **0.2446** —
  and it correctly ranked strong augmentation as a big win. It over-predicted the *size* of later gains.
- **Run-to-run noise ≈ 0.004** (two identical configs gave 0.2372 and 0.2415). Any single-run difference
  below that is meaningless — this invalidated several "promising" results.
- **LB noise ≈ ±0.028** (bootstrap, ~320-scan public set): #1 to #16 is about one standard deviation.
- **Stress tests**: localisation survives cut FOV, 10 mm smoothing, quarter counts, 15–25° tilt, lateral
  crops and extracerebral hot spots (median error ≤2 mm, worst 1% of cases).
- **Error analysis**: ~44% of remaining loss comes from scans whose own uptake contradicts their label
  (abnormal-labelled scans with putamen uptake at the 39th–54th percentile of clear normals, and vice
  versa). Unfixable by any image model. Registration quality is *not* a limiting factor.
- **Every team's LB is worse than their CV**: aryankathpalia 0.2547→0.3398, SAIFULLAH 0.290→0.322,
  MICADEE 0.2310→0.2586, ours 0.2020→0.2403.

## 6. Mistakes made (worth remembering)

1. **Evaluation leak**: mapping fold indices across two bagged splits let split-B models predict scans they
   trained on, producing a fake 0.1527. Always check that "too good" results aren't leaks.
2. **Trusting simulation over measurement**: the label-noise simulation justified calibration; the LB refuted
   it. Post-processing changes should be validated on the board, not in a model of the board.
3. **Single-run conclusions**: label smoothing and 10-fold both looked better than they were until run-to-run
   noise was measured.
4. **Windows sleep kills CUDA contexts** — jobs hang alive at 0% GPU. `powercfg -change -standby-timeout-ac 0`.

## 7. Sources

- Runtime: <https://github.com/drivendataorg/competition-sfmn-parkinsons-runtime>
  (torch 2.12.1+cu129, timm 1.0.27, lightgbm 4.6.0, numpy 2.2.6, pandas 3.0.3, scipy 1.15.3, nibabel 5.4.2).
- **Buddenkotte & Buchert, *J Nucl Med* 2024**, "Unrealistic Data Augmentation…":
  <https://jnm.snmjournals.org/content/65/9/1463> → strong augmentation, our biggest LB-visible gain.
- **Budenkotte et al., *EJNMMI* 2023** (uncertainty detection): slab definition, DVR normalisation
  (75th percentile of a non-striatal reference), 12 mm slab, 72×72 crop (via Europe PMC full text).
- **ThomasBudd/dat_spect_ud** (MIT) — public pretrained DaT nets, tested and rejected.
- Wenzel et al., *EJNMMI* 2019 — multi-site CNN robustness for FP-CIT SPECT.
- "Self-normalized Classification of Parkinson's Disease DaTscan Images" (arXiv 2112.13637) — inspiration
  for **dual normalisation**, which did work.
- Competitor repos: `guelmbaye/dat-parkinsons`, `aryankathpalia/dat-spect-parkinsons-competition`,
  `SAIFULLAH-SHARAFAT/DaT-Parkinson-Challenger`.
- Forum rulings: best **private** score counts automatically; team merges **add each member's quota**;
  weekly limit is a rolling 7-day window from midnight UTC of the submission day; PPMI-trained models are
  **not** prize-eligible (rules out ParkinSIGHT and every large public DaT dataset).

## 8. Rules compliance

- **Data never leaves the machine.** Organiser guidance: scans, labels, predictions and images must not be
  shared with AI assistants; only code, generic errors and scalar metrics may be. All analysis here ran
  locally with only aggregate statistics (groups ≥9 cases, CV scores) reported back. Visual review was
  produced as local-only figures in `eda/` and were not shared or published.
- External models used: ImageNet-pretrained timm backbones only. No PPMI data.
- Each test scan processed independently; no cross-sample statistics, no pseudo-labelling.
- Local data must be deleted after the competition ends.

## 9. Layout

```
solution/          datlib.py (shared), prep.py, train.py, main.py (submission entrypoint)
cache/             crops.npy, reg_crops.npy, template.npy, features*.csv, table.csv, noise_weights.npy
runs/<name>/       oof.csv + assets/ (checkpoints, gbm models, config.json)
data/              niftis/, train_labels.csv, smoke/
submission_v6.zip  final submitted model (1.44 GB, LB 0.2403)
```

Scripts: `exp.py` (one CV config), `assemble_v3/v5/v6.py` (bag → zip), `proxy_eval.py` (scanner-out proxy),
`stress_localise.py` / `stress_rot.py` / `stress_model.py` (robustness), `parity_v3.py` (train/inference
parity), `noise_weights.py`, `soup_test.py`, `multireg_eval2.py`, `eda_for_user.py`.

## 10. Verification checklist (run before every submission)

1. Clean unzip; `main.py` at root; zip integrity.
2. Run on the organisers' smoke data; check columns, row order, range, NaN.
3. Forced worker failure → retry path reproduces identical output.
4. End-to-end parity vs the training pipeline (≤0.001 mean difference).
5. Python 3.12 with runtime-pinned packages: preprocessing, LightGBM load, timm strict load.

## 11. Conclusion

The pipeline is at the ceiling of what these 1,362 scans support. Roughly 44% of the residual loss is label
contradiction that no image model can fix, and the remainder is borderline cases. Sixteen ideas were tested
and rejected in the final push; two (strong augmentation, dual normalisation) were real. Further progress
would need either more labelled data (PPMI is barred) or information beyond the scan — age and sex, which
the organisers themselves flag as a post-challenge research direction.


## 12. Research-plan follow-up (RESEARCH_DaT_Leaderboard_Strategy_2026-09-16.md)

An external review corrected several over-stated conclusions above (sections 4 and 11): one temperature+floor
submission did not refute all calibration; label/feature contradiction is not proof of label noise; the
~320-case public set size is not known; single-seed comparisons should be paired. It also found real code
issues. Everything below was tested with paired folds/seeds and whole-set scalar summaries only.

| ID | Experiment | Result | Verdict |
|---|---|---|---|
| A01 | Evaluated vs deployed ensemble weights | Mismatch real (up to 0.22 weight difference) but held-out loss 0.2015 vs 0.2014 | Fixed: `main.py` now honours explicit per-file, per-model and group weights; assembly asserts consistency |
| A03 | Probability pooling vs logit stack | 0.2071 (fitted) / 0.2079 (equal) vs 0.2014 | Rejected |
| A04 | Temperature / affine / beta calibration, doubly nested; floors separately | All 0.2014-0.2015; floors 0.002-0.01 worsen to 0.2019-0.2044 | Rejected - stack already calibrated in-distribution |
| A05 | Truly background-independent dual branch (I/p99 before centring) | Paired 2 seeds: 0.2337 vs current 0.2220; stack 0.2078 vs 0.2048 | Rejected - current (I-B)/(Q-B) branch is better; comment corrected |
| A06 | Frozen DINOv2 ViT-S/14 (Apache-2.0) + fold-internal PCA/logistic | Alone 0.3280 / AUC 0.934; stack -0.0002 | Rejected before fine-tuning, per plan |
| A07 | Multi-reference (occipital, extrastriatal p75) ratios | Stack 0.2035 vs 0.2027 | Rejected |
| A17 | Integrated excess uptake at 3 VOI dilations | Stack 0.2029 vs 0.2027 | Rejected |
| **A08** | **Slice sequence: 9 slices x shared EfficientNet-B0 + BiGRU + gated attention** | Seed 0 0.2315; 3 seeds 0.2272; corr with dual 0.939; **v6 + seq -0.0026 (0.2014 -> 0.1988), 93% of paired bootstrap resamples improve** | **Only positive result**; below the 0.005 gate |
| A08 robustness | Held-out-scanner training | seq 0.2612 vs dual 0.2567; v7-like stack 0.2342 vs v6-like 0.2330 | No robustness gain |

v7 (`submission_v7.zip`, 928 MB) = v6 without ResNet26d + slice-sequence group member (weight 2), nested OOF
0.1988, fully verified (format, forced-failure retry, parity 0.0005 with weights honoured, timm 1.0.27 strict
load, Python 3.12 pins). It does not meet the plan's 0.005 gate; submitting it is a free extra private draw.
Not yet done from the plan: A02 fold-specific template/atlas (validity, not score), A09-A26.


## 13. Final night (2026-09-16, after v7)

v7's leaderboard gain (−0.0039) was larger than its local gain (−0.0026), so the slice-sequence direction
was pursued first; v8 (more sequence bagging) moved the board only −0.0008, i.e. averaging is saturated.

Screened afterwards against v8 (paired folds/seeds; marginal gain when added to the v8 CNN group) — **all rejected**:

| Item | Alone | Added to v8 |
|---|---|---|
| Wide sequence, 13 slices / ±24 mm | 0.2416 | ±0.0000 |
| ResNet26d sequence | 0.2643 | +0.0013 |
| A10 regional auxiliary targets | 0.2399 | +0.0006 |
| A16 learned bilateral comparison | 0.2431 | +0.0005 |
| A11 per-scan anisotropic blur | 0.2325 | ±0.0000 |
| A12 frozen BatchNorm | 0.2349 | +0.0004 |
| A09 image + VOI fusion network | **0.2234 (best single model)** | +0.0010 (duplicates the tabular models) |
| A15 rigid-only registration | 0.2551 | +0.0009 |
| A24 disagreement-conditioned shrinkage | stack 0.2021 vs 0.1990 | — |
| A23 wide field of view (160 mm) | 0.2415 | +0.0009 |
| A13 SAM | 0.2345 | +0.0005 |
| A18 acquisition-balanced sample weights | 0.2439 | +0.0006 |
| A25 mixup α=0.2 | 0.2580 | ±0.0000 |

Not run (per the plan's own priority notes): A02 fold-specific template (evaluation-only), A14 ROI/pretrained
3D, A19 MixStyle/CORAL/Fishr, A20 self-supervised pretraining, A21 segmentation/annotation, A22 radiomic
residual, A25 distillation (cannot be evaluated without leakage in the time left), A26 (no eligible data).

Final two uploads are deliberate alternative bets with identical local scores (see `SUBMIT_NOW.md`).
All code paths added tonight are opt-in; shipped models only use views axial/dual/seq.
