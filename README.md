# DaT Parkinson's Challenge — solution and experiment record

Code and write-up for team **DataCouples** in the [DrivenData DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)
(French Society of Nuclear Medicine, 2026): predict the probability that a dopamine transporter
(DaT) SPECT scan is abnormal, scored by log loss.

**Project site with charts and the full story:** https://ahammadshawki8.github.io/dat-parkinsons-challenge/

| | |
|---|---|
| Best public log loss | **0.2356** (AUROC 0.9649), from 0.2463 at the first submission |
| Best public rank | #14 at the close |
| Local nested CV of the final ensemble | 0.1990 |
| Experiments run | ~60, of which 5 changes improved the model |

## The final pipeline

1. **Load and orient** each NIfTI to RAS; average 4-D volumes.
2. **Find the striatum** without an atlas: head mask, vertex, brightest smoothed voxel in a band
   below it, midline from the head centroid. A fallback re-localises any scan that later registers
   worse than every training scan did.
3. **Normalise intensity** to a trimmed background mean, and **crop** a 112×160×160 mm box at 2 mm.
4. **Register** each crop (9-parameter affine, GPU, NCC) to a symmetric template built from normal
   training scans.
5. **Measure** caudate / anterior putamen / posterior putamen binding ratios and shape features on
   an atlas derived from the template (~173 left/right-invariant features).
6. **Model**
   - EfficientNet-B0 on three axial striatal slabs, and a **dual-normalised** variant
     (extra channels divided by the scan's own 99th percentile);
   - a **slice-sequence** network: 9 thin slices through a shared EfficientNet-B0, a BiGRU and
     gated attention;
   - LightGBM and a voxel-size-adjusted logistic regression on the measurements;
   - all CNNs trained with strong "unrealistic" blur/noise augmentation, bagged over two fold
     splits and 10-fold runs.
7. **Stack** with a logistic regression on out-of-fold logits; one explicit weight table drives
   both the stacker's inputs and the deployed ensemble.

## What worked

| Change | Local effect | Why |
|---|---|---|
| Template registration + regional binding ratios | 0.2319 → 0.2153 | Turns the image into the quantities readers actually judge |
| Strong blur/noise augmentation ([Buddenkotte & Buchert 2024](https://jnm.snmjournals.org/content/65/9/1463)) | CNN on unseen scanners 0.3529 → 0.2716 | Makes the CNNs indifferent to camera resolution and noise |
| Dual normalisation | single CNN 0.2372 → 0.2249 | A second view of intensity that doesn't hinge on one background estimate |
| Slice-sequence model | ensemble 0.2014 → 0.1988; **public −0.0039** | Keeps the superior–inferior profile that slab averaging throws away |
| Bagging over fold splits | ~−0.002 each time | Variance reduction |

Most other ideas — calibration families, 3D CNNs, MIL, tri-plane, fusion networks, rigid
registration, SAM, mixup, DINOv2 features, extra features, and more — were measured and rejected.
The [project site](https://ahammadshawki8.github.io/dat-parkinsons-challenge/) and
[`docs/project-log.md`](docs/project-log.md) list each one with its numbers.

## Repository layout

```
solution/      datlib.py (shared by training and inference), prep.py, train.py, main.py (submission entrypoint)
scripts/       exp.py (one CV run), assemble_v3/v5/v6/v8.py (bagging), make_variant.py,
               ensemble_weights.py, parity.py, verify_zip.sh, eval_candidate.py, proxy_eval.py
configs/       the nine training runs behind the final ensemble
experiments/   the experiments behind the "what didn't work" table (expect the cache/ and runs/ layout)
docs/          project website (GitHub Pages), project log, research plan
reproduce.sh   end-to-end rebuild of the final ensemble
```

## Reproduce

1. Register for the competition and download the training data. Place the scans in
   `data/niftis/` and the labels in `data/train_labels.csv`, and the smoke-test data in `data/smoke/`.
2. Create a Python 3.12 environment and install PyTorch for your CUDA version, then
   `pip install -r requirements.txt`.
3. Run `bash reproduce.sh`. On one 8 GB GPU the full rebuild takes roughly 5 hours; the result is
   `submission_v8.zip`, verified against the smoke data.

GPU training isn't bit-exact (cuDNN), so expect individual models to differ by ~0.004 in log loss
between runs; the ensemble is much more stable.

## Data and rules

This repository contains **code, configuration and whole-dataset aggregate metrics only**. No scans,
labels, per-scan predictions, derived features or trained weights are included, as the competition
rules require. The only pretrained weights used are ImageNet checkpoints from `timm`.

Licensed under the [MIT License](LICENSE).
