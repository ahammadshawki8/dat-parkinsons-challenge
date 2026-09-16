#!/bin/bash
# Rebuilds the final v8 ensemble (public LB 0.2356) from the competition training data.
# Expects:  data/niftis/*.nii.gz  and  data/train_labels.csv   (download from DrivenData; not included).
# GPU runs are not bit-exact (cuDNN non-determinism): expect run-to-run differences of ~0.004 per model.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

# 1. preprocessing: localise striatum, normalise, crop (CPU, ~3 min with 8 workers)
$PY solution/prep.py data cache

# 2. cross-validated training (the first run also builds the template, registration and VOI features)
for run in v3_s2026 v3_s7 e_10fold v5_10f v6_dual10 v6_dual5b a08_seq a08_seq_s12 v8_seq_b; do
  $PY scripts/exp.py "$run" "configs/$run.json"
done

# 3. bag the runs exactly as they were submitted
$PY scripts/assemble_v3.py                 # runs/v3_bag
$PY scripts/assemble_v5.py --with-v3       # runs/v5_bag
$PY scripts/assemble_v6.py                 # runs/v6_bag
$PY scripts/assemble_v8.py v8 2            # runs/v8 + submission_v8.zip

# 4. verify the archive end to end on the organisers' smoke data (put it in data/smoke/)
bash scripts/verify_zip.sh v8
