"""Run one cross-validated training configuration.

usage: python scripts/exp.py RUN_NAME CONFIG_JSON_OR_FILE [CACHE_DIR]
Writes runs/RUN_NAME/{oof.csv, assets/}. CACHE_DIR defaults to cache/ (built by solution/prep.py).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
import train  # noqa: E402

run, spec = sys.argv[1], sys.argv[2]
cfg = json.loads(Path(spec).read_text()) if spec.endswith(".json") else json.loads(spec)
cache = sys.argv[3] if len(sys.argv) > 3 else "cache"
train.main(str(ROOT / cache), str(ROOT / "data" / "train_labels.csv"), str(ROOT / "runs" / run), cfg=cfg)
