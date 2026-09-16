#!/bin/bash
# usage: verify_zip.sh NAME   (checks submission_NAME.zip end to end)
set -e
NAME=$1
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${TMPDIR:-/tmp}/vchk_$NAME"
S="$ROOT/data/smoke"
rm -rf "$DIR" && mkdir -p "$DIR" && cd "$DIR"
unzip -q "$ROOT/submission_$NAME.zip"
${PYTHON:-python} -c "
import zipfile, json, os
z = zipfile.ZipFile('$ROOT/submission_$NAME.zip'); n = z.namelist()
c = json.load(open('assets/config.json'))
miss = [f for k in c['nets'] for f in k['files'] if not os.path.exists('assets/' + f)] + [f for f in c['gbm'] if not os.path.exists('assets/' + f)]
print('[$NAME] integrity', z.testzip() is None, '| main.py at root', 'main.py' in n, '| roots', sorted({x.split('/')[0] for x in n}),
      '| nets', [(k['name'], len(k['files'])) for k in c['nets']], '| group weights', c['group_weights']['cnn2d'], '| missing', len(miss))
assert z.testzip() is None and 'main.py' in n and not miss"
DAT_DATA_DIR=$S ${PYTHON:-python} main.py > run.log 2>&1
tail -1 run.log
DAT_DATA_DIR=$S DAT_OUT=retry.csv ${PYTHON:-python} -c "
import main
class F:
    def __init__(s, *a, **k): pass
    def __enter__(s): return s
    def __exit__(s, *a): return False
    def map(s, fn, it, chunksize=1): return [(None, None, False) for _ in it]
main.ProcessPoolExecutor = F; main.main()" > retry.log 2>&1
${PYTHON:-python} -c "
import pandas as pd, numpy as np
a = pd.read_csv('submission.csv', dtype={'uid': str}); r = pd.read_csv('retry.csv', dtype={'uid': str})
f = pd.read_csv('$S/submission_format.csv', dtype={'uid': str})
ok = list(a.columns) == list(f.columns) and bool((a.uid == f.uid).all()) and bool(a.is_pathologic.between(0, 1).all()) and int(a.is_pathologic.isna().sum()) == 0
same = float(np.abs(a.is_pathologic - r.is_pathologic).max()) == 0.0
print('[$NAME] format ok', ok, '| retry identical', same)
assert ok and same"
cd "$ROOT" && ${PYTHON:-python} -W ignore scripts/parity.py "$DIR/assets" "$DIR/submission.csv" | sed "s/^/[$NAME] /"
echo "[$NAME] ALL CHECKS PASSED"
