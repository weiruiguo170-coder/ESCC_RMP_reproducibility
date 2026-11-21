from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
HERE = Path(__file__).resolve().parent
ROOT = INPUT_ROOT
OUT = OUTPUT_ROOT / 'simulation'
R1 = Path('FIG_EDSC_Methods_Research_20261118')
R2 = Path('FIG_EDSC_Measurement_Benchmark_R2_20261118')
V2 = Path('FIG_TaskABC_Integrated_Final_Revision_20261117')
S8 = V2 / 'Supplementary_Tables/Tables/S8_PDO_exact_results.tsv'
ALPHA = 0.05

def sha(path):
    with path.open('rb') as handle:
        h = hashlib.sha256()
        for part in iter(lambda : handle.read(2 ** 20), b''):
            h.update(part)
        return h.hexdigest()

def timestamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def write(data, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    comp = {'method': 'gzip', 'mtime': 0} if path.suffix == '.gz' else None
    pd.DataFrame(data).to_csv(path, sep='\t', index=False, float_format='%.15g')

class Audit:
    """Resolve source inputs without retaining execution-session artifacts."""

    def __init__(self, stage):
        self.stage = stage

    def source(self, name):
        return resolve_data(INPUT_ROOT / name)

    def path(self, name):
        return self.source(name)

    def table(self, name, **options):
        return pd.read_csv(self.source(name), sep='\t', **options)

    def read(self, name, **options):
        return self.table(name, **options)

    def result(self, name, **options):
        generated = OUT / name
        return pd.read_csv(generated, sep='\t', **options) if generated.exists() else self.table(name, **options)

    def finish(self, **details):
        print(self.stage, details, flush=True)

def tail_p(observed, draws):
    draws = np.asarray(draws)
    ok = np.isfinite(draws)
    return (1 + (draws[ok] >= observed - 1e-12).sum()) / (ok.sum() + 1) if np.isfinite(observed) and ok.any() else np.nan

def supported_effect(n, N, average, lower, q):
    if not all((np.isfinite(x) for x in [n, N, average, lower, q])) or n < max(5, np.ceil(N / 2)):
        return None
    return bool(average > 0 and lower > 0 and (q < ALPHA))

def pc_test(p, N, u, method='bonferroni'):
    p = np.asarray(p, dtype=float)
    assert len(p) <= N and 1 <= u <= N
    good = p[np.isfinite(p)]
    assert ((good >= 0) & (good <= 1)).all()
    if len(good) < max(5, np.ceil(N / 2), u):
        return np.nan
    ordered = np.sort(np.pad(good, (0, N - len(good)), constant_values=1.0))
    if method == 'fisher':
        subset = ordered[u - 1:]
        return chi2.sf(-2 * np.log(subset).sum(), 2 * len(subset))
    assert method == 'bonferroni'
    return min(1.0, (N - u + 1) * ordered[u - 1])

def holm(values):
    p = np.asarray(values, dtype=float)
    assert len(p) == 7
    adjusted = multipletests(np.nan_to_num(p, nan=1.0), method='holm')[1]
    return np.where(np.isfinite(p), adjusted, np.nan)

def enough_resolution(total, target, draws, family=1):
    return family * (total - target + 1) / (draws + 1) <= ALPHA

def code(decision):
    return 'NOT_EVALUABLE' if decision is None else 'SUPPORTED_WITHIN_SCOPE' if decision else 'NOT_ESTABLISHED'
