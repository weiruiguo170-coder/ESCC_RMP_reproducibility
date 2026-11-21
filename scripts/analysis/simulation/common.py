from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import hashlib
import json
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
HERE = Path(__file__).resolve().parent
ROOT = INPUT_ROOT
OUT = OUTPUT_ROOT / 'simulation'
PREVIOUS = Path('FIG_EDSC_Methods_Research_20261118')
COHERENCE = Path('FIG_REVISION_ROUND2/01_external_coherence')
COHORTS = ['GSE199654', 'GSE203067']
RMPS = [f'RMP_C{i:02d}' for i in range(1, 8)]

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(2 ** 20), b''):
            digest.update(block)
    return digest.hexdigest()

def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')

def save(data, filename):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / filename
    compress = {'method': 'gzip', 'mtime': 0} if path.suffix == '.gz' else None
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

def signatures(audit):
    output = {}
    for rmp in RMPS:
        for n in [50, 100]:
            path = audit.source(f'06_rmp_annotation/{rmp}_signature_top{n}.txt')
            members = path.read_text(encoding='utf-8').splitlines()
            assert len(members) == len(set(members)) == n
            output[f'{rmp}_TOP{n}'] = members
    return output

def load_counts(audit, cohort):
    data = anndata.read_h5ad(audit.source(f'05_phase_E_external_validation/03_processed_external_data/{cohort}_malignant_ready.h5ad'))
    x = sparse.csr_matrix(data.X)
    assert x.nnz and (x.data >= 0).all() and np.equal(x.data, np.floor(x.data)).all()
    assert data.var_names.is_unique
    return (x, data.var_names.to_numpy(), data.obs.patient_id.astype(str))

def legacy_seed(key):
    return (2026111402 + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)) % 2 ** 32

def rng(*components):
    return np.random.default_rng(np.random.SeedSequence(components))

def standardize(x):
    centered = x - x.mean(axis=0)
    sd = centered.std(axis=0)
    return centered / np.where(sd > 1e-12, sd, np.nan)

def regress(z, design):
    return z - design @ (design.T @ z)

def orthogonal_depth(depth, degree):
    x = standardize(np.log1p(depth))
    polynomial = np.column_stack([np.ones(len(x)), *[x ** d for d in range(1, degree + 1)]])
    return np.linalg.qr(polynomial, mode='reduced')[0][:, 1:]

def rank_expression(counts, depth):
    x = np.log1p(counts / depth[:, None] * 10000.0)
    return (standardize(rankdata(x, axis=0)), x)

def composition(z):
    return z.T @ z / len(z)

def poisson_moment_correlation(counts, exposure):
    relative = exposure / np.mean(exposure)
    normalized = counts / relative[:, None]
    centered = normalized - normalized.mean(axis=0)
    observed_cov = centered.T @ centered / (len(counts) - 1)
    corrected = np.diag(observed_cov) - (counts / relative[:, None] ** 2).mean(axis=0)
    sd = np.sqrt(np.where(corrected > 1e-12, corrected, np.nan))
    corr = observed_cov / np.outer(sd, sd)
    return (corr, corrected)

def edge_values(matrix, subsets):
    (a, b) = np.triu_indices(subsets.shape[-1], 1)
    return matrix[subsets[..., a], subsets[..., b]]

def mean_edges(matrix, sets):
    edges = edge_values(matrix, sets)
    valid = np.isfinite(edges)
    n = valid.sum(axis=-1)
    return (np.divide(np.nansum(edges, axis=-1), n, out=np.full(n.shape, np.nan), where=n > 0), n / edges.shape[-1])

def vector_pair_average(values):
    k = values.shape[-1]
    return (values.sum(axis=-1) ** 2 - np.square(values).sum(axis=-1)) / (k * (k - 1))

def bh(p):
    output = np.full(len(p), np.nan)
    eligible = np.flatnonzero(np.isfinite(p))
    sort = eligible[np.argsort(p[eligible])]
    ranks = np.arange(1, len(sort) + 1)
    output[sort] = np.minimum(1.0, np.minimum.accumulate((p[sort] * len(p) / ranks)[::-1])[::-1])
    return output
