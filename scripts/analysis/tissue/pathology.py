from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import gzip
import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
import statsmodels
from matplotlib.gridspec import GridSpec
from scipy import sparse
from scipy.stats import rankdata, spearmanr, wilcoxon
PROJECT_ROOT = INPUT_ROOT
RESULTS_DIR = PROJECT_ROOT / 'results'
OUT = RESULTS_DIR / 'pathological_states'
PHASE_C = RESULTS_DIR / 'cross_cohort_matching'
PHASE_D = PROJECT_ROOT / 'signatures'
PHASE_E = RESULTS_DIR / 'external_reproducibility'
PHASE_F = RESULTS_DIR / 'bulk_context'
RAW = PROJECT_ROOT / 'data' / 'raw' / 'GSE199654'
SEED = 20260827
N_PERMUTATIONS = 10000
N_BOOTSTRAPS = 10000
RMP_IDS = [f'RMP_C{i:02d}' for i in range(1, 8)]
PRIMARY = ['RMP_C01', 'RMP_C02', 'RMP_C04', 'RMP_C05', 'RMP_C07']
EXPLORATORY = ['RMP_C03', 'RMP_C06']
STATE_ORDER = ['NON_TUMOR', 'DYSPLASIA', 'TUMOR']
STATE_CODE = {state: i for (i, state) in enumerate(STATE_ORDER)}
DIRS = {'audit': OUT / '00_audit', 'metadata': OUT / '01_metadata', 'scores': OUT / '02_scores', 'coverage': OUT / '03_signature_coverage', 'stats': OUT / '04_statistics', 'sensitivity': OUT / '05_sensitivity_top100', 'figures': OUT / '06_figures', 'tables': OUT / '07_tables', 'reports': OUT / '08_reports', 'logs': OUT / '09_logs'}
for directory in DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def normpath(path: Path | str) -> str:
    return str(Path(path).resolve()).replace('\\', '/').lower()

def bh_fdr(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    out = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    pv = p[finite]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    tmp = np.empty(n)
    tmp[order] = np.minimum(adjusted, 1.0)
    out[finite] = tmp
    return out

def load_hash_lookups() -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    e_audit = pd.read_csv(resolve_data(PHASE_E / '00_audit' / 'PHASE_E_INPUT_AUDIT.tsv'), sep='\t')
    for row in e_audit.itertuples(index=False):
        lookup[normpath(row.file)] = (str(row.expected_sha256), str(row.source_phase))
    e_sha = pd.read_csv(resolve_data(PHASE_E / 'PHASE_E_SHA256.tsv'), sep='\t')
    for row in e_sha.itertuples(index=False):
        lookup[normpath(PHASE_E / row.relative_path)] = (str(row.sha256), 'PHASE_E')
    f_sha = pd.read_csv(resolve_data(PHASE_F / 'PHASE_F_SHA256.tsv'), sep='\t')
    for row in f_sha.itertuples(index=False):
        lookup[normpath(PHASE_F / row.file)] = (str(row.sha256), 'PHASE_F')
    downloads = pd.read_csv(resolve_data(PHASE_E / '02_raw_external_data' / 'downloaded_source_checksums.tsv'), sep='\t')
    for row in downloads.itertuples(index=False):
        lookup[normpath(row.file)] = (str(row.sha256), 'PHASE_E_SOURCE')
    return lookup

def frozen_input_audit() -> pd.DataFrame:
    lookup = load_hash_lookups()
    required: list[tuple[Path, str, str]] = []
    for rmp in RMP_IDS:
        required.extend([(PHASE_D / f'{rmp}_signature_top50.txt', 'PHASE_D', f'{rmp} frozen Top50'), (PHASE_D / f'{rmp}_signature_top100.txt', 'PHASE_D', f'{rmp} frozen Top100')])
    required.extend([(PHASE_C / 'PHASE_C_REPORT.md', 'PHASE_C', 'Phase C report'), (PHASE_C / 'RMP_evidence_table.tsv', 'PHASE_C', 'Frozen RMP membership'), (PHASE_C / 'output_manifest.tsv', 'PHASE_C', 'Phase C manifest'), (PHASE_D / 'PHASE_D_REPORT.md', 'PHASE_D', 'Phase D report'), (PHASE_D / 'rmp_annotation_table.tsv', 'PHASE_D', 'Frozen annotations'), (PHASE_D / 'rmp_frozen_signatures.tsv', 'PHASE_D', 'Frozen signature table'), (PHASE_D / 'output_manifest.tsv', 'PHASE_D', 'Phase D manifest'), (PHASE_E / '11_reports' / 'PHASE_E_REPORT.md', 'PHASE_E', 'Phase E report'), (PHASE_E / '10_tables' / 'overall_rmp_external_validation.tsv', 'PHASE_E', 'Frozen Phase E support labels'), (PHASE_E / 'exact_configuration.json', 'PHASE_E', 'Phase E scoring specification'), (PHASE_E / 'PHASE_E_MANIFEST.tsv', 'PHASE_E', 'Phase E manifest'), (PHASE_E / 'PHASE_E_SHA256.tsv', 'PHASE_E', 'Phase E SHA256 ledger'), (PHASE_F / '11_reports' / 'PHASE_F_REPORT.md', 'PHASE_F', 'Phase F report'), (PHASE_F / '10_tables' / 'Table_F1_bulk_contextualization_summary.tsv', 'PHASE_F', 'Phase F support/context labels'), (PHASE_F / 'PHASE_F_MANIFEST.tsv', 'PHASE_F', 'Phase F manifest'), (PHASE_F / 'PHASE_F_SHA256.tsv', 'PHASE_F', 'Phase F SHA256 ledger'), (RAW / 'GSE199654_scTDN_UMI_matrix_epithelial_cells.txt.gz', 'PHASE_E_SOURCE', 'GSE199654 raw UMI matrix'), (RAW / 'GSE199654_scTDN_METADATA_epithelial_cells.txt.gz', 'PHASE_E_SOURCE', 'GSE199654 official cell metadata'), (RAW / 'GSE199654_scTDN_barcode_epithelial_cells.txt.gz', 'PHASE_E_SOURCE', 'GSE199654 barcode file'), (RAW / 'GSE199654_scTDN_feature_epithelial_cells.txt.gz', 'PHASE_E_SOURCE', 'GSE199654 feature file')])
    rows = []
    for (path, source, role) in required:
        observed = sha256(path) if path.exists() else 'MISSING'
        expected_info = lookup.get(normpath(path))
        if expected_info is None:
            expected = observed if observed != 'MISSING' else 'NOT_AVAILABLE'
            basis = 'BASELINE_FROZEN_RECORD'
        else:
            (expected, basis) = expected_info
        status = 'PASS' if observed != 'MISSING' and observed.lower() == expected.lower() else 'FAIL'
        rows.append({'file': str(path.resolve()), 'expected_sha256': expected, 'observed_sha256': observed, 'status': status, 'source_phase': source, 'audit_basis': basis, 'role': role})
    audit = pd.DataFrame(rows)
    audit.to_csv(DIRS['audit'] / 'PHASE_G0_INPUT_AUDIT.tsv', sep='\t', index=False)
    audit[['file', 'expected_sha256', 'observed_sha256', 'status', 'source_phase']].to_csv(DIRS['audit'] / 'PHASE_G0_SHA256_COMPARISON.tsv', sep='\t', index=False)
    for rmp in RMP_IDS:
        top50 = [x.strip() for x in (PHASE_D / f'{rmp}_signature_top50.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
        top100 = [x.strip() for x in (PHASE_D / f'{rmp}_signature_top100.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
        if len(top50) != 50 or len(set(top50)) != 50 or len(top100) != 100 or (len(set(top100)) != 100):
            raise RuntimeError(f'Frozen signature membership/count failure: {rmp}')
        if top100[:50] != top50:
            raise RuntimeError(f'Top50 is not the frozen Top100 prefix: {rmp}')
    evidence = pd.read_csv(resolve_data(PHASE_C / 'RMP_evidence_table.tsv'), sep='\t')
    if evidence.rmp_id.tolist() != RMP_IDS:
        raise RuntimeError('RMP membership/order mismatch')
    phase_e_support = pd.read_csv(resolve_data(PHASE_E / '10_tables' / 'overall_rmp_external_validation.tsv'), sep='\t')
    expected_support = {'RMP_C01': 'MODERATE', 'RMP_C02': 'MODERATE', 'RMP_C03': 'MODERATE', 'RMP_C04': 'LIMITED', 'RMP_C05': 'MODERATE', 'RMP_C06': 'STRONG', 'RMP_C07': 'STRONG'}
    observed_support = dict(zip(phase_e_support.rmp_id, phase_e_support.overall_interpretation))
    if observed_support != expected_support:
        raise RuntimeError(f'Phase E support-label mismatch: {observed_support}')
    if not audit.status.eq('PASS').all():
        failed = audit.loc[audit.status.ne('PASS'), ['file', 'expected_sha256', 'observed_sha256']]
        raise RuntimeError('Frozen-input SHA256 audit failed:\n' + failed.to_string(index=False))
    summary = {'phase': 'G0', 'audit_date': '2026-08-27', 'files_audited': int(len(audit)), 'files_passed': int(audit.status.eq('PASS').sum()), 'all_hashes_pass': True, 'top50_role': 'PRIMARY', 'top100_role': 'SENSITIVITY_ONLY', 'frozen_signatures': 'UNCHANGED', 'rmp_membership': 'UNCHANGED', 'phase_d_annotations': 'UNCHANGED', 'phase_e_support_labels': 'UNCHANGED', 'phase_f_results': 'UNCHANGED', 'all_required_inputs': 'PASS'}
    (DIRS['audit'] / 'PHASE_G0_INPUT_AUDIT_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return audit

def canonical(symbol: object) -> str:
    return str(symbol).strip().upper()

def collapse_duplicate_genes(matrix: sparse.csr_matrix, genes: pd.Index) -> tuple[sparse.csr_matrix, pd.Index]:
    symbols = pd.Index([canonical(x) for x in genes])
    keep = (symbols != '') & (symbols != 'NAN')
    matrix = matrix[:, keep]
    symbols = symbols[keep]
    unique = pd.Index(pd.unique(symbols))
    codes = unique.get_indexer(symbols)
    aggregator = sparse.csr_matrix((np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)), shape=(len(codes), len(unique)))
    return ((matrix @ aggregator).tocsr(), unique)

def read_gse199654_all_epithelial() -> tuple[ad.AnnData, pd.DataFrame]:
    matrix_path = RAW / 'GSE199654_scTDN_UMI_matrix_epithelial_cells.txt.gz'
    metadata_path = RAW / 'GSE199654_scTDN_METADATA_epithelial_cells.txt.gz'
    with gzip.open(matrix_path, 'rt') as handle:
        barcodes = handle.readline().rstrip('\n\r').split('\t')
    frame = pd.read_csv(resolve_data(matrix_path), sep='\t', header=None, skiprows=1, index_col=0, names=['gene'] + barcodes).astype(np.float32)
    frame.columns = frame.columns.str.replace('\\.1$', '-1', regex=True)
    matrix = sparse.csr_matrix(frame.to_numpy().T)
    (matrix, genes) = collapse_duplicate_genes(matrix, frame.index.astype(str))
    obs = pd.read_csv(resolve_data(metadata_path), sep='\t').set_index('cellID')
    if obs.index.has_duplicates:
        raise RuntimeError('GSE199654 metadata has duplicate cellID')
    missing = pd.Index(frame.columns).difference(obs.index)
    extra = obs.index.difference(pd.Index(frame.columns))
    if len(missing) or len(extra):
        raise RuntimeError(f'Expression/metadata mismatch: matrix_missing={len(missing)}, metadata_extra={len(extra)}')
    obs = obs.loc[frame.columns].copy()
    n_counts = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = np.diff(matrix.indptr)
    mt_mask = genes.str.startswith('MT-')
    mt_counts = np.asarray(matrix[:, mt_mask].sum(axis=1)).ravel() if mt_mask.any() else np.zeros(matrix.shape[0])
    pct_mito = np.divide(mt_counts, n_counts, out=np.zeros_like(n_counts, dtype=float), where=n_counts > 0) * 100
    obs['n_counts'] = n_counts
    obs['n_genes'] = n_genes
    obs['pct_mito'] = pct_mito
    obs['qc_pass'] = obs.n_genes.ge(200) & obs.n_genes.lt(6000) & obs.pct_mito.lt(5.0)

    def map_state(stage: str) -> str:
        stage = str(stage).strip()
        if stage in {'N', 'INF'}:
            return 'NON_TUMOR'
        if stage in {'LGIN', 'HGIN'}:
            return 'DYSPLASIA'
        if stage.startswith('Stage_'):
            return 'TUMOR'
        raise RuntimeError(f'Unrecognized official pathology stage: {stage}')
    obs['normalized_primary_state'] = obs.stage.map(map_state)
    obs['ordered_state'] = obs.normalized_primary_state.map(STATE_CODE).astype(int)
    obs['patient_id'] = obs.patientID.astype(str)
    obs['sample_id'] = obs.sampleID.astype(str)
    obs['original_pathology_label'] = obs.stage.astype(str)
    obs['original_sample_label'] = obs.sampleID.astype(str)
    adata = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))
    adata.var_names.name = 'gene_symbol'
    unique_samples = obs[['sample_id', 'normalized_primary_state']].drop_duplicates()
    actual = unique_samples.normalized_primary_state.value_counts().to_dict()
    expected = {'NON_TUMOR': 18, 'DYSPLASIA': 15, 'TUMOR': 14}
    if actual != expected:
        raise RuntimeError(f'Official GEO 18/15/14 state structure mismatch: {actual}')
    return (adata, obs)

def build_metadata_outputs(adata: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = adata.obs.copy()
    rows = []
    qc_rows = []
    for ((patient, sample), group) in obs.groupby(['patient_id', 'sample_id'], sort=True):
        states = group.normalized_primary_state.unique()
        stages = group.original_pathology_label.unique()
        if len(states) != 1 or len(stages) != 1:
            raise RuntimeError(f'Non-unique state/pathology within sample: {patient}/{sample}')
        retained = group[group.qc_pass]
        rows.append({'patient_id': patient, 'sample_id': sample, 'original_sample_label': group.original_sample_label.iloc[0], 'original_pathology_label': stages[0], 'normalized_primary_state': states[0], 'number_of_epithelial_cells_before_QC': int(len(group)), 'number_of_epithelial_cells_after_QC': int(len(retained)), 'included_primary_analysis': 'YES' if len(retained) > 0 else 'NO', 'exclusion_reason': '' if len(retained) > 0 else 'NO_EPITHELIAL_CELLS_AFTER_FROZEN_QC'})
        qc_rows.append({'patient_id': patient, 'sample_id': sample, 'normalized_primary_state': states[0], 'original_pathology_label': stages[0], 'total_epithelial_cells': int(len(group)), 'retained_epithelial_cells': int(len(retained)), 'median_genes_per_cell': float(retained.n_genes.median()) if len(retained) else np.nan, 'median_UMI_per_cell': float(retained.n_counts.median()) if len(retained) else np.nan, 'median_mitochondrial_fraction_percent': float(retained.pct_mito.median()) if len(retained) else np.nan, 'frozen_QC': '200<=detected_genes<6000; mitochondrial_UMI_fraction<5%'})
    crosswalk = pd.DataFrame(rows).sort_values(['patient_id', 'normalized_primary_state', 'sample_id'])
    qc = pd.DataFrame(qc_rows).sort_values(['patient_id', 'normalized_primary_state', 'sample_id'])
    crosswalk.to_csv(DIRS['metadata'] / 'GSE199654_state_crosswalk.tsv', sep='\t', index=False)
    qc.to_csv(DIRS['metadata'] / 'GSE199654_patient_sample_state_QC.tsv', sep='\t', index=False)
    retained_obs = obs[obs.qc_pass]
    char_rows = []
    for state in STATE_ORDER:
        all_state = obs[obs.normalized_primary_state.eq(state)]
        kept = retained_obs[retained_obs.normalized_primary_state.eq(state)]
        char_rows.append({'cohort': 'GSE199654', 'state': state, 'state_order': STATE_CODE[state], 'patients_before_QC': int(all_state.patient_id.nunique()), 'patients_evaluable_after_QC': int(kept.patient_id.nunique()), 'samples_before_QC': int(all_state.sample_id.nunique()), 'samples_evaluable_after_QC': int(kept.sample_id.nunique()), 'patients': int(kept.patient_id.nunique()), 'samples': int(kept.sample_id.nunique()), 'epithelial_cells_before_QC': int(len(all_state)), 'epithelial_cells_after_QC': int(len(kept)), 'median_genes_per_retained_cell': float(kept.n_genes.median()), 'median_UMI_per_retained_cell': float(kept.n_counts.median()), 'median_mitochondrial_fraction_percent': float(kept.pct_mito.median()), 'official_GEO_samples_expected': {'NON_TUMOR': 18, 'DYSPLASIA': 15, 'TUMOR': 14}[state]})
    characteristics = pd.DataFrame(char_rows)
    characteristics.to_csv(DIRS['tables'] / 'Table_G0_1_cohort_state_characteristics.tsv', sep='\t', index=False)
    return (crosswalk, qc, characteristics)

def load_signatures() -> dict[str, list[str]]:
    signatures: dict[str, list[str]] = {}
    for rmp in RMP_IDS:
        for size in (50, 100):
            signatures[f'{rmp}_TOP{size}'] = [x.strip() for x in (PHASE_D / f'{rmp}_signature_top{size}.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
    return signatures

def signature_coverage(adata: ad.AnnData, signatures: dict[str, list[str]]) -> pd.DataFrame:
    qc = adata.obs.qc_pass.to_numpy()
    matrix = sparse.csr_matrix(adata.X[qc])
    detected_overall = np.asarray((matrix > 0).sum(axis=0)).ravel() > 0
    gene_to_index = {str(g): i for (i, g) in enumerate(adata.var_names)}
    rows = []
    for (key, genes) in signatures.items():
        (rmp, sigtype) = key.rsplit('_', 1)
        present = [g for g in genes if g in gene_to_index and detected_overall[gene_to_index[g]]]
        missing = [g for g in genes if g not in set(present)]
        row = {'cohort': 'GSE199654_G0_ALL_QC_EPITHELIAL', 'rmp_id': rmp, 'signature_type': sigtype, 'total_frozen_genes': len(genes), 'genes_detected': len(present), 'genes_missing': len(missing), 'coverage_percent': 100.0 * len(present) / len(genes), 'coverage_label': 'GOOD' if len(present) / len(genes) >= 0.8 else 'ACCEPTABLE' if len(present) / len(genes) >= 0.7 else 'CAUTION' if len(present) / len(genes) >= 0.6 else 'LOW_COVERAGE', 'detected_genes': ';'.join(present), 'missing_genes': ';'.join(missing), 'scoring_membership_rule': 'covered across all QC-passed epithelial cells; fixed across pathology states'}
        for state in STATE_ORDER:
            state_mask = qc & adata.obs.normalized_primary_state.eq(state).to_numpy()
            state_detected = np.asarray((adata.X[state_mask] > 0).sum(axis=0)).ravel() > 0
            row[f'genes_detected_in_{state}'] = sum((g in gene_to_index and state_detected[gene_to_index[g]] for g in genes))
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(DIRS['coverage'] / 'GSE199654_G0_signature_coverage.tsv', sep='\t', index=False)
    primary_top50 = table[table.rmp_id.isin(PRIMARY) & table.signature_type.eq('TOP50')]
    if (primary_top50.coverage_percent < 60).any():
        raise RuntimeError('Primary Top50 coverage gate failed')
    return table

def cell_rank_scores(matrix: sparse.spmatrix, var_names: pd.Index, signatures: dict[str, list[str]]) -> pd.DataFrame:
    matrix = sparse.csr_matrix(matrix)
    n_genes = matrix.shape[1]
    detected_global = np.asarray((matrix > 0).sum(axis=0)).ravel() > 0
    name_to_index = {str(g): i for (i, g) in enumerate(var_names)}
    signature_indices: dict[str, np.ndarray] = {}
    for (key, genes) in signatures.items():
        signature_indices[key] = np.asarray([name_to_index[g] for g in genes if g in name_to_index and detected_global[name_to_index[g]]], dtype=int)
    scores = np.zeros((matrix.shape[0], len(signatures)), dtype=np.float32)
    dense_ranks = np.empty(n_genes, dtype=np.float64)
    for i in range(matrix.shape[0]):
        (start, end) = (matrix.indptr[i], matrix.indptr[i + 1])
        indices = matrix.indices[start:end]
        values = matrix.data[start:end]
        nnz = len(indices)
        zero_count = n_genes - nnz
        zero_rank = (zero_count + 1.0) / 2.0 if zero_count else 0.0
        dense_ranks.fill(zero_rank)
        if nnz:
            dense_ranks[indices] = zero_count + rankdata(values, method='average')
        for (j, key) in enumerate(signatures):
            idx = signature_indices[key]
            m = len(idx)
            if m == 0 or n_genes <= m:
                scores[i, j] = np.nan
            else:
                minimum = (m + 1.0) / 2.0
                scores[i, j] = np.clip((dense_ranks[idx].mean() - minimum) / (n_genes - m), 0, 1)
    return pd.DataFrame(scores, columns=list(signatures))

def score_and_aggregate(adata: ad.AnnData, signatures: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    kept = adata[adata.obs.qc_pass.to_numpy()].copy()
    scores = cell_rank_scores(kept.X, kept.var_names, signatures)
    meta_cols = ['patient_id', 'sample_id', 'original_sample_label', 'original_pathology_label', 'normalized_primary_state', 'ordered_state', 'n_genes', 'n_counts', 'pct_mito']
    cell_table = kept.obs[meta_cols].reset_index(names='cell_id').reset_index(drop=True)
    cell_table = pd.concat([cell_table, scores.reset_index(drop=True)], axis=1)
    cell_table.to_csv(DIRS['scores'] / 'cell_level_RMP_scores.tsv', sep='\t', index=False)
    long = cell_table.melt(id_vars=['cell_id', 'patient_id', 'sample_id', 'original_pathology_label', 'normalized_primary_state', 'ordered_state'], value_vars=list(signatures), var_name='rmp_signature', value_name='cell_score')
    parsed = long.rmp_signature.str.extract('^(RMP_C\\d{2})_(TOP\\d+)$')
    long['rmp_id'] = parsed[0]
    long['signature_type'] = parsed[1]
    sample = long.groupby(['patient_id', 'sample_id', 'original_pathology_label', 'normalized_primary_state', 'ordered_state', 'rmp_id', 'signature_type'], observed=True).agg(sample_median_score=('cell_score', 'median'), sample_mean_score=('cell_score', 'mean'), retained_cells=('cell_id', 'size')).reset_index()
    sample.to_csv(DIRS['scores'] / 'sample_level_RMP_scores.tsv', sep='\t', index=False)
    patient_state = sample.groupby(['patient_id', 'normalized_primary_state', 'ordered_state', 'rmp_id', 'signature_type'], observed=True).agg(patient_state_score=('sample_median_score', 'median'), patient_state_mean_of_sample_medians=('sample_median_score', 'mean'), samples_aggregated=('sample_id', 'nunique'), retained_cells=('retained_cells', 'sum')).reset_index()
    patient_state['aggregation_rule'] = 'unweighted median of sample-level median cell scores'
    patient_state.to_csv(DIRS['scores'] / 'patient_state_RMP_scores.tsv', sep='\t', index=False)
    return (cell_table, sample, patient_state)

def build_blocked_permutations(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    x = frame.ordered_state.to_numpy(dtype=float)
    groups = frame.patient_id.astype(str).to_numpy()
    permutations = np.tile(x, (N_PERMUTATIONS, 1))
    for patient in pd.unique(groups):
        idx = np.where(groups == patient)[0]
        if len(idx) > 1:
            base = x[idx]
            for b in range(N_PERMUTATIONS):
                permutations[b, idx] = rng.permutation(base)
    return permutations

def fixed_effect_fallback(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    yc = y.copy().astype(float)
    xc = x.copy().astype(float)
    for patient in pd.unique(groups):
        idx = groups == patient
        yc[idx] -= yc[idx].mean()
        xc[idx] -= xc[idx].mean()
    model = sm.OLS(yc, xc[:, None]).fit(cov_type='cluster', cov_kwds={'groups': groups})
    slope = float(model.params[0])
    se = float(model.bse[0])
    p = float(model.pvalues[0])
    return {'method': 'PATIENT_FIXED_EFFECT_WITH_CLUSTER_ROBUST_SE_FALLBACK', 'slope': slope, 'se': se, 'ci_low': slope - 1.959963984540054 * se, 'ci_high': slope + 1.959963984540054 * se, 'p': p, 'converged': False, 'random_intercept_variance': np.nan, 'residual_variance': float(model.mse_resid), 'warnings': 'MIXED_MODEL_FAILED; pre-specified patient-fixed-effect fallback used'}

def fit_trend(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    exog = sm.add_constant(x)
    caught: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            result = sm.MixedLM(y, exog, groups=groups).fit(reml=True, method='lbfgs', maxiter=1000, disp=False)
            caught = [str(w.message) for w in ws]
        if not bool(result.converged) or not np.all(np.isfinite(result.fe_params)):
            raise RuntimeError('MixedLM did not converge to finite fixed effects')
        slope = float(result.fe_params[1])
        se = float(result.bse_fe[1])
        p = float(result.pvalues[1])
        return {'method': 'RANDOM_INTERCEPT_MIXED_MODEL_REML', 'slope': slope, 'se': se, 'ci_low': slope - 1.959963984540054 * se, 'ci_high': slope + 1.959963984540054 * se, 'p': p, 'converged': True, 'random_intercept_variance': float(np.asarray(result.cov_re)[0, 0]), 'residual_variance': float(result.scale), 'warnings': ' | '.join(caught)}
    except Exception as exc:
        fallback = fixed_effect_fallback(y, x, groups)
        fallback['warnings'] = f"{fallback['warnings']} | original_error={exc} | original_warnings={' | '.join(caught)}"
        return fallback

def permutation_pvalue(y: np.ndarray, x: np.ndarray, groups: np.ndarray, permutations: np.ndarray, fit: dict[str, object]) -> tuple[float, float, np.ndarray]:
    if fit['method'] == 'RANDOM_INTERCEPT_MIXED_MODEL_REML':
        tau2 = max(float(fit['random_intercept_variance']), 0.0)
        sigma2 = max(float(fit['residual_variance']), 1e-12)
        n = len(y)
        W = np.zeros((n, n), dtype=float)
        for patient in pd.unique(groups):
            idx = np.where(groups == patient)[0]
            m = len(idx)
            block = np.eye(m) / sigma2 - tau2 / (sigma2 * (sigma2 + m * tau2)) * np.ones((m, m))
            W[np.ix_(idx, idx)] = block
        one = np.ones(n)
        w1 = W @ one
        wy = W @ y
        one_w_one = float(one @ w1)
        one_w_y = float(one @ wy)
        xp_w1 = permutations @ w1
        xp_wy = permutations @ wy
        xp_w_xp = np.sum(permutations @ W * permutations, axis=1)
        denom = xp_w_xp - xp_w1 ** 2 / one_w_one
        numer = xp_wy - xp_w1 * one_w_y / one_w_one
        beta_perm = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=np.abs(denom) > 1e-15)
        obs_numer = float(x @ wy - x @ w1 * one_w_y / one_w_one)
        obs_denom = float(x @ W @ x - (x @ w1) ** 2 / one_w_one)
        observed_stat = obs_numer / obs_denom
    else:
        yc = y.astype(float).copy()
        xc = x.astype(float).copy()
        pc = permutations.astype(float).copy()
        for patient in pd.unique(groups):
            idx = np.where(groups == patient)[0]
            yc[idx] -= yc[idx].mean()
            xc[idx] -= xc[idx].mean()
            pc[:, idx] -= pc[:, idx].mean(axis=1, keepdims=True)
        observed_stat = float(np.dot(xc, yc) / np.dot(xc, xc))
        denom = np.sum(pc ** 2, axis=1)
        beta_perm = np.divide(pc @ yc, denom, out=np.full(len(pc), np.nan), where=denom > 0)
    finite = np.isfinite(beta_perm)
    p = (np.sum(np.abs(beta_perm[finite]) >= abs(observed_stat) - 1e-15) + 1) / (finite.sum() + 1)
    return (float(observed_stat), float(p), beta_perm)

def trend_analyses(patient_state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    base = patient_state[['patient_id', 'normalized_primary_state', 'ordered_state']].drop_duplicates().sort_values(['patient_id', 'ordered_state']).reset_index(drop=True)
    permutations = build_blocked_permutations(base, rng)
    rows = []
    lopo_rows = []
    for sigtype in ('TOP50', 'TOP100'):
        for rmp in RMP_IDS:
            subset = patient_state[(patient_state.rmp_id == rmp) & (patient_state.signature_type == sigtype)].merge(base, on=['patient_id', 'normalized_primary_state', 'ordered_state'], how='right', validate='one_to_one')
            if subset.patient_state_score.isna().any():
                raise RuntimeError(f'Missing patient-state score after merge: {rmp}/{sigtype}')
            y = subset.patient_state_score.to_numpy(float)
            x = subset.ordered_state.to_numpy(float)
            groups = subset.patient_id.astype(str).to_numpy()
            fit = fit_trend(y, x, groups)
            (perm_stat, perm_p, _) = permutation_pvalue(y, x, groups, permutations, fit)
            current_lopo = []
            for left_out in pd.unique(groups):
                keep = groups != left_out
                lf = fit_trend(y[keep], x[keep], groups[keep])
                current_lopo.append(float(lf['slope']))
                lopo_rows.append({'rmp_id': rmp, 'signature_type': sigtype, 'left_out_patient': left_out, 'full_data_slope': fit['slope'], 'lopo_slope': lf['slope'], 'lopo_ci_low': lf['ci_low'], 'lopo_ci_high': lf['ci_high'], 'lopo_model_method': lf['method'], 'lopo_model_converged': lf['converged'], 'lopo_warnings': lf['warnings']})
            current_lopo = np.asarray(current_lopo)
            full_sign = np.sign(float(fit['slope']))
            sign_fraction = float(np.mean(np.sign(current_lopo) == full_sign)) if full_sign != 0 else np.nan
            major_instability = bool(abs(float(fit['slope'])) >= float(fit['se']) and sign_fraction < 0.8)
            rows.append({'rmp_id': rmp, 'signature_type': sigtype, 'tier': 'TIER_1_PRIMARY' if rmp in PRIMARY else 'TIER_2_EXPLORATORY', 'n_patients': int(pd.Series(groups).nunique()), 'n_patient_state_observations': int(len(y)), 'slope': fit['slope'], 'standard_error': fit['se'], 'CI_low': fit['ci_low'], 'CI_high': fit['ci_high'], 'mixed_model_P': fit['p'], 'permutation_statistic_slope': perm_stat, 'permutation_P': perm_p, 'direction': 'INCREASE' if fit['slope'] > 0 else 'DECREASE' if fit['slope'] < 0 else 'NO_DIRECTION', 'model_method': fit['method'], 'model_converged': fit['converged'], 'random_intercept_variance': fit['random_intercept_variance'], 'residual_variance': fit['residual_variance'], 'model_warnings': fit['warnings'], 'minimum_LOPO_slope': float(np.min(current_lopo)), 'maximum_LOPO_slope': float(np.max(current_lopo)), 'LOPO_sign_fraction': sign_fraction, 'one_patient_determines_direction': 'YES' if sign_fraction < 1.0 else 'NO', 'major_patient_instability': 'YES' if major_instability else 'NO', 'permutation_seed': SEED, 'permutation_count': N_PERMUTATIONS})
    trends = pd.DataFrame(rows)
    for sigtype in ('TOP50', 'TOP100'):
        primary_mask = trends.signature_type.eq(sigtype) & trends.rmp_id.isin(PRIMARY)
        trends.loc[primary_mask, 'BH_FDR'] = bh_fdr(trends.loc[primary_mask, 'permutation_P'])
        trends.loc[primary_mask, 'mixed_model_BH_FDR'] = bh_fdr(trends.loc[primary_mask, 'mixed_model_P'])
        exploratory_mask = trends.signature_type.eq(sigtype) & trends.rmp_id.isin(EXPLORATORY)
        trends.loc[exploratory_mask, 'BH_FDR'] = bh_fdr(trends.loc[exploratory_mask, 'permutation_P'])
        trends.loc[exploratory_mask, 'mixed_model_BH_FDR'] = bh_fdr(trends.loc[exploratory_mask, 'mixed_model_P'])

    def interpretation(row: pd.Series) -> str:
        if row.major_patient_instability == 'YES':
            return 'UNSTABLE'
        if row.BH_FDR < 0.05 and row.CI_low * row.CI_high > 0 and (row.LOPO_sign_fraction >= 0.8):
            return 'SUPPORTED'
        if (row.permutation_P < 0.05 or row.mixed_model_P < 0.05) and row.LOPO_sign_fraction >= 0.8:
            return 'LIMITED'
        return 'NOT_SUPPORTED'
    trends['interpretation'] = trends.apply(interpretation, axis=1)
    lopo = pd.DataFrame(lopo_rows)
    lopo_summary = trends[['rmp_id', 'signature_type', 'slope', 'minimum_LOPO_slope', 'maximum_LOPO_slope', 'LOPO_sign_fraction', 'one_patient_determines_direction', 'major_patient_instability']].copy()
    trends.to_csv(DIRS['stats'] / 'patient_aware_ordered_state_trends.tsv', sep='\t', index=False)
    lopo.to_csv(DIRS['stats'] / 'leave_one_patient_out_trends.tsv', sep='\t', index=False)
    lopo_summary.to_csv(DIRS['stats'] / 'LOPO_stability_summary.tsv', sep='\t', index=False)
    return (trends, lopo, lopo_summary)

def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    medians = np.median(rng.choice(values, size=(N_BOOTSTRAPS, len(values)), replace=True), axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))

def paired_transitions(patient_state: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    transitions = [('DYSPLASIA_MINUS_NON_TUMOR', 'DYSPLASIA', 'NON_TUMOR'), ('TUMOR_MINUS_DYSPLASIA', 'TUMOR', 'DYSPLASIA'), ('TUMOR_MINUS_NON_TUMOR', 'TUMOR', 'NON_TUMOR')]
    rows = []
    for sigtype in ('TOP50', 'TOP100'):
        for rmp in RMP_IDS:
            subset = patient_state[(patient_state.rmp_id == rmp) & (patient_state.signature_type == sigtype)]
            pivot = subset.pivot(index='patient_id', columns='normalized_primary_state', values='patient_state_score')
            for (name, later, earlier) in transitions:
                paired = pivot[[earlier, later]].dropna()
                delta = (paired[later] - paired[earlier]).to_numpy(float)
                median = float(np.median(delta)) if len(delta) else np.nan
                (lo, hi) = bootstrap_median_ci(delta, rng)
                if len(delta) and (not np.allclose(delta, 0)):
                    p = float(wilcoxon(delta, alternative='two-sided', zero_method='wilcox').pvalue)
                else:
                    p = 1.0 if len(delta) else np.nan
                direction = np.sign(median)
                direction_fraction = float(np.mean(np.sign(delta) == direction)) if len(delta) and direction != 0 else np.nan
                rows.append({'rmp_id': rmp, 'signature_type': sigtype, 'tier': 'TIER_1_PRIMARY' if rmp in PRIMARY else 'TIER_2_EXPLORATORY', 'transition': name, 'earlier_state': earlier, 'later_state': later, 'paired_patient_count': int(len(delta)), 'paired_patients': ';'.join(paired.index.astype(str)), 'median_paired_delta': median, 'bootstrap_CI_low': lo, 'bootstrap_CI_high': hi, 'direction': 'INCREASE' if median > 0 else 'DECREASE' if median < 0 else 'NO_CHANGE', 'direction_consistency_fraction': direction_fraction, 'paired_Wilcoxon_P_secondary': p, 'bootstrap_count': N_BOOTSTRAPS, 'seed': SEED})
    result = pd.DataFrame(rows)
    for sigtype in ('TOP50', 'TOP100'):
        mask = result.signature_type.eq(sigtype)
        result.loc[mask, 'paired_Wilcoxon_BH_FDR_secondary'] = bh_fdr(result.loc[mask, 'paired_Wilcoxon_P_secondary'])
    result.to_csv(DIRS['tables'] / 'Table_G0_3_paired_state_transitions.tsv', sep='\t', index=False)
    return result

def top100_sensitivity(patient_state: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rmp in RMP_IDS:
        t50 = trends[(trends.rmp_id == rmp) & (trends.signature_type == 'TOP50')].iloc[0]
        t100 = trends[(trends.rmp_id == rmp) & (trends.signature_type == 'TOP100')].iloc[0]
        pivot = patient_state[patient_state.rmp_id.eq(rmp)].pivot_table(index=['patient_id', 'normalized_primary_state'], columns='signature_type', values='patient_state_score').dropna()
        rho = float(spearmanr(pivot.TOP50, pivot.TOP100).statistic) if len(pivot) >= 3 else np.nan
        sign_consistent = np.sign(t50.slope) == np.sign(t100.slope)
        support50 = t50.BH_FDR < 0.05
        support100 = t100.BH_FDR < 0.05
        major = 'YES' if sign_consistent and support50 == support100 else 'PARTIAL' if sign_consistent else 'NO'
        rows.append({'rmp_id': rmp, 'tier': t50.tier, 'Top50_ordered_state_slope': t50.slope, 'Top50_permutation_P': t50.permutation_P, 'Top50_BH_FDR': t50.BH_FDR, 'Top100_ordered_state_slope': t100.slope, 'Top100_permutation_P': t100.permutation_P, 'Top100_BH_FDR': t100.BH_FDR, 'sign_consistency': 'YES' if sign_consistent else 'NO', 'patient_state_score_spearman_rho': rho, 'major_interpretation_consistency': major, 'size_sensitivity_warning': 'SIZE_SENSITIVITY_WARNING' if not sign_consistent else ''})
    result = pd.DataFrame(rows)
    result.to_csv(DIRS['tables'] / 'Table_G0_4_Top50_Top100_sensitivity.tsv', sep='\t', index=False)
    return result

def c02_c07_comparison(patient_state: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    top50 = patient_state[patient_state.signature_type.eq('TOP50') & patient_state.rmp_id.isin(['RMP_C02', 'RMP_C07'])]
    wide = top50.pivot_table(index=['patient_id', 'normalized_primary_state', 'ordered_state'], columns='rmp_id', values='patient_state_score').reset_index()
    wide['C02_minus_C07'] = wide.RMP_C02 - wide.RMP_C07
    y = wide.C02_minus_C07.to_numpy(float)
    x = wide.ordered_state.to_numpy(float)
    groups = wide.patient_id.astype(str).to_numpy()
    fit = fit_trend(y, x, groups)
    rng = np.random.default_rng(SEED + 2)
    permutations = build_blocked_permutations(wide[['patient_id', 'normalized_primary_state', 'ordered_state']], rng)
    (_, p, _) = permutation_pvalue(y, x, groups, permutations, fit)
    lopo = []
    for patient in pd.unique(groups):
        keep = groups != patient
        lopo.append(float(fit_trend(y[keep], x[keep], groups[keep])['slope']))
    sign_fraction = float(np.mean(np.sign(lopo) == np.sign(fit['slope'])))
    if fit['ci_low'] * fit['ci_high'] > 0 and p < 0.05 and (sign_fraction >= 0.8):
        conclusion = 'YES'
    else:
        t2 = trends[(trends.rmp_id == 'RMP_C02') & (trends.signature_type == 'TOP50')].iloc[0]
        t7 = trends[(trends.rmp_id == 'RMP_C07') & (trends.signature_type == 'TOP50')].iloc[0]
        conclusion = 'NO' if np.sign(t2.slope) == np.sign(t7.slope) and fit['ci_low'] <= 0 <= fit['ci_high'] else 'INCONCLUSIVE'
    row = pd.DataFrame([{'comparison': 'RMP_C02_MINUS_RMP_C07 ordered-state trend', 'difference_slope': fit['slope'], 'standard_error': fit['se'], 'CI_low': fit['ci_low'], 'CI_high': fit['ci_high'], 'mixed_model_P': fit['p'], 'patient_blocked_permutation_P': p, 'LOPO_sign_fraction': sign_fraction, 'model_method': fit['method'], 'C02_C07_DISTINCT_PROGRESSION_TRAJECTORIES': conclusion}])
    row.to_csv(DIRS['stats'] / 'C02_C07_trajectory_comparison.tsv', sep='\t', index=False)
    return row

def primary_tables(trends: pd.DataFrame, sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.read_csv(resolve_data(PHASE_D / 'rmp_annotation_table.tsv'), sep='\t').set_index('rmp_id')
    top50 = trends[trends.signature_type.eq('TOP50')].copy()
    top50 = top50.merge(sensitivity[['rmp_id', 'sign_consistency']], on='rmp_id', how='left')
    top50['annotation'] = top50.rmp_id.map(annotations.biological_label)
    top50['Top100_direction_consistent'] = top50.sign_consistency
    fields = ['rmp_id', 'tier', 'annotation', 'n_patients', 'n_patient_state_observations', 'slope', 'standard_error', 'CI_low', 'CI_high', 'mixed_model_P', 'mixed_model_BH_FDR', 'permutation_P', 'BH_FDR', 'direction', 'LOPO_sign_fraction', 'minimum_LOPO_slope', 'maximum_LOPO_slope', 'major_patient_instability', 'Top100_direction_consistent', 'model_method', 'model_converged', 'model_warnings', 'interpretation']
    primary = top50[top50.rmp_id.isin(PRIMARY)][fields].copy()
    primary.columns = ['RMP' if c == 'rmp_id' else c for c in primary.columns]
    primary.to_csv(DIRS['tables'] / 'Table_G0_2_primary_RMP_progression_tests.tsv', sep='\t', index=False)
    exploratory = top50[top50.rmp_id.isin(EXPLORATORY)][fields].copy()
    exploratory.columns = ['RMP' if c == 'rmp_id' else c for c in exploratory.columns]
    exploratory.to_csv(DIRS['tables'] / 'Table_G0_5_exploratory_C03_C06.tsv', sep='\t', index=False)
    return (primary, exploratory)

def make_figure(characteristics: pd.DataFrame, patient_state: pd.DataFrame, primary: pd.DataFrame) -> None:
    sns.set_theme(style='whitegrid', context='paper')
    colors = {'NON_TUMOR': '#4C78A8', 'DYSPLASIA': '#F2B134', 'TUMOR': '#D1495B'}
    fig = plt.figure(figsize=(15.5, 12.5), constrained_layout=True)
    grid = GridSpec(3, 4, figure=fig, height_ratios=[0.85, 1.25, 1.05])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0:2])
    ax_c = fig.add_subplot(grid[1, 2:4])
    ax_d = fig.add_subplot(grid[2, 0:2])
    ax_e = fig.add_subplot(grid[2, 2:4])
    ax_a.axis('off')
    xs = [0.18, 0.5, 0.82]
    for (i, state) in enumerate(STATE_ORDER):
        row = characteristics[characteristics.state.eq(state)].iloc[0]
        ax_a.text(xs[i], 0.62, state.replace('_', ' '), ha='center', va='center', fontsize=14, fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.55', facecolor=colors[state], edgecolor='none'))
        ax_a.text(xs[i], 0.28, f'{int(row.patients_evaluable_after_QC)}/{int(row.patients_before_QC)} patients evaluable | {int(row.samples_evaluable_after_QC)}/{int(row.samples_before_QC)} samples evaluable\n{int(row.epithelial_cells_after_QC):,} retained epithelial cells', ha='center', va='center', fontsize=10)
        if i < 2:
            ax_a.annotate('', xy=(xs[i + 1] - 0.1, 0.62), xytext=(xs[i] + 0.1, 0.62), arrowprops=dict(arrowstyle='->', lw=2, color='#555555'))
    ax_a.text(0.01, 0.98, 'A', transform=ax_a.transAxes, fontsize=16, fontweight='bold', va='top')
    ax_a.text(0.5, 0.95, 'GSE199654 pathology-state design (official 18 / 15 / 14 sample structure)', ha='center', va='top', fontsize=13)

    def patient_plot(ax: plt.Axes, rmp: str, label: str) -> None:
        data = patient_state[patient_state.rmp_id.eq(rmp) & patient_state.signature_type.eq('TOP50')].copy()
        for (patient, group) in data.groupby('patient_id'):
            group = group.sort_values('ordered_state')
            ax.plot(group.ordered_state, group.patient_state_score, color='#A0A0A0', alpha=0.45, lw=0.8, zorder=1)
        sns.stripplot(data=data, x='normalized_primary_state', y='patient_state_score', order=STATE_ORDER, hue='normalized_primary_state', palette=colors, size=5.5, jitter=0.08, alpha=0.9, legend=False, ax=ax, zorder=2)
        medians = data.groupby('normalized_primary_state').patient_state_score.median().reindex(STATE_ORDER)
        ax.plot(range(3), medians.values, color='black', marker='D', markersize=6, lw=2.2, zorder=3, label='patient median')
        ax.set_xlabel('')
        ax.set_ylabel('Patient-state frozen Top50 rank score')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        ax.legend(frameon=False, loc='best')
    patient_plot(ax_b, 'RMP_C02', 'B  RMP_C02 — patient-aware state estimates')
    patient_plot(ax_c, 'RMP_C07', 'C  RMP_C07 — patient-aware state estimates')
    plot = primary.copy().sort_values('slope')
    y = np.arange(len(plot))
    xerr = np.vstack([plot.slope - plot.CI_low, plot.CI_high - plot.slope])
    point_colors = ['#D1495B' if x > 0 else '#4C78A8' for x in plot.slope]
    ax_d.errorbar(plot.slope, y, xerr=xerr, fmt='none', ecolor='#555555', capsize=3, lw=1.5)
    ax_d.scatter(plot.slope, y, c=point_colors, s=55, zorder=3)
    ax_d.axvline(0, color='black', lw=1, ls='--')
    ax_d.set_yticks(y, plot.RMP)
    ax_d.set_xlabel('Ordered-state slope per step (95% CI)')
    ax_d.set_title('D  Primary RMP patient-aware trends', fontsize=12, fontweight='bold')
    for (yy, (_, row)) in enumerate(plot.iterrows()):
        ax_d.text(ax_d.get_xlim()[1], yy, f'FDR={row.BH_FDR:.3g}', ha='right', va='center', fontsize=8)
    top50 = patient_state[patient_state.signature_type.eq('TOP50') & patient_state.rmp_id.isin(PRIMARY)]
    state_median = top50.groupby(['normalized_primary_state', 'rmp_id']).patient_state_score.median().unstack('rmp_id').reindex(STATE_ORDER)[PRIMARY]
    z = (state_median - state_median.mean(axis=0)) / state_median.std(axis=0, ddof=0).replace(0, np.nan)
    sns.heatmap(z, cmap='vlag', center=0, annot=True, fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Within-program state z-score'}, ax=ax_e)
    ax_e.set_xlabel('')
    ax_e.set_ylabel('')
    ax_e.set_title('E  State-by-program patient-median structure', fontsize=12, fontweight='bold')
    ax_e.set_yticklabels([x.replace('_', ' ') for x in STATE_ORDER], rotation=0)
    fig.suptitle('Frozen recurrent malignant programs across ESCC carcinogenesis', fontsize=17, fontweight='bold')
    fig.savefig(DIRS['figures'] / 'Figure_G0_ESCC_carcinogenesis_RMP_dynamics.pdf', bbox_inches='tight')
    fig.savefig(DIRS['figures'] / 'Figure_G0_ESCC_carcinogenesis_RMP_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def write_methods_lock() -> None:
    config = {'phase': 'G0', 'title': 'Patient-aware projection of frozen recurrent malignant programs across ESCC carcinogenesis', 'date': '2026-08-27', 'seed': SEED, 'dataset': 'GSE199654', 'new_independent_validation_cohort': False, 'states': {'NON_TUMOR': 0, 'DYSPLASIA': 1, 'TUMOR': 2}, 'official_pathology_mapping': {'N': 'NON_TUMOR', 'INF': 'NON_TUMOR', 'LGIN': 'DYSPLASIA', 'HGIN': 'DYSPLASIA', 'Stage_I_to_IV': 'TUMOR'}, 'official_sample_count_gate': {'NON_TUMOR': 18, 'DYSPLASIA': 15, 'TUMOR': 14}, 'qc': '200 <= detected genes < 6000 and mitochondrial UMI fraction < 5%', 'primary_signature': 'Phase D frozen Top50', 'sensitivity_signature': 'Phase D frozen Top100', 'cell_scoring': 'Phase E deterministic within-cell rank score on raw nonnegative UMI counts; covered genes only; zero counts tied low; no imputation/substitution/training', 'sample_summary': 'median cell score', 'patient_state_summary': 'unweighted median of sample-level median scores', 'primary_model': 'RMP_score ~ ordered_state + (1|patient), REML random-intercept mixed model', 'fallback_model': 'within-patient centered fixed-effect regression with patient-cluster-robust SE, only if MixedLM fails to converge', 'permutation': "10,000 patient-blocked label permutations within each patient's observed states; mixed-model GLS statistic with observed variance components", 'primary_multiple_testing': 'BH across five Tier-1 Top50 ordered-state permutation P values', 'paired_bootstrap': N_BOOTSTRAPS, 'lopo_major_instability_rule': 'abs(full slope) >= full SE and <80% LOPO slopes preserve sign', 'supported_rule': 'permutation BH FDR <0.05, mixed-model 95% CI excludes zero, LOPO sign fraction >=0.80, and no major instability', 'limited_rule': 'nominal patient-aware permutation or mixed-model P <0.05 with LOPO sign fraction >=0.80 but supported rule not met', 'c02_c07_distinct_rule': 'difference-score trend CI excludes zero, blocked permutation P<0.05, and LOPO sign fraction>=0.80', 'prohibited': ['signature modification', 'new cNMF', 'rank selection', 'cross-cohort matching', 'pseudotime', 'survival', 'classifier', 'outcome optimization', 'patient deletion for results']}
    (DIRS['audit'] / 'PHASE_G0_METHODS_LOCK.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

def write_environment() -> None:
    lines = [f'Python=={platform.python_version()}', f'numpy=={np.__version__}', f'pandas=={pd.__version__}', f'scipy=={scipy.__version__}', f'anndata=={ad.__version__}', f'statsmodels=={statsmodels.__version__}', f'matplotlib=={plt.matplotlib.__version__}', f'seaborn=={sns.__version__}', f'random_seed={SEED}', f'patient_blocked_permutations={N_PERMUTATIONS}', f'paired_bootstraps={N_BOOTSTRAPS}']
    (OUT / 'requirements_phase_G0.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    write_methods_lock()
    audit = frozen_input_audit()
    (adata, _) = read_gse199654_all_epithelial()
    (crosswalk, qc, characteristics) = build_metadata_outputs(adata)
    signatures = load_signatures()
    coverage = signature_coverage(adata, signatures)
    (cell_table, sample, patient_state) = score_and_aggregate(adata, signatures)
    (trends, lopo, lopo_summary) = trend_analyses(patient_state)
    transitions = paired_transitions(patient_state)
    sensitivity = top100_sensitivity(patient_state, trends)
    comparison = c02_c07_comparison(patient_state, trends)
    (primary, exploratory) = primary_tables(trends, sensitivity)
    make_figure(characteristics, patient_state, primary)
    write_environment()
    warnings_rows = []
    for row in trends.itertuples(index=False):
        if str(row.model_warnings).strip():
            warnings_rows.append({'level': 'WARNING', 'scope': f'{row.rmp_id}/{row.signature_type}', 'message': row.model_warnings})
        if row.model_method != 'RANDOM_INTERCEPT_MIXED_MODEL_REML':
            warnings_rows.append({'level': 'WARNING', 'scope': f'{row.rmp_id}/{row.signature_type}', 'message': 'Pre-specified patient fixed-effect fallback used'})
    if not warnings_rows:
        warnings_rows.append({'level': 'INFO', 'scope': 'FINAL', 'message': 'No unresolved computational errors; model warnings, if any, are retained in trend tables.'})
    pd.DataFrame(warnings_rows).to_csv(DIRS['logs'] / 'warnings_errors.tsv', sep='\t', index=False)
    summary = {'input_audit': 'PASS', 'pathology_metadata_verified': True, 'official_samples': dict(zip(characteristics.state, characteristics.samples.astype(int))), 'patients_by_state': dict(zip(characteristics.state, characteristics.patients.astype(int))), 'retained_cells_by_state': dict(zip(characteristics.state, characteristics.epithelial_cells_after_QC.astype(int))), 'patient_state_observations': int(patient_state[['patient_id', 'normalized_primary_state']].drop_duplicates().shape[0]), 'formal_inference_unit': 'PATIENT_STATE', 'primary_top50': primary.to_dict(orient='records'), 'exploratory_top50': exploratory.to_dict(orient='records'), 'c02_c07_comparison': comparison.to_dict(orient='records')[0], 'frozen_signatures_modified': False, 'outcome_optimized_analysis': False, 'cell_level_pseudoreplication_used': False, 'new_independent_validation_claimed': False}
    (DIRS['stats'] / 'PHASE_G0_RESULTS_SUMMARY.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'audit': f"{audit.status.eq('PASS').sum()}/{len(audit)} PASS", 'retained_cells': int(adata.obs.qc_pass.sum()), 'patients': int(adata.obs.loc[adata.obs.qc_pass, 'patient_id'].nunique()), 'patient_state_observations': summary['patient_state_observations'], 'primary_results': primary[['RMP', 'slope', 'permutation_P', 'BH_FDR', 'LOPO_sign_fraction', 'interpretation']].to_dict(orient='records'), 'c02_c07': comparison.iloc[0].to_dict()}, indent=2, default=str))
if __name__ == '__main__':
    main()
