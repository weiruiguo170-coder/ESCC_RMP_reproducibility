from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import gzip
import hashlib
import json
import logging
import math
import platform
import re
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu, rankdata, spearmanr
PROJECT = INPUT_ROOT
OUT = PROJECT / 'results' / 'pdo_drug_response'
RAW_GEO = PROJECT / 'data' / 'raw' / 'GSE269447'
RAW_PAPER = RAW_GEO / 'supplementary'
PROCESSED = OUT / 'processed'
FIGURES = OUT / 'figures'
LOGS = OUT / 'logs'
PHASE_D = PROJECT / 'signatures'
SEED = 20260831
N_BOOT = 1000
N_RANDOM = 1000
PRIMARY = ['RMP_C01', 'RMP_C02', 'RMP_C07']
SHORT = {'RMP_C01': 'C01', 'RMP_C02': 'C02', 'RMP_C07': 'C07'}
PROGRAM_NAMES = {'RMP_C01': 'proliferation / cell-cycle', 'RMP_C02': 'differentiation / keratinization', 'RMP_C07': 'IFN / inflammatory program'}
RESISTANT = {7, 11, 15, 17, 29, 34, 37}
SENSITIVE_BULK = {12, 18, 19, 20, 25, 35, 36, 39}
NRF2_TARGETS = ['ALDH3A1', 'ABCC1', 'CES1', 'SRXN1', 'CBR1', 'AKR1C1', 'SPP1', 'TXN', 'TXNRD1', 'PTGR1']

def setup() -> logging.Logger:
    for p in (PROCESSED, FIGURES, LOGS):
        p.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('pdo_validation')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s\t%(levelname)s\t%(message)s')
    fh = logging.FileHandler(LOGS / 'analysis.log', mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda : stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest().upper()

def bh_fdr(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out

def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    return tuple((float(x) for x in np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])))

def safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return (np.nan, np.nan)
    stat = spearmanr(x[keep], y[keep])
    return (float(stat.statistic), float(stat.pvalue))

def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int=N_BOOT) -> tuple[float, float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        (rho, _) = safe_spearman(x[idx], y[idx])
        if np.isfinite(rho):
            stats.append(rho)
    if not stats:
        return (np.nan, np.nan, 0)
    (low, high) = np.percentile(stats, [2.5, 97.5])
    return (float(low), float(high), len(stats))

def read_gmt(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open('r', encoding='utf-8') as stream:
        for line in stream:
            fields = line.rstrip('\n').split('\t')
            if len(fields) >= 3:
                out[fields[0]] = [x.strip().upper() for x in fields[2:] if x.strip()]
    return out

def parse_sample_filename(path: Path) -> tuple[str, int, str]:
    match = re.match('(GSM\\d+)_(\\d+)_(Tumor|Normal)-Org_TPM\\.txt\\.gz$', path.name)
    if not match:
        raise ValueError(f'Unexpected GEO filename: {path.name}')
    return (match.group(1), int(match.group(2)), match.group(3))

def load_hgnc_map() -> dict[str, str]:
    path = PROJECT / 'data' / 'raw' / 'reference' / 'HGNC_complete_set.txt'
    hgnc = pd.read_csv(resolve_data(path), sep='\t', dtype=str, low_memory=False)
    hgnc = hgnc.loc[hgnc['status'].eq('Approved') & hgnc['ensembl_gene_id'].notna() & hgnc['symbol'].notna()].copy()
    hgnc['ensembl_gene_id'] = hgnc['ensembl_gene_id'].str.replace('\\.\\d+$', '', regex=True)
    hgnc['symbol'] = hgnc['symbol'].str.strip().str.upper()
    hgnc = hgnc.drop_duplicates('ensembl_gene_id', keep='first')
    return dict(zip(hgnc['ensembl_gene_id'], hgnc['symbol']))

def load_expression(logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = sorted(RAW_GEO.glob('GSM*_TPM.txt.gz'))
    if len(paths) != 20:
        raise RuntimeError(f'Expected 20 GEO TPM files, found {len(paths)}')
    mapping = load_hgnc_map()
    tables: dict[str, pd.Series] = {}
    meta_rows: list[dict[str, object]] = []
    id_type = 'Ensembl gene ID (GRCh38.105; stable ID without version)'
    for path in paths:
        (gsm, pdo_num, tissue) = parse_sample_filename(path)
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            head = stream.readline()
        if 'tpm' not in head.lower():
            raise RuntimeError(f'TPM header missing: {path}')
        frame = pd.read_csv(resolve_data(path), sep='\t', index_col=0)
        tpm_col = next((c for c in frame.columns if str(c).strip().lower() == 'tpm'), None)
        if tpm_col is None:
            raise RuntimeError(f'TPM column missing in {path}')
        ens = pd.Index(frame.index.astype(str)).str.replace('\\.\\d+$', '', regex=True)
        symbols = pd.Index([mapping.get(x, x.upper()) for x in ens])
        series = pd.Series(pd.to_numeric(frame[tpm_col], errors='coerce').to_numpy(dtype=float), index=symbols, name=f'ESCCO{pdo_num}')
        series = series.groupby(level=0, sort=True).mean()
        key = f'ESCCO{pdo_num}_{tissue}'
        tables[key] = series
        response = 'Resistant' if pdo_num in RESISTANT else 'Sensitive' if pdo_num in SENSITIVE_BULK else 'Not_classified'
        meta_rows.append({'GSM': gsm, 'PDO_ID': f'ESCCO{pdo_num}', 'patient_ID': str(pdo_num), 'PDO_number': pdo_num, 'tissue': tissue, 'response_group': response, 'source_file': str(path.resolve()), 'source_sha256': sha256(path)})
    common = set.intersection(*(set(x.index) for x in tables.values()))
    if len(common) < 20000:
        raise RuntimeError(f'Unexpectedly small common expression universe: {len(common)}')
    universe = sorted(common)
    matrix = pd.DataFrame({k: v.reindex(universe) for (k, v) in tables.items()}, index=universe)
    if matrix.isna().any().any():
        raise RuntimeError('NA values after common-universe expression assembly')
    metadata = pd.DataFrame(meta_rows).sort_values(['PDO_number', 'tissue']).reset_index(drop=True)
    logger.info('Loaded %d samples; common canonical feature universe=%d; identifier=%s', matrix.shape[1], matrix.shape[0], id_type)
    return (matrix, metadata, {'original_identifier_type': id_type, 'mapping_source': 'data/raw/reference/HGNC_complete_set.txt'})

def load_signatures(expression_genes: pd.Index, logger: logging.Logger) -> tuple[dict[str, list[str]], pd.DataFrame]:
    frozen_table_path = PHASE_D / 'rmp_frozen_signatures.tsv'
    frozen = pd.read_csv(resolve_data(frozen_table_path), sep='\t')
    signatures: dict[str, list[str]] = {}
    rows: list[dict[str, object]] = []
    expression_set = set(expression_genes.astype(str))
    for rmp in PRIMARY:
        for size in (50, 100):
            kind = f'TOP{size}'
            path = PHASE_D / f'{rmp}_signature_top{size}.txt'
            genes = [x.strip().upper() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
            if len(genes) != size or len(set(genes)) != size:
                raise RuntimeError(f'Frozen signature size/duplicate failure: {path}')
            expected = frozen.loc[frozen['rmp_id'].eq(rmp) & frozen['signature_type'].eq(kind)].sort_values('consensus_rank')
            if genes != expected['gene'].astype(str).str.upper().tolist():
                raise RuntimeError(f'Frozen table/list mismatch: {rmp} {kind}')
            file_hash = sha256(path)
            recorded = expected['signature_sha256'].astype(str).str.upper().unique().tolist()
            if recorded != [file_hash]:
                raise RuntimeError(f'Frozen signature SHA mismatch: {rmp} {kind}: recorded={recorded}, file={file_hash}')
            signatures[f'{rmp}_{kind}'] = genes
            matched = [g for g in genes if g in expression_set]
            missing_fraction = 1.0 - len(matched) / size
            warning = 'WARNING_TOP50_MISSING_GT20_PERCENT' if size == 50 and missing_fraction > 0.2 else 'NONE'
            for (rank, gene) in enumerate(genes, start=1):
                rows.append({'RMP': SHORT[rmp], 'program_name': PROGRAM_NAMES[rmp], 'gene': gene, 'rank': rank, 'Top50_or_Top100': kind, 'source_file': str(path.resolve()), 'source_file_SHA256': file_hash, 'signature_version': expected.iloc[0]['signature_version'], 'gene_identifier_type': 'HGNC-approved gene symbol (uppercase)', 'matched_in_PDO_RNAseq': gene in expression_set, 'matched_genes_in_signature': len(matched), 'signature_size': size, 'missing_fraction': missing_fraction, 'coverage_warning': warning})
            logger.info('Frozen %s %s coverage: %d/%d', rmp, kind, len(matched), size)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUT / '02_FROZEN_RMP_MANIFEST.tsv', sep='\t', index=False)
    return (signatures, manifest)

def rank_scores(expression: pd.DataFrame, gene_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, np.ndarray, pd.Index]:
    genes = expression.index
    values = expression.to_numpy(dtype=float).T
    ranks = np.apply_along_axis(rankdata, 1, values, method='average')
    detected_global = (values > 0).any(axis=0)
    name_to_idx = {str(g): i for (i, g) in enumerate(genes)}
    score_data: dict[str, np.ndarray] = {}
    for (key, gene_list) in gene_sets.items():
        idx = np.asarray([name_to_idx[g] for g in gene_list if g in name_to_idx and detected_global[name_to_idx[g]]], dtype=int)
        n_genes = ranks.shape[1]
        m = len(idx)
        if m == 0 or n_genes <= m:
            score_data[key] = np.full(ranks.shape[0], np.nan)
        else:
            minimum = (m + 1.0) / 2.0
            score_data[key] = np.clip((ranks[:, idx].mean(axis=1) - minimum) / (n_genes - m), 0, 1)
    return (pd.DataFrame(score_data, index=expression.columns), ranks, genes)

def parse_source_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = RAW_PAPER / '42003_2025_7869_MOESM3_ESM.xlsx'
    auc_sheet = pd.read_excel(resolve_data(source), sheet_name='Fig. 6e', header=None)
    auc_rows = auc_sheet.loc[auc_sheet.iloc[:, 0].astype(str).str.fullmatch('ESCCO\\d+')].copy()
    auc = pd.DataFrame({'PDO_ID': auc_rows.iloc[:, 0].astype(str), 'CDDP_AUC': pd.to_numeric(auc_rows.iloc[:, 1])})
    dose = pd.read_excel(resolve_data(source), sheet_name='Fig. 6a', header=None)
    header = dose.iloc[2]
    concentrations = pd.to_numeric(dose.iloc[4:, 0]).to_numpy(dtype=float)
    auc_check: list[dict[str, object]] = []
    for (col, value) in header.items():
        if isinstance(value, str) and re.fullmatch('ESCCO\\d+', value):
            means = dose.iloc[4:, col:col + 3].apply(pd.to_numeric, errors='coerce').mean(axis=1).to_numpy(dtype=float)
            recomputed = float(np.trapezoid(means, concentrations))
            reported = float(auc.loc[auc['PDO_ID'].eq(value), 'CDDP_AUC'].iloc[0])
            auc_check.append({'PDO_ID': value, 'CDDP_AUC_reported': reported, 'CDDP_AUC_recomputed_trapezoid': recomputed, 'absolute_difference': abs(recomputed - reported)})
    auc_audit = pd.DataFrame(auc_check)
    if len(auc_audit) != 10:
        raise RuntimeError(f'Expected 10 mapped CDDP AUC values, found {len(auc_audit)}')
    auc_audit['reconstruction_QC'] = np.where(auc_audit['absolute_difference'] <= 1.0, 'PASS_WITHIN_1_AUC_UNIT', 'SOURCE_DATA_WARNING_REPORTED_AUC_DIFFERS_FROM_TRAPEZOID_RECONSTRUCTION')
    fixed = pd.read_excel(resolve_data(source), sheet_name='Fig. 4b', header=None)
    fixed_header = fixed.iloc[2]
    fixed_rows: list[dict[str, object]] = []
    for (col, value) in fixed_header.items():
        if isinstance(value, str) and re.fullmatch('ESCCO\\d+', value):
            block = fixed.iloc[4:, [0, col, col + 1, col + 2]].copy()
            block.columns = ['drug', 'well1', 'well2', 'well3']
            for (_, row) in block.iterrows():
                vals = pd.to_numeric(row[['well1', 'well2', 'well3']], errors='coerce').to_numpy(dtype=float)
                fixed_rows.append({'PDO_ID': value, 'drug': str(row['drug']), 'relative_viability_mean': float(np.nanmean(vals)), 'relative_viability_sd': float(np.nanstd(vals, ddof=1)), 'technical_replicates': int(np.isfinite(vals).sum())})
    fixed_response = pd.DataFrame(fixed_rows)
    clinical = pd.read_excel(resolve_data(source), sheet_name='Supplementary Table 1', header=None)
    clin_rows = clinical.iloc[4:, :13].copy()
    clin_rows.columns = ['patient_ID', 'sex', 'age', 'type', 'pTNM', 'pStage', 'neoadjuvant_therapy', 'neoadjuvant_drug', 'clinical_response', 'normal_organoid', 'tumor_organoid', 'response_group', 'NRF2_mutation']
    clin_rows = clin_rows.loc[pd.to_numeric(clin_rows['patient_ID'], errors='coerce').notna()].copy()
    clin_rows['patient_ID'] = pd.to_numeric(clin_rows['patient_ID']).astype(int).astype(str)
    return (auc, auc_audit, fixed_response, clin_rows)

def primary_analysis(scores: pd.DataFrame, auc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    joined = scores.join(auc.set_index('PDO_ID'), how='inner')
    rows: list[dict[str, object]] = []
    for rmp in PRIMARY:
        x = joined[f'{rmp}_TOP50'].to_numpy(dtype=float)
        y = joined['CDDP_AUC'].to_numpy(dtype=float)
        (rho, p) = safe_spearman(x, y)
        (low, high, valid) = bootstrap_spearman(x, y, rng)
        rows.append({'RMP': SHORT[rmp], 'program_name': PROGRAM_NAMES[rmp], 'signature': 'Top50', 'endpoint': 'CDDP_AUC', 'n_PDO': len(joined), 'spearman_rho': rho, 'bootstrap_95CI_low': low, 'bootstrap_95CI_high': high, 'bootstrap_valid_resamples': valid, 'p_value': p, 'AUC_direction': 'higher AUC = greater survival/resistance'})
    out = pd.DataFrame(rows)
    out['BH_FDR_primary_3_RMP'] = bh_fdr(out['p_value'])
    return out

def secondary_analysis(scores_meta: pd.DataFrame, fixed_response: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rmp in PRIMARY:
        col = f'{rmp}_TOP50'
        resistant = scores_meta.loc[scores_meta['response_group'].eq('Resistant'), col].to_numpy(dtype=float)
        sensitive = scores_meta.loc[scores_meta['response_group'].eq('Sensitive'), col].to_numpy(dtype=float)
        test = mannwhitneyu(resistant, sensitive, alternative='two-sided')
        (q1r, medr, q3r) = quantiles(resistant)
        (q1s, meds, q3s) = quantiles(sensitive)
        rank_biserial = 2.0 * float(test.statistic) / (len(resistant) * len(sensitive)) - 1.0
        rows.append({'analysis_type': 'Resistant_vs_Sensitive', 'RMP': SHORT[rmp], 'program_name': PROGRAM_NAMES[rmp], 'signature': 'Top50', 'endpoint': 'author_classification', 'n_resistant': len(resistant), 'n_sensitive': len(sensitive), 'resistant_median': medr, 'resistant_IQR_Q1': q1r, 'resistant_IQR_Q3': q3r, 'sensitive_median': meds, 'sensitive_IQR_Q1': q1s, 'sensitive_IQR_Q3': q3s, 'effect_size_rank_biserial_resistant_minus_sensitive': rank_biserial, 'mann_whitney_U': float(test.statistic), 'p_value': float(test.pvalue)})
    group_idx = [i for (i, row) in enumerate(rows) if row['analysis_type'] == 'Resistant_vs_Sensitive']
    group_fdr = bh_fdr([rows[i]['p_value'] for i in group_idx])
    for (i, fdr) in zip(group_idx, group_fdr):
        rows[i]['BH_FDR_within_secondary_family_3_RMP'] = fdr
    fu = fixed_response.loc[fixed_response['drug'].eq('5-FU'), ['PDO_ID', 'relative_viability_mean']].set_index('PDO_ID')
    joined = scores_meta.set_index('PDO_ID').join(fu, how='inner')
    start = len(rows)
    for rmp in PRIMARY:
        x = joined[f'{rmp}_TOP50'].to_numpy(dtype=float)
        y = joined['relative_viability_mean'].to_numpy(dtype=float)
        (rho, p) = safe_spearman(x, y)
        (low, high, valid) = bootstrap_spearman(x, y, rng)
        rows.append({'analysis_type': '5FU_continuous', 'RMP': SHORT[rmp], 'program_name': PROGRAM_NAMES[rmp], 'signature': 'Top50', 'endpoint': '5-FU relative viability at 2.46 uM (higher = more resistant)', 'n_PDO': len(joined), 'spearman_rho': rho, 'bootstrap_95CI_low': low, 'bootstrap_95CI_high': high, 'bootstrap_valid_resamples': valid, 'p_value': p})
    idx = list(range(start, len(rows)))
    fdrs = bh_fdr([rows[i]['p_value'] for i in idx])
    for (i, fdr) in zip(idx, fdrs):
        rows[i]['BH_FDR_within_secondary_family_3_RMP'] = fdr
    return pd.DataFrame(rows)

def robustness_analysis(scores: pd.DataFrame, auc: pd.DataFrame, primary: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = scores.join(auc.set_index('PDO_ID'), how='inner')
    summary: list[dict[str, object]] = []
    detail: list[dict[str, object]] = []
    for rmp in PRIMARY:
        y = joined['CDDP_AUC'].to_numpy(dtype=float)
        x50 = joined[f'{rmp}_TOP50'].to_numpy(dtype=float)
        x100 = joined[f'{rmp}_TOP100'].to_numpy(dtype=float)
        (rho50, p50) = safe_spearman(x50, y)
        (rho100, p100) = safe_spearman(x100, y)
        (score_rho, score_p) = safe_spearman(x50, x100)
        loo: list[float] = []
        for (i, pdo) in enumerate(joined.index):
            keep = np.arange(len(joined)) != i
            (rho, p) = safe_spearman(x50[keep], y[keep])
            loo.append(rho)
            detail.append({'RMP': SHORT[rmp], 'omitted_PDO': pdo, 'loo_spearman_rho': rho, 'loo_p_value': p})
        loo_arr = np.asarray(loo, dtype=float)
        full_sign = np.sign(rho50)
        direction_consistency = float(np.mean(np.sign(loo_arr[np.isfinite(loo_arr)]) == full_sign) * 100.0) if full_sign != 0 else np.nan
        ci_row = primary.loc[primary['RMP'].eq(SHORT[rmp])].iloc[0]
        summary.append({'RMP': SHORT[rmp], 'program_name': PROGRAM_NAMES[rmp], 'endpoint': 'CDDP_AUC', 'Top50_rho': rho50, 'Top50_p': p50, 'Top100_rho': rho100, 'Top100_p': p100, 'Top50_Top100_score_rho_across_PDO': score_rho, 'Top50_Top100_score_p': score_p, 'Top50_Top100_association_direction_consistent': bool(np.sign(rho50) == np.sign(rho100)), 'LOO_min_rho': float(np.nanmin(loo_arr)), 'LOO_max_rho': float(np.nanmax(loo_arr)), 'LOO_median_rho': float(np.nanmedian(loo_arr)), 'LOO_direction_consistency_percent': direction_consistency, 'bootstrap_resamples': N_BOOT, 'bootstrap_95CI_low': float(ci_row['bootstrap_95CI_low']), 'bootstrap_95CI_high': float(ci_row['bootstrap_95CI_high'])})
    return (pd.DataFrame(summary), pd.DataFrame(detail))

def matched_random_null(expression: pd.DataFrame, ranks: np.ndarray, signatures: dict[str, list[str]], scores: pd.DataFrame, auc: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    genes = expression.index
    detection = (expression.to_numpy(dtype=float) > 0).mean(axis=1)
    bins = pd.qcut(pd.Series(detection, index=genes).rank(method='first'), 10, labels=False).astype(int)
    name_to_idx = {g: i for (i, g) in enumerate(genes)}
    detected_global = detection > 0
    joined_ids = scores.index.intersection(auc['PDO_ID'])
    auc_vec = auc.set_index('PDO_ID').loc[joined_ids, 'CDDP_AUC'].to_numpy(dtype=float)
    score_rows = scores.loc[joined_ids]
    summary: list[dict[str, object]] = []
    draws: list[dict[str, object]] = []
    set_rows: list[dict[str, object]] = []
    for rmp in PRIMARY:
        sig_genes = [g for g in signatures[f'{rmp}_TOP50'] if g in name_to_idx and detected_global[name_to_idx[g]]]
        sig_idx = np.asarray([name_to_idx[g] for g in sig_genes], dtype=int)
        sig_set = set(sig_idx.tolist())
        bin_counts = pd.Series(bins.iloc[sig_idx].to_numpy()).value_counts().sort_index().to_dict()
        pools = {int(b): np.asarray([i for i in np.where(bins.to_numpy() == b)[0] if i not in sig_set and detected_global[i]], dtype=int) for b in bin_counts}
        (observed_rho, _) = safe_spearman(score_rows[f'{rmp}_TOP50'].to_numpy(dtype=float), auc_vec)
        random_rhos: list[float] = []
        for permutation in range(1, N_RANDOM + 1):
            selected: list[int] = []
            for (b, count) in bin_counts.items():
                pool = pools[int(b)]
                if len(pool) < int(count):
                    raise RuntimeError(f'Matched-null pool too small: {rmp} bin={b} need={count} pool={len(pool)}')
                selected.extend(rng.choice(pool, size=int(count), replace=False).tolist())
            idx = np.asarray(selected, dtype=int)
            m = len(idx)
            minimum = (m + 1.0) / 2.0
            random_scores = np.clip((ranks[:, idx].mean(axis=1) - minimum) / (ranks.shape[1] - m), 0, 1)
            sample_positions = [scores.index.get_loc(x) for x in joined_ids]
            (rho, _) = safe_spearman(random_scores[sample_positions], auc_vec)
            random_rhos.append(rho)
            draws.append({'RMP': SHORT[rmp], 'permutation': permutation, 'rho_random': rho, 'abs_rho_random': abs(rho)})
            set_rows.append({'RMP': SHORT[rmp], 'permutation': permutation, 'genes': ';'.join(genes[idx].tolist())})
        arr = np.asarray(random_rhos, dtype=float)
        exceed = int(np.sum(np.abs(arr) >= abs(observed_rho)))
        summary.append({'RMP': SHORT[rmp], 'program_name': PROGRAM_NAMES[rmp], 'signature': 'Top50', 'endpoint': 'CDDP_AUC', 'n_PDO': len(joined_ids), 'observed_rho': observed_rho, 'random_sets': N_RANDOM, 'matching_strategy': 'PDO bulk gene-detection-rate decile; identical covered gene-set size; outcome-blind; fixed seed', 'exceedance_count_abs_rho_random_ge_abs_observed': exceed, 'empirical_fraction': exceed / N_RANDOM, 'empirical_p_plus1': (exceed + 1) / (N_RANDOM + 1), 'random_rho_median': float(np.nanmedian(arr)), 'random_rho_2.5pct': float(np.nanpercentile(arr, 2.5)), 'random_rho_97.5pct': float(np.nanpercentile(arr, 97.5))})
    return (pd.DataFrame(summary), pd.DataFrame(draws), pd.DataFrame(set_rows))

def classify_support(primary: pd.DataFrame, robustness: pd.DataFrame, random_summary: pd.DataFrame) -> pd.DataFrame:
    out = primary.copy()
    labels: list[str] = []
    for (_, row) in out.iterrows():
        rmp = row['RMP']
        robust = robustness.loc[robustness['RMP'].eq(rmp)].iloc[0]
        null = random_summary.loc[random_summary['RMP'].eq(rmp)].iloc[0]
        ci_excludes_zero = bool(row['bootstrap_95CI_low'] > 0 and row['bootstrap_95CI_high'] > 0 or (row['bootstrap_95CI_low'] < 0 and row['bootstrap_95CI_high'] < 0))
        direction_consistent = bool(robust['Top50_Top100_association_direction_consistent'])
        loo_percent = float(robust['LOO_direction_consistency_percent'])
        empirical_p = float(null['empirical_p_plus1'])
        strict_components = [float(row['BH_FDR_primary_3_RMP']) < 0.05, ci_excludes_zero, direction_consistent, loo_percent >= 80.0, empirical_p < 0.05]
        moderate_stat_components = [float(row['BH_FDR_primary_3_RMP']) < 0.1, ci_excludes_zero, empirical_p < 0.1]
        if all(strict_components):
            label = 'STRONG SUPPORT'
        elif direction_consistent and loo_percent >= 70.0 and (sum(moderate_stat_components) >= 2):
            label = 'MODERATE SUPPORT'
        else:
            label = 'WEAK/NO SUPPORT'
        labels.append(label)
    out['support_classification'] = labels
    out['classification_rule'] = 'STRONG: FDR<0.05, bootstrap CI excludes 0, Top50/Top100 direction consistent, LOO direction consistency>=80%, empirical P<0.05; MODERATE: Top50/Top100 consistent, LOO>=70%, and >=2 of FDR<0.10/CI excludes 0/empirical P<0.10; otherwise WEAK/NO'
    return out

def positive_controls(expression: pd.DataFrame, score_index: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    gmt = read_gmt(PHASE_D / 'enrichment_gene_sets' / 'Reactome_Pathways_2024.gmt')
    sets = {'NRF2_10_REPORTED_TARGETS': NRF2_TARGETS, 'REACTOME_IFN_ALPHA_BETA_SIGNALING': gmt['Interferon Alpha Beta Signaling'], 'REACTOME_IFN_GAMMA_SIGNALING': gmt['Interferon Gamma Signaling']}
    (scores, _, _) = rank_scores(expression, sets)
    scores = scores.loc[score_index]
    response = pd.Series({f'ESCCO{x}_Tumor': 'Resistant' if x in RESISTANT else 'Sensitive' for x in sorted(RESISTANT | SENSITIVE_BULK)}, name='response_group')
    rows: list[dict[str, object]] = []
    per_sample = scores.join(response)
    for (key, genes) in sets.items():
        r = per_sample.loc[per_sample['response_group'].eq('Resistant'), key].to_numpy(dtype=float)
        s = per_sample.loc[per_sample['response_group'].eq('Sensitive'), key].to_numpy(dtype=float)
        test = mannwhitneyu(r, s, alternative='two-sided')
        (q1r, medr, q3r) = quantiles(r)
        (q1s, meds, q3s) = quantiles(s)
        effect = 2.0 * float(test.statistic) / (len(r) * len(s)) - 1.0
        expected = 'Resistant > Sensitive' if key.startswith('NRF2') else 'Resistant < Sensitive'
        observed = 'Resistant > Sensitive' if medr > meds else 'Resistant < Sensitive' if medr < meds else 'Equal medians'
        rows.append({'positive_control': key, 'gene_set_size_defined': len(genes), 'genes_covered': sum((g in expression.index for g in genes)), 'expected_direction': expected, 'observed_direction': observed, 'direction_reproduced': expected == observed, 'n_resistant': len(r), 'n_sensitive': len(s), 'resistant_median': medr, 'resistant_IQR_Q1': q1r, 'resistant_IQR_Q3': q3r, 'sensitive_median': meds, 'sensitive_IQR_Q1': q1s, 'sensitive_IQR_Q3': q3s, 'effect_size_rank_biserial_resistant_minus_sensitive': effect, 'mann_whitney_U': float(test.statistic), 'p_value': float(test.pvalue)})
    result = pd.DataFrame(rows)
    result['BH_FDR_across_3_positive_controls'] = bh_fdr(result['p_value'])
    return (result, per_sample)

def make_primary_auc_figure(scores_meta: pd.DataFrame, auc: pd.DataFrame, primary: pd.DataFrame, output_stem: str='Figure_A_RMP_vs_CDDP_AUC') -> None:
    """Render the frozen Top50 RMP–CDDP AUC panels without recomputing statistics."""
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {'C01': '#D55E00', 'C02': '#0072B2', 'C07': '#009E73'}
    joined = scores_meta.set_index('PDO_ID').join(auc.set_index('PDO_ID'), how='inner')
    manual_offsets = {('C01', '15'): ((0, -6), 'center', False), ('C02', '15'): ((0, -6), 'center', False), ('C07', '15'): ((0, -6), 'center', False), ('C01', '7'): ((-5, 7), 'right', True), ('C01', '25'): ((6, -4), 'left', True), ('C07', '7'): ((-5, 7), 'right', True), ('C07', '25'): ((6, -4), 'left', True), ('C01', '35'): ((-5, -4), 'right', False), ('C02', '35'): ((4, -5), 'left', False)}
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    for (ax, rmp) in zip(axes, PRIMARY):
        short = SHORT[rmp]
        x = joined[f'{rmp}_TOP50'].to_numpy(dtype=float)
        y = joined['CDDP_AUC'].to_numpy(dtype=float)
        ax.scatter(x, y, s=28, color=colors[short], edgecolor='white', linewidth=0.5, zorder=3)
        if np.unique(x).size > 1:
            coef = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            ax.plot(grid, np.polyval(coef, grid), color=colors[short], lw=1.1, alpha=0.75, zorder=2)
        x_span = float(np.ptp(x)) or 1.0
        y_span = float(np.ptp(y)) or 1.0
        ax.set_xlim(float(x.min() - 0.06 * x_span), float(x.max() + 0.08 * x_span))
        ax.set_ylim(float(y.min() - 0.04 * y_span), float(y.max() + 0.5 * y_span))
        for (pdo, xx, yy) in zip(joined.index, x, y):
            label = pdo.replace('ESCCO', '')
            (offset, horizontal_alignment, leader) = manual_offsets.get((short, label), ((3, 3), 'left', False))
            ax.annotate(label, (xx, yy), xytext=offset, textcoords='offset points', fontsize=5.4, color='#404040', ha=horizontal_alignment, va='bottom' if offset[1] >= 0 else 'top', bbox={'boxstyle': 'round,pad=0.08', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.72}, arrowprops={'arrowstyle': '-', 'color': '#707070', 'lw': 0.45, 'shrinkA': 1.5, 'shrinkB': 1.5} if leader else None, annotation_clip=True, zorder=4)
        row = primary.loc[primary['RMP'].eq(short)].iloc[0]
        ax.text(0.03, 0.98, short, transform=ax.transAxes, va='top', ha='left', fontsize=9.0, fontweight='bold')
        ax.text(0.03, 0.875, f'ρ = {row.spearman_rho:.2f}\nP = {row.p_value:.3g}\nFDR = {row.BH_FDR_primary_3_RMP:.3g}', transform=ax.transAxes, va='top', ha='left', fontsize=7.8, linespacing=1.15)
        ax.set_xlabel('Frozen Top50 rank score')
        ax.set_ylabel('CDDP AUC' if ax is axes[0] else '')
        ax.ticklabel_format(axis='y', style='plain')
    fig.savefig(FIGURES / f'{output_stem}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / f'{output_stem}.png', dpi=600, bbox_inches='tight')
    fig.savefig(FIGURES / f'{output_stem}.svg', bbox_inches='tight')
    plt.close(fig)

def make_figures(scores_meta: pd.DataFrame, auc: pd.DataFrame, primary: pd.DataFrame, secondary: pd.DataFrame, robustness_detail: pd.DataFrame, random_summary: pd.DataFrame, random_draws: pd.DataFrame) -> None:
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {'C01': '#D55E00', 'C02': '#0072B2', 'C07': '#009E73'}
    make_primary_auc_figure(scores_meta, auc, primary)
    (fig, axes) = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    rng = np.random.default_rng(SEED + 99)
    for (ax, rmp) in zip(axes, PRIMARY):
        short = SHORT[rmp]
        groups = [scores_meta.loc[scores_meta['response_group'].eq(g), f'{rmp}_TOP50'].to_numpy(dtype=float) for g in ('Sensitive', 'Resistant')]
        box = ax.boxplot(groups, positions=[0, 1], widths=0.48, patch_artist=True, showfliers=False, medianprops={'color': 'black', 'linewidth': 1.0}, whiskerprops={'linewidth': 0.8}, capprops={'linewidth': 0.8})
        box['boxes'][0].set(facecolor='#BFD7EA', edgecolor='#4C78A8', linewidth=0.8)
        box['boxes'][1].set(facecolor='#F2B8A2', edgecolor='#C44E52', linewidth=0.8)
        for (pos, vals, color) in [(0, groups[0], '#4C78A8'), (1, groups[1], '#C44E52')]:
            jitter = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), pos) + jitter, vals, s=20, color=color, edgecolor='white', linewidth=0.4, zorder=3)
        row = secondary.loc[secondary['analysis_type'].eq('Resistant_vs_Sensitive') & secondary['RMP'].eq(short)].iloc[0]
        ax.text(0.03, 0.97, f'{short}\nrᵣᵦ = {row.effect_size_rank_biserial_resistant_minus_sensitive:.2f}\nP = {row.p_value:.3g}\nFDR = {row.BH_FDR_within_secondary_family_3_RMP:.3g}', transform=ax.transAxes, va='top', zorder=5, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.88, 'pad': 1.5})
        ax.set_xticks([0, 1], ['Sensitive', 'Resistant'])
        ax.set_ylabel('Frozen Top50 rank score' if ax is axes[0] else '')
    fig.savefig(FIGURES / 'Figure_B_Resistant_vs_Sensitive.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / 'Figure_B_Resistant_vs_Sensitive.png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    (fig, axes) = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    for (j, rmp) in enumerate(PRIMARY):
        short = SHORT[rmp]
        ax = axes[0, j]
        vals = robustness_detail.loc[robustness_detail['RMP'].eq(short), 'loo_spearman_rho'].to_numpy(dtype=float)
        ax.scatter(np.arange(1, len(vals) + 1), vals, color=colors[short], s=20)
        full = float(primary.loc[primary['RMP'].eq(short), 'spearman_rho'].iloc[0])
        ax.axhline(full, color='black', lw=0.9, ls='--', label='Full')
        ax.axhline(0, color='#9A9A9A', lw=0.7)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel('Leave-one-PDO-out iteration')
        ax.set_ylabel('Spearman ρ' if j == 0 else '')
        ax.text(0.03, 0.96, short, transform=ax.transAxes, va='top', fontweight='bold')
        ax = axes[1, j]
        null = random_draws.loc[random_draws['RMP'].eq(short), 'rho_random'].to_numpy(dtype=float)
        ax.hist(null, bins=np.linspace(-1, 1, 31), color='#C9CDD1', edgecolor='white', linewidth=0.4)
        ax.axvline(full, color=colors[short], lw=1.3)
        ax.axvline(-abs(full), color=colors[short], lw=0.9, ls=':')
        row = random_summary.loc[random_summary['RMP'].eq(short)].iloc[0]
        ax.text(0.03, 0.96, f'empirical P = {row.empirical_p_plus1:.3g}', transform=ax.transAxes, va='top')
        ax.set_xlabel('Random-set Spearman ρ')
        ax.set_ylabel('Count' if j == 0 else '')
    fig.savefig(FIGURES / 'Supplementary_Robustness_LOO_random_null.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / 'Supplementary_Robustness_LOO_random_null.png', dpi=600, bbox_inches='tight')
    plt.close(fig)

def write_hash_manifest() -> None:
    paths: list[Path] = []
    for base in (RAW_GEO,):
        paths.extend((p for p in base.rglob('*') if p.is_file()))
    paths.extend([PHASE_D / 'rmp_frozen_signatures.tsv', PROJECT / 'config' / 'frozen_rmp_construction.json'])
    for rmp in PRIMARY:
        paths.extend([PHASE_D / f'{rmp}_signature_top50.txt', PHASE_D / f'{rmp}_signature_top100.txt'])
    rows = [{'file': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(set(paths))]
    pd.DataFrame(rows).to_csv(LOGS / 'SHA256SUMS.tsv', sep='\t', index=False)

def main() -> None:
    logger = setup()
    rng = np.random.default_rng(SEED)
    (expression, sample_meta, identifier_info) = load_expression(logger)
    tumor_columns = [c for c in expression.columns if c.endswith('_Tumor')]
    tumor_expression = expression[tumor_columns].copy()
    (signatures, manifest) = load_signatures(tumor_expression.index, logger)
    (scores, ranks, rank_genes) = rank_scores(tumor_expression, signatures)
    (auc, auc_audit, fixed_response, clinical) = parse_source_data()
    if auc_audit['reconstruction_QC'].str.startswith('SOURCE_DATA_WARNING').any():
        logger.warning('Author-reported AUC retained as endpoint; source-data reconstruction warning(s):\n%s', auc_audit.loc[auc_audit['reconstruction_QC'].str.startswith('SOURCE_DATA_WARNING')].to_string(index=False))
    auc_audit.to_csv(PROCESSED / 'CDDP_AUC_reconstruction_audit.tsv', sep='\t', index=False)
    fixed_response.to_csv(PROCESSED / 'fixed_dose_drug_response.tsv', sep='\t', index=False)
    clinical.to_csv(PROCESSED / 'supplementary_table1_clinical.tsv', sep='\t', index=False)
    expression.to_csv(PROCESSED / 'GSE269447_TPM_canonicalized.tsv', sep='\t')
    score_rows: list[dict[str, object]] = []
    meta_by_key = sample_meta.assign(sample_key=sample_meta['PDO_ID'] + '_' + sample_meta['tissue']).set_index('sample_key')
    fixed_pivot = fixed_response.pivot(index='PDO_ID', columns='drug', values='relative_viability_mean')
    auc_map = auc.set_index('PDO_ID')['CDDP_AUC']
    auc_qc_map = auc_audit.set_index('PDO_ID')['reconstruction_QC']
    clinical_by_patient = clinical.set_index('patient_ID')
    for (key, row) in scores.iterrows():
        meta = meta_by_key.loc[key]
        pdo = str(meta['PDO_ID'])
        patient_id = str(meta['patient_ID'])
        clin = clinical_by_patient.loc[patient_id]
        record: dict[str, object] = {'PDO_ID': pdo, 'patient_ID': patient_id, 'GSM': meta['GSM'], 'response_group': meta['response_group'], 'sex': clin['sex'], 'age': clin['age'], 'pTNM': clin['pTNM'], 'pStage': clin['pStage'], 'neoadjuvant_therapy': clin['neoadjuvant_therapy'], 'neoadjuvant_drug': clin['neoadjuvant_drug'], 'clinical_response': clin['clinical_response'], 'NRF2_mutation': clin['NRF2_mutation'], 'CDDP_AUC': auc_map.get(pdo, np.nan), 'CDDP_fixed_dose_viability': fixed_pivot.loc[pdo, 'Cisplatin'] if pdo in fixed_pivot.index else np.nan, 'CDDP_AUC_QC': auc_qc_map.get(pdo, 'NOT_AVAILABLE'), '5FU_AUC': np.nan, '5FU_fixed_dose_viability': fixed_pivot.loc[pdo, '5-FU'] if pdo in fixed_pivot.index else np.nan, 'AUC_note': 'CDDP AUC from source-data Fig. 6e; no 5-FU AUC was publicly provided'}
        for rmp in PRIMARY:
            record[f'{SHORT[rmp]}_Top50'] = float(row[f'{rmp}_TOP50'])
            record[f'{SHORT[rmp]}_Top100'] = float(row[f'{rmp}_TOP100'])
        score_rows.append(record)
    pdo_scores = pd.DataFrame(score_rows).sort_values('patient_ID', key=lambda x: x.astype(int)).reset_index(drop=True)
    pdo_scores.to_csv(OUT / '03_PDO_RMP_SCORES.tsv', sep='\t', index=False, na_rep='NA')
    score_for_analysis = pdo_scores.set_index('PDO_ID')
    score_for_analysis = score_for_analysis.rename(columns={f'{SHORT[r]}_Top{s}': f'{r}_TOP{s}' for r in PRIMARY for s in (50, 100)})
    score_for_endpoint_models = score_for_analysis.drop(columns=['CDDP_AUC'])
    primary = primary_analysis(score_for_endpoint_models, auc, rng)
    secondary = secondary_analysis(score_for_analysis.reset_index(), fixed_response, rng)
    (robustness, robustness_detail) = robustness_analysis(score_for_endpoint_models, auc, primary, rng)
    (random_summary, random_draws, random_sets) = matched_random_null(tumor_expression, ranks, signatures, score_for_endpoint_models, auc, rng)
    primary = classify_support(primary, robustness, random_summary)
    (positive, positive_scores) = positive_controls(tumor_expression, scores.index)
    primary.to_csv(OUT / '05_PRIMARY_RESULTS.tsv', sep='\t', index=False)
    secondary.to_csv(OUT / '06_SECONDARY_RESULTS.tsv', sep='\t', index=False, na_rep='NA')
    robustness.to_csv(OUT / '07_ROBUSTNESS_RESULTS.tsv', sep='\t', index=False)
    robustness_detail.to_csv(PROCESSED / 'leave_one_PDO_out_details.tsv', sep='\t', index=False)
    random_summary.to_csv(OUT / '08_RANDOM_NULL_RESULTS.tsv', sep='\t', index=False)
    random_draws.to_csv(PROCESSED / 'random_null_rho_draws.tsv', sep='\t', index=False)
    random_sets.to_csv(PROCESSED / 'random_null_gene_sets.tsv', sep='\t', index=False)
    positive.to_csv(OUT / '04_POSITIVE_CONTROL_RESULTS.tsv', sep='\t', index=False)
    positive_scores.to_csv(PROCESSED / 'positive_control_per_PDO_scores.tsv', sep='\t', index=True, index_label='sample_key')
    singlecell = pd.DataFrame([{'status': 'NOT_EVALUABLE_PUBLIC_DATA_UNAVAILABLE', 'reported_resistant_PDOs': 'ESCCO15;ESCCO17;ESCCO34', 'reported_sensitive_PDOs': 'ESCCO18;ESCCO20;ESCCO25', 'reported_n_resistant_PDO': 3, 'reported_n_sensitive_PDO': 3, 'public_matrix_found': False, 'reason': 'GSE269447/PRJNA1121932 expose 20 bulk RNA-seq samples only; no scRNA count matrix or scRNA SRA runs were publicly linked. Figure pixels were not digitized.', 'statistics_performed': False}])
    singlecell.to_csv(OUT / '09_SINGLECELL_PDO_RESULTS.tsv', sep='\t', index=False)
    make_figures(score_for_endpoint_models.reset_index(), auc, primary, secondary, robustness_detail, random_summary, random_draws)
    config = {'phase': 'Frozen RMP x ESCC PDO functional validation', 'seed': SEED, 'primary_RMPs': PRIMARY, 'primary_signature': 'Phase D frozen Top50', 'sensitivity_signature': 'Phase D frozen Top100', 'scoring': 'Phase E locked single-sample rank score: (mean_rank-(m+1)/2)/(N-m), bounded [0,1]', 'expression': 'author-provided TPM; no re-normalization; Ensembl GRCh38.105 IDs mapped to HGNC-approved symbols; duplicate canonical symbols collapsed by arithmetic mean', 'primary_endpoint': 'Spearman correlation, frozen Top50 score vs author-reported CDDP AUC; n=10; BH across C01/C02/C07', 'secondary': ['Mann-Whitney resistant vs sensitive, n=7 vs 8; rank-biserial effect', 'Spearman vs fixed-dose 2.46 uM 5-FU relative viability, n=12'], 'robustness': ['Top100', 'leave-one-PDO-out', f'{N_BOOT} pair-bootstrap resamples'], 'matched_random_null': f'{N_RANDOM} sets per primary RMP, gene-detection-rate decile matched, outcome-blind', 'identifier_info': identifier_info, 'no_outcome_driven_changes': True}
    (OUT / 'exact_configuration.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    env = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'matplotlib': mpl.__version__}
    (LOGS / 'environment.json').write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding='utf-8')
    write_hash_manifest()
    logger.info('Analysis complete. Primary results:\n%s', primary.to_string(index=False))
    logger.info('Positive controls:\n%s', positive.to_string(index=False))
    logger.info('Random-null summary:\n%s', random_summary.to_string(index=False))
if __name__ == '__main__':
    main()
