from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import csv
import gzip
import hashlib
import json
import math
import platform
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from scipy.stats import rankdata, spearmanr, wilcoxon
SEED = 20260827
BOOTSTRAPS = 2000
RNG = np.random.default_rng(SEED)
PROJECT_ROOT = INPUT_ROOT
OUT = PROJECT_ROOT / 'results' / 'bulk_context'
DISC = OUT / '01_dataset_discovery'
RAW = PROJECT_ROOT / 'data' / 'raw' / 'GSE53625'
PROC = OUT / '03_processed_spatial_data'
COV = OUT / '04_signature_coverage'
SCORE = OUT / '05_spatial_scoring'
STAT = OUT / '06_spatial_statistics'
CTX = OUT / '07_ecosystem_context'
SENS = OUT / '08_sensitivity_top100'
FIG = OUT / '09_figures'
TAB = OUT / '10_tables'
LOG = OUT / '13_logs'
for directory in (PROC, COV, SCORE, STAT, CTX, SENS, FIG, TAB, LOG):
    directory.mkdir(parents=True, exist_ok=True)
RMP_IDS = [f'RMP_C{i:02d}' for i in range(1, 8)]
PRIMARY = {'RMP_C01', 'RMP_C02', 'RMP_C04', 'RMP_C05', 'RMP_C07'}
EXPLORATORY = {'RMP_C03', 'RMP_C06'}
PHASE_D_TIER = {'RMP_C01': 'TIER_1', 'RMP_C02': 'TIER_1', 'RMP_C03': 'TIER_2_EXPLORATORY', 'RMP_C04': 'TIER_1', 'RMP_C05': 'TIER_1', 'RMP_C06': 'TIER_2_EXPLORATORY', 'RMP_C07': 'TIER_1'}
PHASE_D_CONFIDENCE = {'RMP_C01': 'MODERATE', 'RMP_C02': 'HIGH', 'RMP_C03': 'MODERATE', 'RMP_C04': 'MODERATE', 'RMP_C05': 'MODERATE', 'RMP_C06': 'MODERATE', 'RMP_C07': 'HIGH'}
PHASE_E_SUPPORT = {'RMP_C01': 'MODERATE', 'RMP_C02': 'MODERATE', 'RMP_C03': 'MODERATE_EXPLORATORY', 'RMP_C04': 'LIMITED_MIXED', 'RMP_C05': 'MODERATE', 'RMP_C06': 'STRONG_EXPLORATORY', 'RMP_C07': 'STRONG'}
CONTEXT_SIGNATURES = {'IMMUNE_TNK': ['PTPRC', 'CD3D', 'CD3E', 'TRAC', 'CD247', 'NKG7', 'GNLY', 'KLRD1', 'CD8A', 'CD8B'], 'MYELOID': ['LST1', 'TYROBP', 'FCER1G', 'CTSS', 'AIF1', 'LILRB1', 'CSTA', 'CYBA', 'CD68', 'CSF1R'], 'INTERFERON_CONTEXT': ['STAT1', 'STAT2', 'IRF1', 'IRF7', 'ISG15', 'MX1', 'MX2', 'OAS1', 'OAS2', 'IFIT1', 'IFIT2', 'IFIT3', 'CXCL10'], 'EPITHELIAL': ['EPCAM', 'KRT8', 'KRT18', 'KRT19', 'SFN', 'KRT5', 'KRT14', 'TP63'], 'DIFFERENTIATION_KERATINIZATION': ['KRT4', 'KRT13', 'KRT1', 'KRT10', 'IVL', 'FLG', 'LOR', 'TGM1', 'DSG1', 'DSC1', 'KRT16', 'KRT17'], 'FIBROBLAST_ECM': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'COL6A1', 'COL6A2', 'PDGFRA', 'FAP', 'COL5A1']}
DECLARED_ASSOCIATIONS = [('RMP_C07', 'IMMUNE_TNK', 'PRIMARY_C07'), ('RMP_C07', 'INTERFERON_CONTEXT', 'PRIMARY_C07'), ('RMP_C02', 'EPITHELIAL', 'PRIMARY_C02'), ('RMP_C02', 'DIFFERENTIATION_KERATINIZATION', 'PRIMARY_C02'), ('RMP_C05', 'IMMUNE_TNK', 'SUPPORTIVE'), ('RMP_C05', 'INTERFERON_CONTEXT', 'SUPPORTIVE'), ('RMP_C01', 'EPITHELIAL', 'SUPPORTIVE'), ('RMP_C04', 'EPITHELIAL', 'HETEROGENEITY_LIMITED'), ('RMP_C03', 'INTERFERON_CONTEXT', 'EXPLORATORY'), ('RMP_C06', 'FIBROBLAST_ECM', 'EXPLORATORY')]

def bh_fdr(pvalues: list[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if not len(vals):
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    q = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    restored = np.empty_like(q)
    restored[order] = q
    out[ok] = restored
    return out

def bootstrap_median_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    draws = RNG.choice(values, size=(BOOTSTRAPS, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return tuple(np.quantile(medians, [0.025, 0.975]))

def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    (x, y) = (x[ok], y[ok])
    if len(x) < 4:
        return (np.nan, np.nan)
    vals = []
    for _ in range(BOOTSTRAPS):
        idx = RNG.integers(0, len(x), len(x))
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        vals.append(spearmanr(x[idx], y[idx]).statistic)
    return tuple(np.quantile(vals, [0.025, 0.975])) if vals else (np.nan, np.nan)

def parse_gtf(gtf_path: Path) -> tuple[dict[str, str], set[str]]:
    transcript_to_gene: dict[str, str] = {}
    genes: set[str] = set()
    rx_tid = re.compile('transcript_id "([^"]+)"')
    rx_gene = re.compile('gene_name "([^"]+)"')
    with gzip.open(gtf_path, 'rt', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9:
                continue
            attrs = fields[8]
            gm = rx_gene.search(attrs)
            if not gm:
                continue
            gene = gm.group(1).upper()
            genes.add(gene)
            tm = rx_tid.search(attrs)
            if tm:
                transcript_to_gene[tm.group(1).split('.')[0]] = gene
    return (transcript_to_gene, genes)

def read_sdrf(path: Path, subseries: str) -> pd.DataFrame:
    frame = pd.read_csv(resolve_data(path), sep='\t', dtype=str, low_memory=False)
    frame['gsm'] = frame['Source Name'].str.extract('(GSM\\d+)', expand=False)
    frame['patient_id'] = frame['Characteristics [patient id]'].str.strip().str.lower()
    frame['organism_part'] = frame['Characteristics [organism part]'].str.lower()
    frame['tissue'] = np.where(frame['organism_part'].str.startswith('cancer tissue'), 'TUMOR', 'NORMAL')
    frame['processed_file'] = frame['Derived Array Data File']
    frame['subseries'] = subseries
    return frame[['gsm', 'patient_id', 'tissue', 'processed_file', 'subseries', 'organism_part']]

def extract_feature_mapping(raw_feature_path: Path, target_features: set[str], transcript_to_gene: dict[str, str], gene_symbols: set[str]) -> pd.DataFrame:
    rows = []
    header = None
    with raw_feature_path.open('r', encoding='utf-8', errors='replace', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        for row in reader:
            if not row:
                continue
            if row[0] == 'FEATURES':
                header = row[1:]
                continue
            if header is None or row[0] != 'DATA':
                continue
            record = dict(zip(header, row[1:]))
            feature = record.get('FeatureNum', '')
            if feature not in target_features:
                continue
            candidates = []
            for field in (record.get('GeneName', ''), record.get('SystematicName', ''), record.get('ProbeName', '')):
                for token in re.split('[|;,/\\s]+', field or ''):
                    token = token.strip().split('.')[0]
                    if not token:
                        continue
                    if token in transcript_to_gene:
                        candidates.append(transcript_to_gene[token])
                    elif token.upper() in gene_symbols:
                        candidates.append(token.upper())
            unique = sorted(set(candidates))
            rows.append({'feature_id': feature, 'probe_name': record.get('ProbeName', ''), 'systematic_name': record.get('SystematicName', ''), 'control_type': record.get('ControlType', ''), 'gene_symbol': unique[0] if len(unique) == 1 else '', 'mapping_status': 'UNIQUE' if len(unique) == 1 else 'AMBIGUOUS' if len(unique) > 1 else 'UNMAPPED'})
    return pd.DataFrame(rows).drop_duplicates('feature_id', keep='first')

def load_expression(zip_path: Path, metadata: pd.DataFrame) -> tuple[list[str], list[str], np.ndarray]:
    sample_order = metadata['gsm'].tolist()
    with zipfile.ZipFile(zip_path) as archive:
        available = set(archive.namelist())
        arrays = []
        feature_order = None
        for (i, record) in metadata.reset_index(drop=True).iterrows():
            filename = record['processed_file']
            if filename not in available:
                raise RuntimeError(f'Missing processed file in ZIP: {filename}')
            with archive.open(filename) as handle:
                table = pd.read_csv(resolve_data(handle), sep='\t', dtype={'Reporter Identifier': str})
            ids = table['Reporter Identifier'].astype(str).tolist()
            values = pd.to_numeric(table['VALUE'], errors='coerce').to_numpy(dtype=np.float32)
            if feature_order is None:
                feature_order = ids
            elif ids != feature_order:
                lookup = pd.Series(values, index=ids)
                values = lookup.reindex(feature_order).to_numpy(dtype=np.float32)
            arrays.append(values)
    return (feature_order or [], sample_order, np.column_stack(arrays))

def collapse_to_genes(feature_ids: list[str], feature_matrix: np.ndarray, mapping: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    gene_by_feature = mapping.set_index('feature_id')['gene_symbol'].to_dict()
    grouped: dict[str, list[int]] = defaultdict(list)
    for (idx, feature) in enumerate(feature_ids):
        gene = gene_by_feature.get(feature, '')
        if gene:
            grouped[gene].append(idx)
    genes = sorted(grouped)
    out = np.empty((len(genes), feature_matrix.shape[1]), dtype=np.float32)
    for (i, gene) in enumerate(genes):
        out[i, :] = np.nanmedian(feature_matrix[grouped[gene], :], axis=0)
    return (genes, out)

def percentile_rank_matrix(gene_matrix: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(gene_matrix, dtype=np.float32)
    for j in range(gene_matrix.shape[1]):
        values = gene_matrix[:, j]
        ok = np.isfinite(values)
        col = np.full(len(values), np.nan, dtype=np.float32)
        if ok.sum():
            col[ok] = rankdata(values[ok], method='average').astype(np.float32) / float(ok.sum())
        ranks[:, j] = col
    return ranks

def score_gene_sets(rank_matrix: np.ndarray, genes: list[str], sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_index = {g: i for (i, g) in enumerate(genes)}
    (score_rows, coverage_rows) = ({}, [])
    for (name, members) in sets.items():
        ordered = [g.upper() for g in members]
        detected = [g for g in ordered if g in gene_index]
        missing = [g for g in ordered if g not in gene_index]
        if detected:
            score_rows[name] = np.nanmean(rank_matrix[[gene_index[g] for g in detected], :], axis=0)
        else:
            score_rows[name] = np.full(rank_matrix.shape[1], np.nan)
        pct = 100 * len(detected) / len(ordered)
        label = 'GOOD' if pct >= 80 else 'ACCEPTABLE' if pct >= 70 else 'CAUTION' if pct >= 60 else 'LOW_COVERAGE'
        coverage_rows.append({'signature': name, 'total_frozen_genes': len(ordered), 'genes_detected': len(detected), 'genes_missing': len(missing), 'coverage_percent': pct, 'coverage_label': label, 'detected_gene_list': ';'.join(detected), 'missing_gene_list': ';'.join(missing)})
    return (pd.DataFrame(score_rows), pd.DataFrame(coverage_rows))

def association_row(data: pd.DataFrame, rmp: str, context: str, family: str, size: str='TOP50') -> dict:
    x = data[rmp].to_numpy(float)
    y = data[context].to_numpy(float)
    result = spearmanr(x, y, nan_policy='omit')
    (ci_low, ci_high) = bootstrap_spearman_ci(x, y)
    lopos = []
    for i in range(len(data)):
        keep = np.arange(len(data)) != i
        lopos.append(spearmanr(x[keep], y[keep], nan_policy='omit').statistic)
    subseries = {}
    for (sub, part) in data.groupby('subseries'):
        subseries[sub] = spearmanr(part[rmp], part[context], nan_policy='omit').statistic
    sign = np.sign(result.statistic) if np.isfinite(result.statistic) else 0
    return {'signature_size': size, 'RMP': rmp, 'context': context, 'hypothesis_family': family, 'patients': int(data[[rmp, context]].dropna().shape[0]), 'spearman_rho': result.statistic, 'ci_low': ci_low, 'ci_high': ci_high, 'nominal_p': result.pvalue, 'lopo_rho_min': np.nanmin(lopos), 'lopo_rho_max': np.nanmax(lopos), 'lopo_sign_consistency': float(np.mean(np.sign(lopos) == sign)) if sign else np.nan, 'GSE53622_rho': subseries.get('GSE53622', np.nan), 'GSE53624_rho': subseries.get('GSE53624', np.nan), 'subseries_direction_consistent': bool(all((np.sign(v) == sign for v in subseries.values() if np.isfinite(v)))) if sign else False}

def contextual_support(rmp: str, assoc: pd.DataFrame, coverage: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    cov = float(coverage.loc[(coverage['RMP'] == rmp) & (coverage['signature_size'] == 'TOP50'), 'coverage_percent'].iloc[0])
    a = assoc[(assoc['RMP'] == rmp) & (assoc['signature_size'] == 'TOP50')]
    sens = sensitivity.loc[sensitivity['RMP'] == rmp, 'top50_top100_rho'].iloc[0]
    if cov < 60 or a.empty:
        return 'NOT_EVALUABLE'
    positive = (a['spearman_rho'] > 0).mean()
    stable = (a['lopo_sign_consistency'] >= 0.95).all()
    strata = a['subseries_direction_consistent'].all()
    interval = (a['ci_low'] > 0).any()
    robust_size = sens >= 0.7
    if positive == 1 and stable and strata and interval and robust_size:
        return 'SUPPORTED'
    if positive >= 0.5 and stable and robust_size:
        return 'MIXED'
    return 'NOT_SUPPORTED'

def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f'{stem}.png', dpi=220, bbox_inches='tight')
    fig.savefig(FIG / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)

def main() -> None:
    log_lines = [f'seed={SEED}', f'bootstraps={BOOTSTRAPS}']
    zip_path = RAW / 'E-GEOD-53625.processed.1.zip'
    rescued_crosswalk = PROC / 'GPL18109_feature_to_HGNC_crosswalk.tsv'
    rescue_summary = OUT / '00_audit' / 'GPL18109_ANNOTATION_RESCUE_SUMMARY.json'
    for required in (zip_path, rescued_crosswalk, rescue_summary):
        if not required.exists():
            raise FileNotFoundError(required)
    rescue = json.loads(rescue_summary.read_text(encoding='utf-8'))
    if rescue.get('mapping_gate') != 'PASS':
        raise RuntimeError('GPL18109 annotation/mapping gate is not PASS')
    meta22 = read_sdrf(DISC / 'GSE53622_SDRF.txt', 'GSE53622')
    meta24 = read_sdrf(DISC / 'GSE53624_SDRF.txt', 'GSE53624')
    metadata = pd.concat([meta22, meta24], ignore_index=True)
    if metadata['gsm'].duplicated().any():
        raise RuntimeError('Duplicated GSM in subseries metadata')
    pair_counts = metadata.groupby(['patient_id', 'tissue']).size().unstack(fill_value=0)
    valid_patients = pair_counts.index[(pair_counts.get('TUMOR', 0) == 1) & (pair_counts.get('NORMAL', 0) == 1)]
    metadata = metadata[metadata['patient_id'].isin(valid_patients)].copy()
    metadata = metadata.sort_values(['patient_id', 'tissue']).reset_index(drop=True)
    if len(valid_patients) != 179 or len(metadata) != 358:
        raise RuntimeError(f'Expected 179 paired patients/358 samples, observed {len(valid_patients)}/{len(metadata)}')
    metadata.to_csv(PROC / 'GSE53625_sample_metadata.tsv', sep='\t', index=False)
    (feature_ids, sample_order, feature_matrix) = load_expression(zip_path, metadata)
    rescued = pd.read_csv(resolve_data(rescued_crosswalk), sep='\t', dtype=str, low_memory=False).fillna('')
    mapping = rescued.loc[rescued['gene_mapping_status'] == 'UNIQUE_ONE_TO_ONE', ['feature_number', 'hgnc_symbol']].rename(columns={'feature_number': 'feature_id', 'hgnc_symbol': 'gene_symbol'})
    mapping['mapping_status'] = 'UNIQUE_ONE_TO_ONE'
    mapping.to_csv(PROC / 'GSE53625_probe_to_gene_mapping.tsv', sep='\t', index=False)
    mapped = len(mapping)
    log_lines.append(f'processed_features={len(feature_ids)}')
    log_lines.append(f'uniquely_mapped_features={mapped}')
    (genes, gene_matrix) = collapse_to_genes(feature_ids, feature_matrix, mapping)
    del feature_matrix
    gene_table = pd.DataFrame(gene_matrix, index=genes, columns=sample_order)
    gene_table.index.name = 'gene_symbol'
    gene_table.to_csv(PROC / 'GSE53625_gene_expression.tsv', sep='\t')
    rank_matrix = percentile_rank_matrix(gene_matrix)
    log_lines.append(f'mapped_unique_genes={len(genes)}')
    rmp_sets: dict[str, list[str]] = {}
    coverage_parts = []
    score_parts = []
    for size in (50, 100):
        current = {}
        for rmp in RMP_IDS:
            path = PROJECT_ROOT / 'signatures' / f'{rmp}_signature_top{size}.txt'
            members = [line.strip().upper() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
            current[rmp] = members
            rmp_sets[f'{rmp}_TOP{size}'] = members
        (scores, coverage) = score_gene_sets(rank_matrix, genes, current)
        scores.columns = RMP_IDS
        scores['gsm'] = sample_order
        scores['signature_size'] = f'TOP{size}'
        score_parts.append(scores)
        coverage = coverage.rename(columns={'signature': 'RMP'})
        coverage['signature_size'] = f'TOP{size}'
        coverage_parts.append(coverage)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    coverage.to_csv(COV / 'bulk_signature_coverage_GSE53625.tsv', sep='\t', index=False)
    top50_union = set().union(*(set(rmp_sets[f'{r}_TOP50']) for r in RMP_IDS))
    (overlap_rows, cleaned_contexts) = ([], {})
    for (name, members) in CONTEXT_SIGNATURES.items():
        removed = sorted(set(members) & top50_union)
        retained = [g for g in members if g not in top50_union]
        cleaned_contexts[name] = retained
        overlap_rows.append({'context_signature': name, 'original_n': len(members), 'retained_n': len(retained), 'removed_from_TME_reference_due_to_overlap': ';'.join(removed) if removed else 'NONE', 'retained_genes': ';'.join(retained)})
    pd.DataFrame(overlap_rows).to_csv(CTX / 'context_signature_overlap_audit.tsv', sep='\t', index=False)
    (context_scores, context_coverage) = score_gene_sets(rank_matrix, genes, cleaned_contexts)
    context_scores['gsm'] = sample_order
    context_coverage.to_csv(CTX / 'context_signature_coverage.tsv', sep='\t', index=False)
    scores_all = pd.concat(score_parts, ignore_index=True)
    scores_all = scores_all.merge(metadata, on='gsm', how='left', validate='many_to_one')
    scores_all.to_csv(SCORE / 'GSE53625_sample_RMP_scores.tsv', sep='\t', index=False)
    context_scores = context_scores.merge(metadata, on='gsm', how='left', validate='one_to_one')
    context_scores.to_csv(CTX / 'GSE53625_sample_context_scores.tsv', sep='\t', index=False)
    top50 = scores_all[scores_all['signature_size'] == 'TOP50'].merge(context_scores.drop(columns=['patient_id', 'tissue', 'processed_file', 'subseries', 'organism_part']), on='gsm', how='left', validate='one_to_one')
    tumor50 = top50[top50['tissue'] == 'TUMOR'].copy().sort_values('patient_id')
    normal50 = top50[top50['tissue'] == 'NORMAL'].copy().sort_values('patient_id')
    if tumor50['patient_id'].tolist() != normal50['patient_id'].tolist():
        raise RuntimeError('Tumor/normal patient pairing mismatch')
    paired_rows = []
    paired_scores = tumor50[['patient_id', 'subseries']].copy()
    for rmp in RMP_IDS:
        delta = tumor50[rmp].to_numpy() - normal50[rmp].to_numpy()
        (ci_low, ci_high) = bootstrap_median_ci(delta)
        test = wilcoxon(delta, zero_method='wilcox', alternative='two-sided')
        paired_scores[f'{rmp}_TUMOR'] = tumor50[rmp].to_numpy()
        paired_scores[f'{rmp}_NORMAL'] = normal50[rmp].to_numpy()
        paired_scores[f'{rmp}_DELTA'] = delta
        paired_rows.append({'RMP': rmp, 'patients': len(delta), 'median_tumor_score': np.median(tumor50[rmp]), 'median_normal_score': np.median(normal50[rmp]), 'median_paired_delta': np.median(delta), 'ci_low': ci_low, 'ci_high': ci_high, 'positive_delta_fraction': np.mean(delta > 0), 'wilcoxon_p': test.pvalue})
    paired_results = pd.DataFrame(paired_rows)
    paired_results['wilcoxon_fdr'] = bh_fdr(paired_results['wilcoxon_p'].tolist())
    paired_results.to_csv(STAT / 'GSE53625_paired_tumor_normal_RMP_effects.tsv', sep='\t', index=False)
    paired_scores.to_csv(STAT / 'GSE53625_patient_paired_RMP_scores.tsv', sep='\t', index=False)
    paired_stability_rows = []
    for rmp in RMP_IDS:
        delta = paired_scores[f'{rmp}_DELTA'].to_numpy(float)
        overall_sign = np.sign(np.median(delta))
        lopo = np.array([np.median(np.delete(delta, i)) for i in range(len(delta))])
        sub = paired_scores.groupby('subseries')[f'{rmp}_DELTA'].median().to_dict()
        paired_stability_rows.append({'RMP': rmp, 'patients': len(delta), 'overall_median_delta': np.median(delta), 'lopo_median_min': lopo.min(), 'lopo_median_max': lopo.max(), 'lopo_direction_consistency': float(np.mean(np.sign(lopo) == overall_sign)), 'GSE53622_median_delta': sub.get('GSE53622', np.nan), 'GSE53624_median_delta': sub.get('GSE53624', np.nan), 'subseries_direction_consistent': bool(all((np.sign(v) == overall_sign for v in sub.values())))})
    paired_stability = pd.DataFrame(paired_stability_rows)
    paired_stability.to_csv(STAT / 'GSE53625_patient_level_paired_stability.tsv', sep='\t', index=False)
    association_rows = [association_row(tumor50, r, c, f, 'TOP50') for (r, c, f) in DECLARED_ASSOCIATIONS]
    assoc = pd.DataFrame(association_rows)
    assoc['BH_FDR'] = bh_fdr(assoc['nominal_p'].tolist())
    top100 = scores_all[scores_all['signature_size'] == 'TOP100'].merge(context_scores.drop(columns=['patient_id', 'tissue', 'processed_file', 'subseries', 'organism_part']), on='gsm', how='left', validate='one_to_one')
    tumor100 = top100[top100['tissue'] == 'TUMOR'].copy().sort_values('patient_id')
    association_rows100 = [association_row(tumor100, r, c, f, 'TOP100') for (r, c, f) in DECLARED_ASSOCIATIONS]
    assoc100 = pd.DataFrame(association_rows100)
    assoc100['BH_FDR'] = bh_fdr(assoc100['nominal_p'].tolist())
    associations = pd.concat([assoc, assoc100], ignore_index=True)
    associations.to_csv(CTX / 'GSE53625_RMP_context_associations.tsv', sep='\t', index=False)
    sensitivity_rows = []
    for rmp in RMP_IDS:
        rho = spearmanr(tumor50[rmp], tumor100[rmp], nan_policy='omit').statistic
        a50 = assoc[assoc['RMP'] == rmp].set_index('context')['spearman_rho']
        a100 = assoc100[assoc100['RMP'] == rmp].set_index('context')['spearman_rho']
        common = sorted(set(a50.index) & set(a100.index))
        directions = [np.sign(a50[c]) == np.sign(a100[c]) for c in common]
        sensitivity_rows.append({'RMP': rmp, 'top50_top100_rho': rho, 'declared_associations_compared': len(common), 'association_direction_consistency': float(np.mean(directions)) if directions else np.nan, 'sensitivity_label': 'ROBUST_TO_SIGNATURE_SIZE' if rho >= 0.7 and all(directions) else 'MIXED_SIZE_SENSITIVITY'})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(SENS / 'GSE53625_top50_top100_sensitivity.tsv', sep='\t', index=False)
    distinctness_rows = []
    for (left, right) in [('RMP_C05', 'RMP_C07'), ('RMP_C01', 'RMP_C04')]:
        rho = spearmanr(tumor50[left], tumor50[right]).statistic
        distinctness_rows.append({'program_A': left, 'program_B': right, 'patients': len(tumor50), 'tumor_score_spearman_rho': rho, 'near_redundant_threshold': 0.9, 'near_redundant': bool(abs(rho) >= 0.9), 'interpretation': 'NEAR_REDUNDANT' if abs(rho) >= 0.9 else 'DISTINCT_NOT_NEAR_REDUNDANT'})
    distinctness = pd.DataFrame(distinctness_rows)
    distinctness.to_csv(STAT / 'GSE53625_RMP_distinctness.tsv', sep='\t', index=False)
    coverage['RMP_class'] = np.where(coverage['RMP'].isin(PRIMARY), 'PRIMARY', 'EXPLORATORY')
    bulk_rows = []
    for rmp in RMP_IDS:
        cov50 = coverage[(coverage['RMP'] == rmp) & (coverage['signature_size'] == 'TOP50')].iloc[0]
        paired = paired_results[paired_results['RMP'] == rmp].iloc[0]
        support = contextual_support(rmp, associations, coverage, sensitivity)
        a = assoc[assoc['RMP'] == rmp]
        bulk_rows.append({'RMP': rmp, 'Phase_D_tier': PHASE_D_TIER[rmp], 'Phase_D_annotation_confidence': PHASE_D_CONFIDENCE[rmp], 'Phase_E_external_support': PHASE_E_SUPPORT[rmp], 'bulk_cohort': 'GSE53625', 'patients': 179, 'paired_tumor_normal_samples': 358, 'Top50_coverage': cov50['coverage_percent'], 'paired_tumor_normal_median_delta': paired['median_paired_delta'], 'paired_delta_ci': f"[{paired['ci_low']:.4f}, {paired['ci_high']:.4f}]", 'declared_context_associations': ';'.join((f'{x.context}:rho={x.spearman_rho:.3f}' for x in a.itertuples())), 'patient_LOPO_stability': 'STABLE' if (a['lopo_sign_consistency'] >= 0.95).all() else 'UNSTABLE', 'subseries_direction_consistency': 'YES' if a['subseries_direction_consistent'].all() else 'NO', 'Top50_Top100_consistency': sensitivity.loc[sensitivity['RMP'] == rmp, 'sensitivity_label'].iloc[0], 'Phase_FB_bulk_context_interpretation': support, 'evidence_role': 'PRIMARY' if rmp in PRIMARY else 'EXPLORATORY'})
    bulk_summary = pd.DataFrame(bulk_rows)
    bulk_summary.to_csv(TAB / 'Table_F1_bulk_contextualization_summary.tsv', sep='\t', index=False)
    spatial_rows = []
    for rmp in RMP_IDS:
        cov50 = coverage[(coverage['RMP'] == rmp) & (coverage['signature_size'] == 'TOP50')].iloc[0]
        spatial_rows.append({'RMP': rmp, 'Phase_D_tier': PHASE_D_TIER[rmp], 'Phase_D_annotation_confidence': PHASE_D_CONFIDENCE[rmp], 'Phase_E_external_support': PHASE_E_SUPPORT[rmp], 'spatial_cohort': 'NONE_ELIGIBLE', 'patients': 0, 'sections': 0, 'Top50_coverage': 'NOT_EVALUATED_IN_SPATIAL_DATA', 'spatial_autocorrelation': 'NOT_EVALUABLE', 'patient_consistency': 'NOT_EVALUABLE', 'tumor_localization': 'NOT_EVALUABLE', 'ecosystem_association': 'NOT_EVALUABLE', 'Top50_Top100_consistency': 'NOT_EVALUABLE_IN_SPATIAL_DATA', 'Phase_F_interpretation': 'NOT EVALUABLE — no public ESCC dataset passed the expression/coordinate/provenance gate', 'fallback_bulk_Top50_coverage': cov50['coverage_percent'], 'fallback_bulk_context': bulk_summary.loc[bulk_summary['RMP'] == rmp, 'Phase_FB_bulk_context_interpretation'].iloc[0]})
    pd.DataFrame(spatial_rows).to_csv(TAB / 'Table_F1_spatial_validation_summary.tsv', sep='\t', index=False)
    sns.set_theme(style='whitegrid', context='talk')
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(['Spatial\neligible', 'Bulk fallback\npatients', 'Bulk fallback\nsamples'], [0, 179, 358], color=['#bdbdbd', '#2a6f97', '#61a5c2'])
    axes[0].set_title('Phase F data gate')
    axes[0].set_ylabel('Count')
    cov_plot = coverage[coverage['signature_size'] == 'TOP50'].copy()
    colors = ['#d95f02' if r in EXPLORATORY else '#1b9e77' for r in cov_plot['RMP']]
    axes[1].bar(cov_plot['RMP'], cov_plot['coverage_percent'], color=colors)
    axes[1].axhline(80, ls='--', lw=1, color='black')
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel('Top50 coverage (%)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_title('GSE53625 frozen-signature coverage')
    fig.suptitle('Figure F1 — Spatial gate and orthogonal bulk fallback', y=1.04)
    save_figure(fig, 'Figure_F1_study_architecture_and_coverage')
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 5.5))
    for (ax, rmp, color) in zip(axes, ['RMP_C07', 'RMP_C02'], ['#3a86ff', '#ff9f1c']):
        plot_df = pd.DataFrame({'Normal': normal50[rmp].to_numpy(), 'Tumor': tumor50[rmp].to_numpy()})
        for (_, row) in plot_df.iterrows():
            ax.plot([0, 1], row.values, color='#a6a6a6', alpha=0.2, lw=0.6)
        ax.boxplot([plot_df['Normal'], plot_df['Tumor']], positions=[0, 1], widths=0.45, patch_artist=True, boxprops={'facecolor': color, 'alpha': 0.55}, medianprops={'color': 'black'})
        ax.set_xticks([0, 1], ['Adjacent normal', 'Tumor'])
        ax.set_ylabel('Frozen Top50 rank score')
        ax.set_title(rmp)
    fig.suptitle('Figure F2 — Patient-paired primary RMP contextualization (n=179)')
    save_figure(fig, 'Figure_F2_primary_RMP_paired_bulk_context')
    (fig, ax) = plt.subplots(figsize=(10, 7))
    plot_assoc = assoc.copy().iloc[::-1]
    y = np.arange(len(plot_assoc))
    ax.errorbar(plot_assoc['spearman_rho'], y, xerr=[plot_assoc['spearman_rho'] - plot_assoc['ci_low'], plot_assoc['ci_high'] - plot_assoc['spearman_rho']], fmt='o', color='#2a6f97', ecolor='#89c2d9', capsize=3)
    ax.axvline(0, color='black', lw=1)
    ax.set_yticks(y, [f'{r.RMP} ↔ {r.context}' for r in plot_assoc.itertuples()])
    ax.set_xlabel('Patient-level Spearman rho (bootstrap 95% CI)')
    ax.set_title('Figure F3 — Pre-specified tumor ecosystem context associations')
    save_figure(fig, 'Figure_F3_patient_level_context_associations')
    (fig, axes) = plt.subplots(1, 2, figsize=(13, 6.4))
    for (ax, (left, right), color) in zip(axes, [('RMP_C05', 'RMP_C07'), ('RMP_C01', 'RMP_C04')], ['#7b2cbf', '#2d6a4f']):
        sns.regplot(data=tumor50, x=left, y=right, ax=ax, scatter_kws={'s': 25, 'alpha': 0.65}, line_kws={'color': color}, color=color)
        rho = spearmanr(tumor50[left], tumor50[right]).statistic
        ax.set_title(f'{left} vs {right}\nSpearman rho={rho:.3f}')
    fig.suptitle('Figure F4 — Program distinctness in independent patient tumors', y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save_figure(fig, 'Figure_F4_program_distinctness')
    sens_matrix = sensitivity.set_index('RMP')[['top50_top100_rho']]
    (fig, ax) = plt.subplots(figsize=(4.5, 6))
    sns.heatmap(sens_matrix, annot=True, fmt='.2f', vmin=0, vmax=1, cmap='viridis', ax=ax, cbar_kws={'label': 'Spearman rho'})
    ax.set_title('Supplementary — Top50/Top100 score concordance')
    save_figure(fig, 'Supplementary_F1_top50_top100_sensitivity')
    software = {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'matplotlib': plt.matplotlib.__version__, 'seaborn': sns.__version__, 'seed': SEED, 'bootstraps': BOOTSTRAPS}
    (OUT / '00_audit' / 'phase_fb_software_versions.json').write_text(json.dumps(software, indent=2), encoding='utf-8')
    (LOG / 'phase_fb_bulk_contextualization.log').write_text('\n'.join(log_lines) + '\nstatus=PASS\n', encoding='utf-8')
    print(json.dumps({'patients': 179, 'samples': 358, 'processed_features': len(feature_ids), 'mapped_genes': len(genes), 'top50_coverage_min': float(coverage[coverage.signature_size == 'TOP50'].coverage_percent.min()), 'bulk_interpretations': dict(zip(bulk_summary.RMP, bulk_summary.Phase_FB_bulk_context_interpretation))}, indent=2))
if __name__ == '__main__':
    main()
