from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import json
import math
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr, wilcoxon
PROJECT_ROOT = INPUT_ROOT
PHASE_E = PROJECT_ROOT / 'results' / 'external_reproducibility'
PHASE_D = PROJECT_ROOT / 'signatures'
SEED = 20260827
RMP_IDS = [f'RMP_C{i:02d}' for i in range(1, 8)]
PRIMARY = {'RMP_C01', 'RMP_C02', 'RMP_C04', 'RMP_C05', 'RMP_C07'}
EXPLORATORY = {'RMP_C03', 'RMP_C06'}

def bh_fdr(values: pd.Series) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out

def load_signatures() -> tuple[dict[str, list[str]], pd.DataFrame]:
    signatures: dict[str, list[str]] = {}
    for rmp_id in RMP_IDS:
        for size in (50, 100):
            signatures[f'{rmp_id}_TOP{size}'] = [x.strip() for x in (PHASE_D / f'{rmp_id}_signature_top{size}.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
    frozen = pd.read_csv(resolve_data(PHASE_D / 'rmp_frozen_signatures.tsv'), sep='\t')
    return (signatures, frozen)

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

def patient_pseudobulk(data: ad.AnnData) -> tuple[list[str], sparse.csr_matrix, np.ndarray]:
    patients = sorted(data.obs.patient_id.astype(str).unique())
    groups = []
    for patient in patients:
        mask = data.obs.patient_id.astype(str).eq(patient).to_numpy()
        groups.append(sparse.csr_matrix(data.X[mask]).sum(axis=0))
    counts = sparse.vstack([sparse.csr_matrix(x) for x in groups]).tocsr()
    totals = np.asarray(counts.sum(axis=1)).ravel()
    log_cpm = counts.multiply(np.divide(10000.0, totals, out=np.zeros_like(totals), where=totals > 0)[:, None]).toarray()
    log_cpm = np.log1p(log_cpm)
    ranks = np.apply_along_axis(rankdata, 1, counts.toarray(), method='average')
    return (patients, counts, log_cpm, ranks)

def rank_score_from_ranks(ranks: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n_genes = ranks.shape[1]
    m = len(indices)
    if m == 0 or n_genes <= m:
        return np.full(ranks.shape[0], np.nan)
    minimum = (m + 1.0) / 2.0
    return np.clip((ranks[:, indices].mean(axis=1) - minimum) / (n_genes - m), 0, 1)

def matched_sets(signature_indices: np.ndarray, bins: np.ndarray, n_permutations: int, rng: np.random.Generator) -> list[np.ndarray]:
    signature_set = set(signature_indices.tolist())
    bin_counts = pd.Series(bins[signature_indices]).value_counts().to_dict()
    pools = {int(bin_id): np.asarray([i for i in np.where(bins == bin_id)[0] if i not in signature_set], dtype=int) for bin_id in bin_counts}
    sets = []
    for _ in range(n_permutations):
        selected = []
        for (bin_id, count) in sorted(bin_counts.items()):
            pool = pools[int(bin_id)]
            if len(pool) < count:
                raise RuntimeError(f'Matched-null pool too small for detection bin {bin_id}')
            selected.extend(rng.choice(pool, size=int(count), replace=False).tolist())
        sets.append(np.asarray(selected, dtype=int))
    return sets

def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int=2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return (float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975)))

def coherence_metrics(log_cpm: np.ndarray, patients: list[str], indices: np.ndarray, genes: list[str], frozen: pd.DataFrame, rmp_id: str, signature_type: str) -> tuple[dict[str, float | str], list[dict[str, object]]]:
    rank_lookup = frozen.loc[frozen.rmp_id.eq(rmp_id) & frozen.signature_type.eq(signature_type), ['gene', 'consensus_rank']].drop_duplicates('gene').set_index('gene')['consensus_rank'].to_dict()
    ordered_genes = [g for g in genes if g in rank_lookup]
    if len(ordered_genes) != len(indices):
        raise RuntimeError(f'Frozen rank alignment failure: {rmp_id} {signature_type}')
    frozen_order = -np.asarray([rank_lookup[g] for g in ordered_genes], dtype=float)
    patient_rhos = []
    rows = []
    for (i, patient) in enumerate(patients):
        rho = float(spearmanr(log_cpm[i, indices], frozen_order).statistic)
        patient_rhos.append(rho)
        rows.append({'patient_id': patient, 'rank_structure_rho': rho})
    gene_matrix = log_cpm[:, indices]
    variable = np.nanstd(gene_matrix, axis=0) > 0
    usable = int(variable.sum())
    if usable >= 2:
        correlation = spearmanr(gene_matrix[:, variable], axis=0).statistic
        if np.ndim(correlation) == 0:
            median_gene_rho = float(correlation)
        else:
            upper = np.asarray(correlation)[np.triu_indices(usable, 1)]
            median_gene_rho = float(np.nanmedian(upper))
    else:
        median_gene_rho = np.nan
    median_structure = float(np.nanmedian(patient_rhos))
    positive_fraction = float(np.mean(np.asarray(patient_rhos) > 0))
    if median_structure >= 0.15 and positive_fraction > 0.5 and (median_gene_rho > 0):
        label = 'COHERENT'
    elif median_structure > 0 or median_gene_rho > 0:
        label = 'MIXED'
    else:
        label = 'LOW'
    return ({'median_rank_structure_rho': median_structure, 'positive_patient_structure_fraction': positive_fraction, 'median_gene_gene_rho': median_gene_rho, 'coherence_usable_genes': usable, 'coherence_label': label}, rows)

def lopo_metrics(patient_z: np.ndarray, patients: list[str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    patient_z = np.asarray(patient_z, dtype=float)
    full = float(np.median(patient_z))
    records = []
    lopo_values = []
    for (i, patient) in enumerate(patients):
        leave = np.delete(patient_z, i)
        value = float(np.median(leave))
        lopo_values.append(value)
        records.append({'left_out_patient': patient, 'full_patient_median_z': full, 'lopo_patient_median_z': value, 'absolute_change': abs(value - full), 'sign_reversal': bool(full * value < 0)})
    maximum_change = float(np.max(np.abs(np.asarray(lopo_values) - full)))
    iqr = float(np.quantile(patient_z, 0.75) - np.quantile(patient_z, 0.25))
    positive = np.maximum(patient_z, 0)
    dominance = float(positive.max() / positive.sum()) if positive.sum() > 0 else 0.0
    sign_reversal = any((x['sign_reversal'] for x in records))
    major = sign_reversal or maximum_change > max(1.0, iqr) or dominance >= 0.8
    return ({'full_patient_median_z': full, 'lopo_max_absolute_change': maximum_change, 'patient_z_iqr': iqr, 'largest_patient_positive_z_fraction': dominance, 'major_patient_instability': bool(major), 'patient_stability': 'MAJOR_INSTABILITY' if major else 'STABLE'}, records)

def interpret_cohort(row: pd.Series) -> str:
    if row.coverage_percent < 60 or row.patients_evaluable < 3:
        return 'NOT_EVALUABLE'
    if row.detectability == 'NOT_DETECTED' and row.coherence_label == 'LOW':
        return 'NOT_SUPPORTED_IN_THIS_COHORT'
    positives = sum([row.detectability == 'DETECTED', row.coherence_label == 'COHERENT', row.signature_size_consistency == 'ROBUST_TO_SIGNATURE_SIZE'])
    if row.coverage_percent >= 80 and row.detectability == 'DETECTED' and (not row.major_patient_instability) and (row.coherence_label == 'COHERENT') and (row.signature_size_consistency == 'ROBUST_TO_SIGNATURE_SIZE'):
        return 'STRONG_EXTERNAL_SUPPORT'
    if row.coverage_percent >= 70 and (not row.major_patient_instability) and (positives >= 2):
        return 'MODERATE_EXTERNAL_SUPPORT'
    return 'LIMITED_MIXED_SUPPORT'

def process_cohort(cohort: str, data: ad.AnnData, signatures: dict[str, list[str]], frozen: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED + sum((ord(x) for x in cohort)))
    cell_scores = cell_rank_scores(data.X, data.var_names, signatures)
    cell_scores.insert(0, 'cell_id', data.obs_names.astype(str))
    cell_scores.insert(1, 'patient_id', data.obs.patient_id.astype(str).to_numpy())
    cell_scores.insert(2, 'sample_id', data.obs.sample_id.astype(str).to_numpy())
    cell_scores.insert(3, 'cohort', cohort)
    for key in signatures:
        threshold = float(cell_scores[key].quantile(0.75))
        cell_scores[f'{key}_high'] = cell_scores[key] > threshold
    cell_scores.to_csv(PHASE_E / '05_program_scoring' / f'{cohort}_cell_program_scores.tsv', sep='\t', index=False)
    (patients, pb_counts, log_cpm, pb_ranks) = patient_pseudobulk(data)
    var_names = pd.Index(data.var_names.astype(str))
    name_to_idx = {g: i for (i, g) in enumerate(var_names)}
    detection = np.asarray((sparse.csr_matrix(data.X) > 0).mean(axis=0)).ravel()
    bins = pd.qcut(pd.Series(detection).rank(method='first'), 10, labels=False).to_numpy()
    detected_global = detection > 0
    metric_rows = []
    patient_rows = []
    lopo_rows = []
    coherence_rows = []
    null_rows = []
    for (key, genes_all) in signatures.items():
        (rmp_id, signature_type) = key.rsplit('_', 1)
        genes = [g for g in genes_all if g in name_to_idx and detected_global[name_to_idx[g]]]
        indices = np.asarray([name_to_idx[g] for g in genes], dtype=int)
        observed_pb = rank_score_from_ranks(pb_ranks, indices)
        random_indices = matched_sets(indices, bins, 1000, rng)
        random_scores = np.vstack([rank_score_from_ranks(pb_ranks, x) for x in random_indices])
        null_mean = random_scores.mean(axis=0)
        null_sd = random_scores.std(axis=0, ddof=1)
        patient_z = np.divide(observed_pb - null_mean, null_sd, out=np.zeros_like(observed_pb), where=null_sd > 0)
        for (permutation, random_set) in enumerate(random_indices, start=1):
            null_rows.append({'cohort': cohort, 'rmp_id': rmp_id, 'signature_type': signature_type, 'permutation': permutation, 'genes': ';'.join(var_names[random_set].tolist())})
        group = cell_scores.groupby('patient_id', sort=True)[key]
        medians = group.median().reindex(patients)
        means = group.mean().reindex(patients)
        upper = group.quantile(0.75).reindex(patients)
        high_fraction = cell_scores.groupby('patient_id', sort=True)[f'{key}_high'].mean().reindex(patients)
        cell_counts = cell_scores.groupby('patient_id', sort=True).size().reindex(patients)
        for (i, patient) in enumerate(patients):
            patient_rows.append({'cohort': cohort, 'rmp_id': rmp_id, 'signature_type': signature_type, 'patient_id': patient, 'malignant_cells': int(cell_counts.loc[patient]), 'median_cell_score': float(medians.loc[patient]), 'mean_cell_score': float(means.loc[patient]), 'upper_quartile_cell_score': float(upper.loc[patient]), 'fraction_high_scoring_cells': float(high_fraction.loc[patient]), 'pseudobulk_rank_score': float(observed_pb[i]), 'matched_null_mean': float(null_mean[i]), 'matched_null_sd': float(null_sd[i]), 'matched_null_z': float(patient_z[i])})
        cover = coverage.loc[coverage.cohort.eq(cohort) & coverage.rmp_id.eq(rmp_id) & coverage.signature_type.eq(signature_type)].iloc[0]
        positive_fraction = float(np.mean(patient_z > 0))
        median_z = float(np.median(patient_z))
        if cover.coverage_percent < 60 or len(patients) < 3:
            detectability = 'NOT_EVALUABLE'
        elif median_z > 0 and positive_fraction >= 0.7 and (cover.coverage_percent >= 70):
            detectability = 'DETECTED'
        elif median_z > 0 or positive_fraction >= 0.5:
            detectability = 'MIXED'
        else:
            detectability = 'NOT_DETECTED'
        (ci_low, ci_high) = bootstrap_median_ci(patient_z, rng)
        try:
            p_value = float(wilcoxon(patient_z, alternative='two-sided', zero_method='wilcox').pvalue)
        except ValueError:
            p_value = 1.0
        (stability, lopo) = lopo_metrics(patient_z, patients)
        for record in lopo:
            record.update({'cohort': cohort, 'rmp_id': rmp_id, 'signature_type': signature_type})
            lopo_rows.append(record)
        (coherence, patient_coherence) = coherence_metrics(log_cpm, patients, indices, genes, frozen, rmp_id, signature_type)
        for record in patient_coherence:
            record.update({'cohort': cohort, 'rmp_id': rmp_id, 'signature_type': signature_type})
            coherence_rows.append(record)
        metric_rows.append({'cohort': cohort, 'rmp_id': rmp_id, 'rmp_role': 'PRIMARY' if rmp_id in PRIMARY else 'EXPLORATORY', 'signature_type': signature_type, 'coverage_percent': float(cover.coverage_percent), 'coverage_label': cover.coverage_label, 'patients_evaluable': len(patients), 'malignant_cells': data.n_obs, 'median_patient_z': median_z, 'median_patient_z_ci_low': ci_low, 'median_patient_z_ci_high': ci_high, 'positive_patient_fraction': positive_fraction, 'detectability': detectability, 'wilcoxon_p': p_value, **stability, **coherence})
    null_table = pd.DataFrame(null_rows)
    null_table.to_csv(PHASE_E / '05_program_scoring' / f'{cohort}_matched_null_gene_sets.tsv', sep='\t', index=False)
    return (pd.DataFrame(metric_rows), pd.DataFrame(patient_rows), pd.DataFrame(lopo_rows), pd.DataFrame(coherence_rows), cell_scores)

def signature_size_consistency(patient_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ((cohort, rmp_id), group) in patient_table.groupby(['cohort', 'rmp_id']):
        pivot = group.pivot(index='patient_id', columns='signature_type', values='median_cell_score')
        rho = float(spearmanr(pivot['TOP50'], pivot['TOP100']).statistic)
        if np.isfinite(rho) and rho >= 0.7:
            label = 'ROBUST_TO_SIGNATURE_SIZE'
        elif np.isfinite(rho) and rho > 0:
            label = 'MIXED_SIZE_SENSITIVITY'
        else:
            label = 'NOT_ROBUST_TO_SIGNATURE_SIZE'
        rows.append({'cohort': cohort, 'rmp_id': rmp_id, 'top50_top100_patient_rho': rho, 'signature_size_consistency': label})
    return pd.DataFrame(rows)

def distinctness_tables(cell_tables: dict[str, pd.DataFrame], patient_table: pd.DataFrame, signatures: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed_pairs = [('RMP_C01', 'RMP_C04'), ('RMP_C05', 'RMP_C07')]
    rows = []
    cell_correlations = []
    patient_correlations = []
    for (cohort, cell) in cell_tables.items():
        top50_cols = [f'{r}_TOP50' for r in RMP_IDS]
        corr = cell[top50_cols].corr(method='spearman')
        for a in RMP_IDS:
            for b in RMP_IDS:
                cell_correlations.append({'cohort': cohort, 'program_a': a, 'program_b': b, 'spearman_rho': float(corr.loc[f'{a}_TOP50', f'{b}_TOP50'])})
        p = patient_table.loc[patient_table.cohort.eq(cohort) & patient_table.signature_type.eq('TOP50')].pivot(index='patient_id', columns='rmp_id', values='median_cell_score')
        pcorr = p.corr(method='spearman')
        for a in RMP_IDS:
            for b in RMP_IDS:
                patient_correlations.append({'cohort': cohort, 'program_a': a, 'program_b': b, 'spearman_rho': float(pcorr.loc[a, b])})
        for (a, b) in fixed_pairs:
            cell_rho = float(corr.loc[f'{a}_TOP50', f'{b}_TOP50'])
            patient_rho = float(pcorr.loc[a, b])
            (genes_a, genes_b) = (set(signatures[f'{a}_TOP50']), set(signatures[f'{b}_TOP50']))
            jaccard = len(genes_a & genes_b) / len(genes_a | genes_b)
            near = cell_rho >= 0.9 and patient_rho >= 0.9 and (jaccard >= 0.8)
            rows.append({'cohort': cohort, 'program_a': a, 'program_b': b, 'cell_level_spearman': cell_rho, 'patient_level_spearman': patient_rho, 'top50_jaccard': jaccard, 'near_redundancy_review': 'YES' if near else 'NO', 'interpretation': 'NEAR_REDUNDANCY_REVIEW' if near else 'PRESERVED_NONIDENTICAL_STRUCTURE'})
    return (pd.DataFrame(rows), pd.DataFrame(cell_correlations), pd.DataFrame(patient_correlations))

def localization_gse203(signatures: dict[str, list[str]]) -> pd.DataFrame:
    data = ad.read_h5ad(PHASE_E / '03_processed_external_data' / 'GSE203067_primary_all_qc.h5ad')
    primary_signatures = {f'{r}_TOP50': signatures[f'{r}_TOP50'] for r in sorted(PRIMARY)}
    scores = cell_rank_scores(data.X, data.var_names, primary_signatures)
    scores['cell_type'] = data.obs.cell_type_inference.astype(str).to_numpy()
    scores['patient_id'] = data.obs.patient_id.astype(str).to_numpy()
    rows = []
    for ((cell_type, patient), group) in scores.groupby(['cell_type', 'patient_id']):
        for rmp_id in sorted(PRIMARY):
            values = group[f'{rmp_id}_TOP50']
            rows.append({'cohort': 'GSE203067', 'cell_type': cell_type, 'patient_id': patient, 'rmp_id': rmp_id, 'cells': len(values), 'median_score': float(values.median()), 'mean_score': float(values.mean()), 'q25': float(values.quantile(0.25)), 'q75': float(values.quantile(0.75))})
    return pd.DataFrame(rows)

def main() -> None:
    for directory in (PHASE_E / '05_program_scoring', PHASE_E / '06_patient_level_validation', PHASE_E / '07_coherence_and_distinctness', PHASE_E / '08_sensitivity_top100', PHASE_E / '10_tables'):
        directory.mkdir(parents=True, exist_ok=True)
    (signatures, frozen) = load_signatures()
    coverage = pd.read_csv(resolve_data(PHASE_E / '04_signature_coverage' / 'signature_coverage_by_cohort.tsv'), sep='\t')
    all_metrics = []
    all_patients = []
    all_lopo = []
    all_coherence = []
    cell_tables: dict[str, pd.DataFrame] = {}
    for cohort in ('GSE199654', 'GSE203067'):
        data = ad.read_h5ad(PHASE_E / '03_processed_external_data' / f'{cohort}_malignant_ready.h5ad')
        (metrics, patients, lopo, coherence, cells) = process_cohort(cohort, data, signatures, frozen, coverage)
        all_metrics.append(metrics)
        all_patients.append(patients)
        all_lopo.append(lopo)
        all_coherence.append(coherence)
        cell_tables[cohort] = cells
    metrics = pd.concat(all_metrics, ignore_index=True)
    patients = pd.concat(all_patients, ignore_index=True)
    lopo = pd.concat(all_lopo, ignore_index=True)
    coherence_patient = pd.concat(all_coherence, ignore_index=True)
    metrics['wilcoxon_fdr'] = metrics.groupby(['cohort', 'signature_type'])['wilcoxon_p'].transform(lambda x: bh_fdr(x))
    consistency = signature_size_consistency(patients)
    metrics = metrics.merge(consistency, on=['cohort', 'rmp_id'], how='left')
    metrics['cohort_validation_interpretation'] = metrics.apply(interpret_cohort, axis=1)
    metrics.to_csv(PHASE_E / '10_tables' / 'program_validation_metrics.tsv', sep='\t', index=False)
    patients.to_csv(PHASE_E / '06_patient_level_validation' / 'patient_program_scores.tsv', sep='\t', index=False)
    lopo.to_csv(PHASE_E / '06_patient_level_validation' / 'patient_level_stability.tsv', sep='\t', index=False)
    coherence_patient.to_csv(PHASE_E / '07_coherence_and_distinctness' / 'patient_signature_rank_coherence.tsv', sep='\t', index=False)
    consistency.to_csv(PHASE_E / '08_sensitivity_top100' / 'top50_top100_consistency.tsv', sep='\t', index=False)
    (distinct, cell_corr, patient_corr) = distinctness_tables(cell_tables, patients, signatures)
    distinct.to_csv(PHASE_E / '07_coherence_and_distinctness' / 'program_distinctness.tsv', sep='\t', index=False)
    cell_corr.to_csv(PHASE_E / '07_coherence_and_distinctness' / 'cell_program_score_correlations.tsv', sep='\t', index=False)
    patient_corr.to_csv(PHASE_E / '07_coherence_and_distinctness' / 'patient_program_score_correlations.tsv', sep='\t', index=False)
    localization = localization_gse203(signatures)
    localization.to_csv(PHASE_E / '07_coherence_and_distinctness' / 'cell_type_localization_GSE203067.tsv', sep='\t', index=False)
    top50 = metrics.loc[metrics.signature_type.eq('TOP50')].copy()
    overall_rows = []
    order = {'NOT_SUPPORTED_IN_THIS_COHORT': 0, 'LIMITED_MIXED_SUPPORT': 1, 'MODERATE_EXTERNAL_SUPPORT': 2, 'STRONG_EXTERNAL_SUPPORT': 3}
    for rmp_id in RMP_IDS:
        group = top50.loc[top50.rmp_id.eq(rmp_id)]
        evaluable = group.loc[~group.cohort_validation_interpretation.eq('NOT_EVALUABLE')]
        values = [order[x] for x in evaluable.cohort_validation_interpretation]
        if not values:
            overall = 'NOT_EVALUABLE'
        elif all((v == 0 for v in values)):
            overall = 'NOT_SUPPORTED'
        elif len(values) == 2 and min(values) >= 2 and (max(values) >= 3):
            overall = 'STRONG'
        elif len(values) == 2 and min(values) >= 1 and (max(values) >= 2):
            overall = 'MODERATE'
        elif len(values) == 1 and values[0] >= 3:
            overall = 'MODERATE'
        else:
            overall = 'LIMITED'
        if rmp_id in EXPLORATORY:
            overall_field = 'exploratory_support'
        else:
            overall_field = 'external_validation'
        overall_rows.append({'rmp_id': rmp_id, 'rmp_role': 'PRIMARY' if rmp_id in PRIMARY else 'EXPLORATORY', overall_field: overall, 'overall_interpretation': overall, 'cohorts_evaluable': len(evaluable)})
    overall_table = pd.DataFrame(overall_rows)
    overall_table.to_csv(PHASE_E / '10_tables' / 'overall_rmp_external_validation.tsv', sep='\t', index=False)
    print(top50[['cohort', 'rmp_id', 'coverage_percent', 'median_patient_z', 'detectability', 'patient_stability', 'coherence_label', 'signature_size_consistency', 'cohort_validation_interpretation']].to_string(index=False))
    print('\nOVERALL\n', overall_table.to_string(index=False))
if __name__ == '__main__':
    main()
