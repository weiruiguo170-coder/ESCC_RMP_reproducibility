from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp, wilcoxon
PROJECT_ROOT = INPUT_ROOT
PHASE_H_SCRIPT = REPO_ROOT / 'scripts/analysis/proteomics/global_z.py'
PROTEOMICS_DATA = PROJECT_ROOT / 'data' / 'interim' / 'processed_protein_matrix_author_supdata1.tsv'
PILOT_METRICS = PROJECT_ROOT / 'supplementary' / 'results' / 'cnmf_pilot_k_selection_metrics.tsv'
FORMAL_METRICS = PROJECT_ROOT / 'supplementary' / 'results' / 'cnmf_formal_metrics.tsv'
OUT = PROJECT_ROOT / 'results' / 'robustness_and_sensitivity'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260908
BOOTSTRAPS = 5000
RANDOM_SETS = 10000
RMPS = [f'RMP_C{i:02d}' for i in range(1, 8)]
SHORT = {r: r.replace('RMP_', '') for r in RMPS}
COLORS = {'RMP_C01': '#C33C54', 'RMP_C02': '#3F72AF', 'RMP_C03': '#8A6D3B', 'RMP_C04': '#4C956C', 'RMP_C05': '#765D69', 'RMP_C06': '#D17A22', 'RMP_C07': '#7B2CBF'}

def bh_fdr(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_phase_h_module():
    spec = importlib.util.spec_from_file_location('phase_h_analysis', PHASE_H_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

def replay_original(ph):
    (signatures, _) = ph.read_signatures()
    (raw, source, gene_matrix, z, sample_cols, patient_ids, *_) = ph.read_proteome()
    rng = np.random.default_rng(ph.SEED)
    (score50, _, map50) = ph.score_signatures(signatures[50], z, sample_cols)
    (stats50, _) = ph.paired_stats(score50, map50, patient_ids, rng)
    (score100, _, map100) = ph.score_signatures(signatures[100], z, sample_cols)
    (stats100, _) = ph.paired_stats(score100, map100, patient_ids, rng)
    (random50, _, _) = ph.random_set_calibration(stats50, map50, z, gene_matrix, sample_cols, patient_ids, rng)
    stored = pd.read_csv(resolve_data(PROJECT_ROOT / 'supplementary' / 'results' / 'proteomics_paired_statistics.csv'))
    stored_random = pd.read_csv(resolve_data(PROJECT_ROOT / 'supplementary' / 'results' / 'proteomics_random_set_calibration.csv'))
    checks = []
    for col in ['mean_paired_difference_T_minus_N', 'paired_Cohens_dz', 'raw_P', 'FDR']:
        a = stats50.set_index('RMP')[col].sort_index()
        b = stored.set_index('RMP')[col].sort_index()
        checks.append({'item': f'Top50 {col}', 'max_abs_difference': float(np.nanmax(np.abs(a - b))), 'pass': bool(np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True))})
    for col in ['observed_effect', 'empirical_P_two_sided_centered_on_null_median', 'empirical_FDR']:
        a = random50.set_index('RMP')[col].sort_index()
        b = stored_random.set_index('RMP')[col].sort_index()
        checks.append({'item': f'Top50 random calibration {col}', 'max_abs_difference': float(np.nanmax(np.abs(a - b))), 'pass': bool(np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True))})
    check_df = pd.DataFrame(checks)
    check_df.to_csv(OUT / 'original_proteomics_reproduction_check.tsv', sep='\t', index=False)
    if not check_df['pass'].all():
        raise RuntimeError('Original proteomics results did not reproduce exactly.')
    return (signatures, raw, source, gene_matrix, z, sample_cols, patient_ids, stats50, stats100, random50)

def rank_biserial(diff):
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return np.nan
    ranks = rankdata(np.abs(d), method='average')
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())

def bootstrap_patient_statistics(diff, rng):
    n = len(diff)
    idx = rng.integers(0, n, size=(BOOTSTRAPS, n))
    b = diff[idx]
    means = np.nanmean(b, axis=1)
    sds = np.nanstd(b, axis=1, ddof=1)
    dz = np.divide(means, sds, out=np.full_like(means, np.nan), where=sds > 0)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(np.nanquantile(dz, 0.025)), float(np.nanquantile(dz, 0.975)), float(np.mean(means > 0)) if np.mean(diff) > 0 else float(np.mean(means < 0)))

def gene_standardized_summary(block, rng):
    means = np.nanmean(block, axis=1)
    sds = np.nanstd(block, axis=1, ddof=1)
    valid = np.isfinite(means) & np.isfinite(sds) & (sds > 0)
    obs = float(np.mean(means[valid] / sds[valid]))
    draws = np.empty(BOOTSTRAPS)
    n = block.shape[1]
    for b in range(BOOTSTRAPS):
        x = block[:, rng.integers(0, n, n)]
        m = np.nanmean(x, axis=1)
        s = np.nanstd(x, axis=1, ddof=1)
        ok = np.isfinite(m) & np.isfinite(s) & (s > 0)
        draws[b] = np.mean(m[ok] / s[ok])
    return (obs, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), int(valid.sum()))

def make_strata(gene_matrix):
    props = pd.DataFrame(index=gene_matrix.index)
    props['missingness'] = gene_matrix.isna().mean(axis=1)
    props['mean_abundance'] = gene_matrix.mean(axis=1, skipna=True)
    props['missing_bin'] = pd.qcut(props['missingness'].rank(method='first'), 5, labels=False)
    props['abundance_bin'] = pd.qcut(props['mean_abundance'].rank(method='first'), 5, labels=False)
    props['stratum'] = props['missing_bin'].astype(int).astype(str) + '_' + props['abundance_bin'].astype(int).astype(str)
    return props

def matched_null_raw(target_genes, diff_matrix, props, rng):
    gene_to_idx = {g: i for (i, g) in enumerate(diff_matrix.index)}
    target_idx = {gene_to_idx[g] for g in target_genes}
    by_stratum = {s: [gene_to_idx[g] for g in grp.index] for (s, grp) in props.groupby('stratum')}
    counts = props.loc[target_genes, 'stratum'].value_counts().to_dict()
    arr = diff_matrix.to_numpy(float)
    null = np.empty(RANDOM_SETS)
    fallback = 0
    for b in range(RANDOM_SETS):
        (chosen, used) = ([], set())
        for (stratum, count) in counts.items():
            candidates = [i for i in by_stratum[stratum] if i not in target_idx and i not in used]
            if len(candidates) < count:
                fallback += 1
                candidates = [i for i in range(len(arr)) if i not in target_idx and i not in used]
            pick = rng.choice(candidates, count, replace=False).tolist()
            chosen.extend(pick)
            used.update(pick)
        patient_scores = np.nanmean(arr[np.asarray(chosen)], axis=0)
        null[b] = float(np.nanmean(patient_scores))
    return (null, fallback)

def run_paired_difference(ph, signatures, gene_matrix, sample_cols, patient_ids, original50, original100):
    normal_cols = [f'{p}N' for p in patient_ids]
    tumor_cols = [f'{p}T' for p in patient_ids]
    diff = gene_matrix[tumor_cols].to_numpy(float) - gene_matrix[normal_cols].to_numpy(float)
    diff_matrix = pd.DataFrame(diff, index=gene_matrix.index, columns=patient_ids)
    props = make_strata(gene_matrix)
    (detail_rows, null_frames, patient_frames) = ([], [], [])
    for size in (50, 100):
        for (rmp_i, rmp) in enumerate(RMPS):
            mapped = [g for g in signatures[size][rmp] if g in diff_matrix.index]
            block = diff_matrix.loc[mapped].to_numpy(float)
            patient_score = np.nanmean(block, axis=0)
            keep = np.isfinite(patient_score)
            d = patient_score[keep]
            (stat, wp) = wilcoxon(d, alternative='two-sided', zero_method='wilcox')
            tp = ttest_1samp(d, 0, nan_policy='omit').pvalue
            rng = np.random.default_rng(SEED + size * 100 + rmp_i)
            (lo, hi, dzlo, dzhi, sign_stability) = bootstrap_patient_statistics(d, rng)
            sd = np.std(d, ddof=1)
            dz = float(np.mean(d) / sd) if sd > 0 else np.nan
            (standardized, slo, shi, std_genes) = gene_standardized_summary(block[:, keep], rng)
            (null, fallback) = matched_null_raw(mapped, diff_matrix, props, rng)
            center = float(np.median(null))
            obs = float(np.mean(d))
            ep = float((1 + np.sum(np.abs(null - center) >= abs(obs - center))) / (RANDOM_SETS + 1))
            detail_rows.append({'RMP': rmp, 'signature': f'Top{size}', 'N_pairs': len(d), 'mapped_proteins': len(mapped), 'raw_mean_delta': obs, 'raw_median_delta': float(np.median(d)), 'raw_bootstrap_CI_low': lo, 'raw_bootstrap_CI_high': hi, 'bootstrap_sign_stability': sign_stability, 'wilcoxon_statistic': float(stat), 'wilcoxon_P': float(wp), 'paired_t_test_P': float(tp), 'Cohens_dz': dz, 'dz_CI_low': dzlo, 'dz_CI_high': dzhi, 'rank_biserial_correlation': rank_biserial(d), 'gene_standardized_mean_effect': standardized, 'gene_standardized_CI_low': slo, 'gene_standardized_CI_high': shi, 'genes_with_valid_standardized_effect': std_genes, 'matched_null_mean': float(np.mean(null)), 'matched_null_median': center, 'matched_null_SD': float(np.std(null, ddof=1)), 'matched_null_empirical_P': ep, 'matched_null_fallback_draw_events': fallback})
            null_frames.append(pd.DataFrame({'RMP': rmp, 'signature': f'Top{size}', 'replicate': np.arange(1, RANDOM_SETS + 1), 'raw_mean_delta': null}))
            patient_frames.append(pd.DataFrame({'RMP': rmp, 'signature': f'Top{size}', 'patient_id': np.asarray(patient_ids)[keep], 'raw_paired_delta': d}))
    detail = pd.DataFrame(detail_rows)
    detail['wilcoxon_FDR'] = np.nan
    detail['matched_null_FDR'] = np.nan
    for (signature, idx) in detail.groupby('signature').groups.items():
        detail.loc[idx, 'wilcoxon_FDR'] = bh_fdr(detail.loc[idx, 'wilcoxon_P'])
        detail.loc[idx, 'matched_null_FDR'] = bh_fdr(detail.loc[idx, 'matched_null_empirical_P'])
    detail.to_csv(OUT / 'proteomics_paired_difference_detailed.tsv', sep='\t', index=False)
    null_all = pd.concat(null_frames, ignore_index=True)
    patient_all = pd.concat(patient_frames, ignore_index=True)
    null_all.to_csv(OUT / 'proteomics_paired_difference_matched_null_draws.tsv', sep='\t', index=False)
    patient_all.to_csv(OUT / 'proteomics_paired_difference_patient_scores.tsv', sep='\t', index=False)
    original = pd.concat([original50.assign(signature='Top50'), original100.assign(signature='Top100')], ignore_index=True)
    original = original[['RMP', 'signature', 'mean_paired_difference_T_minus_N', 'paired_Cohens_dz', 'FDR', 'bootstrap_CI_low', 'bootstrap_CI_high']]
    original = original.rename(columns={'mean_paired_difference_T_minus_N': 'original_global_z_mean_delta', 'paired_Cohens_dz': 'original_global_z_Cohens_dz', 'FDR': 'original_global_z_FDR', 'bootstrap_CI_low': 'original_global_z_CI_low', 'bootstrap_CI_high': 'original_global_z_CI_high'})
    detail = detail.merge(original, on=['RMP', 'signature'], validate='one_to_one')
    detail['direction_vs_original'] = np.where(np.sign(detail['raw_mean_delta']) == np.sign(detail['original_global_z_mean_delta']), 'CONCORDANT', 'DISCORDANT')
    detail['raw_CI_excludes_zero'] = (detail['raw_bootstrap_CI_low'] > 0) | (detail['raw_bootstrap_CI_high'] < 0)
    detail['matched_null_supported'] = detail['matched_null_FDR'] < 0.05
    detail['paired_raw_support'] = np.where(detail['raw_CI_excludes_zero'] & (detail['wilcoxon_FDR'] < 0.05) & detail['matched_null_supported'], 'SUPPORTED', 'LIMITED')
    stored_support = pd.read_csv(resolve_data(PROJECT_ROOT / 'supplementary' / 'results' / 'proteomics_paired_statistics.csv')).set_index('RMP')['orthogonal_protein_support'].to_dict()
    comparison_rows = []
    for rmp in RMPS:
        r50 = detail[(detail.RMP == rmp) & (detail.signature == 'Top50')].iloc[0]
        r100 = detail[(detail.RMP == rmp) & (detail.signature == 'Top100')].iloc[0]
        comparison_rows.append({'RMP': rmp, 'mapped_Top50': int(r50.mapped_proteins), 'mapped_Top100': int(r100.mapped_proteins), 'original_direction': 'UP' if r50.original_global_z_mean_delta > 0 else 'DOWN', 'original_support': stored_support[rmp], 'paired_raw_mean': r50.raw_mean_delta, 'paired_raw_median': r50.raw_median_delta, 'paired_bootstrap_CI_low': r50.raw_bootstrap_CI_low, 'paired_bootstrap_CI_high': r50.raw_bootstrap_CI_high, 'paired_Wilcoxon_P': r50.wilcoxon_P, 'paired_FDR': r50.wilcoxon_FDR, 'paired_t_test_P': r50.paired_t_test_P, 'Cohens_dz': r50.Cohens_dz, 'dz_CI_low': r50.dz_CI_low, 'dz_CI_high': r50.dz_CI_high, 'rank_biserial_correlation': r50.rank_biserial_correlation, 'gene_standardized_mean_effect': r50.gene_standardized_mean_effect, 'gene_standardized_CI_low': r50.gene_standardized_CI_low, 'gene_standardized_CI_high': r50.gene_standardized_CI_high, 'matched_null_empirical_P': r50.matched_null_empirical_P, 'matched_null_FDR': r50.matched_null_FDR, 'Top100_direction': 'UP' if r100.raw_mean_delta > 0 else 'DOWN', 'Top100_raw_mean': r100.raw_mean_delta, 'Top100_raw_median': r100.raw_median_delta, 'Top100_bootstrap_CI_low': r100.raw_bootstrap_CI_low, 'Top100_bootstrap_CI_high': r100.raw_bootstrap_CI_high, 'Top100_Wilcoxon_P': r100.wilcoxon_P, 'Top100_FDR': r100.wilcoxon_FDR, 'Top100_Cohens_dz': r100.Cohens_dz, 'Top100_dz_CI_low': r100.dz_CI_low, 'Top100_dz_CI_high': r100.dz_CI_high, 'Top100_matched_null_empirical_P': r100.matched_null_empirical_P, 'Top100_matched_null_FDR': r100.matched_null_FDR, 'Top50_Top100_concordance': 'CONCORDANT' if np.sign(r50.raw_mean_delta) == np.sign(r100.raw_mean_delta) else 'DISCORDANT', 'conclusion': f'Top50 {r50.paired_raw_support.lower()}; Top100 {r100.paired_raw_support.lower()}; original direction {r50.direction_vs_original.lower()}'})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / 'proteomics_paired_difference_sensitivity.tsv', sep='\t', index=False)
    return (detail, comparison, null_all, patient_all)

def build_candidate_k_table():
    pilot = pd.read_csv(resolve_data(PILOT_METRICS), sep='\t')
    pilot['k'] = pilot['k'].astype(int)
    selected = {'GSE160269': 10, 'GSE188900': 8, 'GSE196756': 5}
    reasons = {'GSE160269': 'K=10 was the first tested rank after which reconstruction improvement fell below about 1% per added component, while silhouette returned to ~0.84 and remained similar through K=15.', 'GSE188900': 'K=8 retained silhouette 0.871 while per-component reconstruction gains had declined; K=15 was disfavored by lower stability. P6 imbalance was retained as a limitation.', 'GSE196756': 'K=5 was the conservative stable rank for 489 cells; the unresolved reconstruction plateau and 84.87% P4 dominance made confidence low. No biological interpretability criterion was used.'}
    formal = pd.read_csv(resolve_data(FORMAL_METRICS), sep='\t').set_index('cohort')
    out = pd.DataFrame({'Cohort': pilot.cohort, 'K': pilot.k, 'pilot_replicates': pilot.replicates_requested.astype(int), 'completed_replicates': pilot.replicates_completed.astype(int), 'consensus_silhouette': pilot.silhouette, 'reconstruction_metric': pilot.prediction_error, 'reconstruction_metric_name': 'Frobenius SSE', 'other_stability_metric': 'NA', 'density_threshold_outcome': 'not applied for pilot rank statistics', 'retained_runs_or_solutions': '20/20 factorization spectra available; density retention not applied', 'selected': ['YES' if k == selected[c] else 'NO' for (c, k) in zip(pilot.cohort, pilot.k)], 'selection_reason': [reasons[c] if k == selected[c] else 'not selected' for (c, k) in zip(pilot.cohort, pilot.k)]})
    out['formal_replicates_requested'] = [int(formal.loc[c, 'replicates_requested']) if k == selected[c] else pd.NA for (c, k) in zip(pilot.cohort, pilot.k)]
    out['formal_replicates_completed'] = [int(formal.loc[c, 'replicates_completed']) if k == selected[c] else pd.NA for (c, k) in zip(pilot.cohort, pilot.k)]
    out['formal_consensus_silhouette'] = [float(formal.loc[c, 'silhouette']) if k == selected[c] else np.nan for (c, k) in zip(pilot.cohort, pilot.k)]
    out['formal_reconstruction_SSE'] = [float(formal.loc[c, 'prediction_error']) if k == selected[c] else np.nan for (c, k) in zip(pilot.cohort, pilot.k)]
    out['formal_density_threshold'] = [float(formal.loc[c, 'density_threshold']) if k == selected[c] else np.nan for (c, k) in zip(pilot.cohort, pilot.k)]
    out['formal_components_retained'] = [f"{int(formal.loc[c, 'components_retained'])}/{int(formal.loc[c, 'components_total'])}" if k == selected[c] else 'NA' for (c, k) in zip(pilot.cohort, pilot.k)]
    out.to_csv(OUT / 'candidate_K_diagnostics.tsv', sep='\t', index=False, na_rep='NA')
    return out

def make_s1(candidate):
    cohorts = ['GSE160269', 'GSE188900', 'GSE196756']
    selected = {'GSE160269': 10, 'GSE188900': 8, 'GSE196756': 5}
    (fig, axes) = plt.subplots(2, 3, figsize=(11.5, 6.8), constrained_layout=True)
    for (j, cohort) in enumerate(cohorts):
        d = candidate[candidate.Cohort == cohort].sort_values('K')
        axes[0, j].plot(d.K, d.consensus_silhouette, 'o-', color='#2B6F9C', lw=1.8, ms=5)
        axes[0, j].axvline(selected[cohort], color='#C43C39', ls='--', lw=1.2)
        axes[0, j].scatter([selected[cohort]], d.loc[d.K == selected[cohort], 'consensus_silhouette'], s=70, facecolor='#C43C39', edgecolor='white', zorder=5)
        axes[0, j].set_title(cohort, fontsize=11, fontweight='bold')
        axes[0, j].set_ylabel('Consensus silhouette')
        axes[0, j].set_xticks(d.K)
        axes[0, j].set_ylim(max(0.6, d.consensus_silhouette.min() - 0.06), 1.02)
        axes[1, j].plot(d.K, d.reconstruction_metric, 'o-', color='#6A8E3A', lw=1.8, ms=5)
        axes[1, j].axvline(selected[cohort], color='#C43C39', ls='--', lw=1.2)
        axes[1, j].scatter([selected[cohort]], d.loc[d.K == selected[cohort], 'reconstruction_metric'], s=70, facecolor='#C43C39', edgecolor='white', zorder=5)
        axes[1, j].set_xlabel('Candidate K')
        axes[1, j].set_ylabel('Frobenius reconstruction SSE')
        axes[1, j].set_xticks(d.K)
        axes[1, j].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        for i in range(2):
            axes[i, j].spines[['top', 'right']].set_visible(False)
            axes[i, j].grid(axis='y', color='#D9D9D9', lw=0.6, alpha=0.7)
    for (label, ax) in zip('ABCDEF', axes.flat):
        ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
    fig.suptitle('cNMF candidate-rank diagnostics across discovery cohorts', fontsize=13, fontweight='bold')
    fig.savefig(OUT / 'Supplementary_Figure_S1.png', dpi=450, bbox_inches='tight')
    fig.savefig(OUT / 'Supplementary_Figure_S1.pdf', bbox_inches='tight')
    plt.close(fig)

def make_s2(detail, null_all, patient_all):
    fig = plt.figure(figsize=(11.5, 8.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    prim = ['RMP_C01', 'RMP_C02', 'RMP_C07']
    pdata = patient_all[(patient_all.signature == 'Top50') & patient_all.RMP.isin(prim)]
    vals = [pdata.loc[pdata.RMP == r, 'raw_paired_delta'].to_numpy() for r in prim]
    parts = axA.violinplot(vals, showextrema=False, widths=0.78)
    for (body, r) in zip(parts['bodies'], prim):
        body.set_facecolor(COLORS[r])
        body.set_alpha(0.28)
        body.set_edgecolor(COLORS[r])
    axA.boxplot(vals, widths=0.28, showfliers=False, medianprops={'color': 'black'}, boxprops={'color': '#555555'}, whiskerprops={'color': '#777777'}, capprops={'color': '#777777'})
    jitter = np.random.default_rng(SEED).normal(0, 0.055, (len(vals), max(map(len, vals))))
    for (i, (v, r)) in enumerate(zip(vals, prim), 1):
        axA.scatter(i + jitter[i - 1, :len(v)], v, s=8, color=COLORS[r], alpha=0.35, edgecolors='none')
    axA.axhline(0, color='#777777', lw=0.8)
    axA.set_xticks(range(1, 4), [SHORT[r] for r in prim])
    axA.set_ylabel('Patient-level raw tumor-normal delta')
    d50 = detail[detail.signature == 'Top50'].set_index('RMP').loc[RMPS]
    y = np.arange(7)
    axB.errorbar(d50.Cohens_dz, y, xerr=[d50.Cohens_dz - d50.dz_CI_low, d50.dz_CI_high - d50.Cohens_dz], fmt='none', ecolor='#555555', capsize=3, lw=1.2)
    axB.scatter(d50.Cohens_dz, y, c=[COLORS[r] for r in RMPS], s=42, zorder=3)
    axB.axvline(0, color='#777777', lw=0.8)
    axB.set_yticks(y, [SHORT[r] for r in RMPS])
    axB.invert_yaxis()
    axB.set_xlabel("Paired Cohen's dz (95% bootstrap CI)")
    axC.scatter(d50.original_global_z_Cohens_dz, d50.Cohens_dz, c=[COLORS[r] for r in RMPS], s=48)
    for (r, row) in d50.iterrows():
        axC.annotate(SHORT[r], (row.original_global_z_Cohens_dz, row.Cohens_dz), xytext=(4, 3), textcoords='offset points', fontsize=8)
    lim = [min(axC.get_xlim()[0], axC.get_ylim()[0]), max(axC.get_xlim()[1], axC.get_ylim()[1])]
    axC.plot(lim, lim, ls='--', color='#999999', lw=0.8)
    axC.set_xlim(lim)
    axC.set_ylim(lim)
    axC.set_xlabel('Original global-z paired dz')
    axC.set_ylabel('Raw paired-difference dz')
    n50 = null_all[(null_all.signature == 'Top50') & null_all.RMP.isin(prim)]
    for r in prim:
        x = n50.loc[n50.RMP == r, 'raw_mean_delta'].to_numpy()
        obs = d50.loc[r, 'raw_mean_delta']
        axD.hist(x, bins=55, density=True, histtype='step', lw=1.2, color=COLORS[r], label=SHORT[r], alpha=0.9)
        axD.axvline(obs, color=COLORS[r], lw=1.5)
    axD.set_xlabel('Matched-set mean raw paired delta')
    axD.set_ylabel('Density')
    axD.legend(frameon=False, title='Observed lines')
    for (label, ax) in zip('ABCD', [axA, axB, axC, axD]):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E1E1E1', lw=0.5, alpha=0.6)
    fig.suptitle('Paired-difference sensitivity analysis of Protein-RMPs', fontsize=13, fontweight='bold')
    fig.savefig(OUT / 'Supplementary_Figure_S2.png', dpi=450, bbox_inches='tight')
    fig.savefig(OUT / 'Supplementary_Figure_S2.pdf', bbox_inches='tight')
    plt.close(fig)

def main():
    ph = load_phase_h_module()
    (signatures, raw, source, gene_matrix, z, sample_cols, patient_ids, original50, original100, random50) = replay_original(ph)
    (detail, comparison, null_all, patient_all) = run_paired_difference(ph, signatures, gene_matrix, sample_cols, patient_ids, original50, original100)
    candidate = build_candidate_k_table()
    make_s1(candidate)
    make_s2(detail, null_all, patient_all)
    manifest = {'status': 'PASS', 'seed': SEED, 'bootstrap_replicates': BOOTSTRAPS, 'matched_random_sets_per_RMP_per_signature': RANDOM_SETS, 'input_matrix_sha256': sha256(PROTEOMICS_DATA), 'original_results_reproduced': True}
    (OUT / 'supplementary_analysis_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(comparison.to_string(index=False))
if __name__ == '__main__':
    main()
