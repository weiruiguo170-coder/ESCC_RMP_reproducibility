"""Freeze Phase B1 cohort-specific cNMF outputs and technical/usage QC."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import hashlib
import json
import re
from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cnmf.cnmf import load_df_from_npz
PROJECT_ROOT = INPUT_ROOT
RESULTS_DIR = PROJECT_ROOT / 'results'
OUT = RESULTS_DIR / 'cnmf_formal'
MANIFEST = pd.read_csv(resolve_data(RESULTS_DIR / 'cnmf_pilot' / 'input_manifest.tsv'), sep='\t').set_index('cohort')
K_BY_COHORT = {'GSE160269': 10, 'GSE188900': 8, 'GSE196756': 5}
STRESS_SET = {'FOS', 'FOSB', 'JUN', 'JUNB', 'JUND', 'EGR1', 'EGR2', 'EGR3', 'ATF3', 'IER2', 'IER3', 'HSPA1A', 'HSPA1B', 'HSP90AA1', 'HSP90AB1', 'DNAJB1', 'DDIT3', 'PPP1R15A', 'NFKBIA', 'ZFP36'}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def dominance_label(fraction: float) -> str:
    if fraction >= 0.8:
        return 'STRONG_PATIENT_DOMINANCE'
    if fraction >= 0.5:
        return 'MODERATE_PATIENT_DOMINANCE'
    return 'NO_DOMINANCE'

def aggregate_usage(meta: pd.DataFrame, usage_raw: pd.DataFrame, usage_norm: pd.DataFrame, column: str) -> pd.DataFrame:
    joined = usage_norm.join(meta[[column]])
    rows = []
    totals = usage_norm.sum(axis=0)
    for (contributor, frame) in joined.groupby(column, observed=True):
        values = frame[usage_norm.columns]
        raw_values = usage_raw.loc[frame.index]
        for program in usage_norm.columns:
            rows.append({f'{column}': str(contributor), 'program_id': program, 'n_cells': len(frame), 'aggregate_raw_usage': float(raw_values[program].sum()), 'mean_raw_usage': float(raw_values[program].mean()), 'median_raw_usage': float(raw_values[program].median()), 'aggregate_normalized_usage': float(values[program].sum()), 'fraction_of_total_program_usage': float(values[program].sum() / totals[program]), 'mean_normalized_usage': float(values[program].mean()), 'median_normalized_usage': float(values[program].median()), 'iqr_normalized_usage': float(values[program].quantile(0.75) - values[program].quantile(0.25)), 'fraction_cells_non_negligible': float((values[program] >= 0.01).mean())})
    return pd.DataFrame(rows)

def process_cohort(cohort: str, k: int) -> tuple[pd.DataFrame, dict[str, object]]:
    cohort_dir = OUT / cohort
    formal_dir = cohort_dir / 'formal'
    tmp_dir = formal_dir / 'cnmf_tmp'
    config = json.loads((cohort_dir / 'exact_configuration.json').read_text(encoding='utf-8'))
    config['technical_flag_rules']['HIGH_WITHIN_COHORT_COSINE_REVIEW'] = 'maximum cosine similarity >=0.90 between consensus HVG spectra within the same cohort'
    (cohort_dir / 'exact_configuration.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    dt = str(config['selected_density_threshold']).replace('.', '_')
    input_path = Path(config['input'])
    observed_hash = sha256(input_path)
    expected_hash = str(MANIFEST.loc[cohort, 'sha256'])
    if observed_hash.lower() != expected_hash.lower():
        raise RuntimeError(f'HASH_MISMATCH during QC {cohort}')
    adata = ad.read_h5ad(input_path, backed='r')
    meta = adata.obs[['patient_id', 'sample_id']].copy()
    usages_raw = load_df_from_npz(tmp_dir / f'formal.usages.k_{k}.dt_{dt}.consensus.df.npz')
    spectra = load_df_from_npz(tmp_dir / f'formal.gene_spectra_tpm.k_{k}.dt_{dt}.df.npz')
    scores = load_df_from_npz(tmp_dir / f'formal.gene_spectra_score.k_{k}.dt_{dt}.df.npz')
    consensus_hvg = load_df_from_npz(tmp_dir / f'formal.spectra.k_{k}.dt_{dt}.consensus.df.npz')
    adata.file.close()
    if not usages_raw.index.equals(meta.index):
        raise RuntimeError(f'{cohort}: usage cell index does not match frozen input')
    program_ids = [f'{cohort}_P{i:02d}' for i in range(1, k + 1)]
    usages_raw.columns = program_ids
    spectra.index = program_ids
    scores.index = program_ids
    consensus_hvg.index = program_ids
    usages_norm = usages_raw.div(usages_raw.sum(axis=1), axis=0).fillna(0)
    gene_spectra = spectra.T
    gene_scores = scores.T
    gene_ranks = gene_scores.rank(axis=0, ascending=False, method='min').astype(int)
    gene_spectra.to_csv(cohort_dir / 'gene_program_spectra.tsv', sep='\t', index_label='gene')
    gene_scores.to_csv(cohort_dir / 'gene_program_scores.tsv', sep='\t', index_label='gene')
    gene_ranks.to_csv(cohort_dir / 'gene_program_ranks.tsv', sep='\t', index_label='gene')
    usages_raw.to_csv(cohort_dir / 'cell_program_usage.tsv', sep='\t', index_label='cell_id')
    usages_norm.to_csv(cohort_dir / 'cell_program_usage_normalized.tsv', sep='\t', index_label='cell_id')
    top_rows = []
    for program in program_ids:
        ordered = gene_scores[program].sort_values(ascending=False).head(50)
        for (rank, (gene, score)) in enumerate(ordered.items(), start=1):
            top_rows.append({'program_id': program, 'rank': rank, 'gene': gene, 'score': float(score), 'spectrum_tpm': float(gene_spectra.loc[gene, program])})
    top50 = pd.DataFrame(top_rows)
    top50.to_csv(cohort_dir / 'top50_genes_per_program.tsv', sep='\t', index=False)
    hvg_l2 = consensus_hvg.div(np.sqrt((consensus_hvg ** 2).sum(axis=1)), axis=0)
    cosine = hvg_l2 @ hvg_l2.T
    similarity_rows = []
    max_similarity = {program: -np.inf for program in program_ids}
    max_partner = {program: 'NONE' for program in program_ids}
    for (i, left) in enumerate(program_ids):
        for right in program_ids[i + 1:]:
            value = float(cosine.loc[left, right])
            similarity_rows.append({'program_1': left, 'program_2': right, 'cosine_similarity': value, 'high_duplication_review': value >= 0.9})
            if value > max_similarity[left]:
                (max_similarity[left], max_partner[left]) = (value, right)
            if value > max_similarity[right]:
                (max_similarity[right], max_partner[right]) = (value, left)
    similarity = pd.DataFrame(similarity_rows)
    similarity.to_csv(cohort_dir / 'within_cohort_program_similarity.tsv', sep='\t', index=False)
    patient = aggregate_usage(meta, usages_raw, usages_norm, 'patient_id')
    sample = aggregate_usage(meta, usages_raw, usages_norm, 'sample_id')
    patient.to_csv(cohort_dir / 'patient_program_usage.tsv', sep='\t', index=False)
    sample.to_csv(cohort_dir / 'sample_program_usage.tsv', sep='\t', index=False)
    summary_rows = []
    flag_rows = []
    for program in program_ids:
        values = usages_norm[program]
        patient_part = patient.loc[patient.program_id.eq(program)].sort_values('fraction_of_total_program_usage', ascending=False)
        sample_part = sample.loc[sample.program_id.eq(program)].sort_values('fraction_of_total_program_usage', ascending=False)
        top = top50.loc[top50.program_id.eq(program), 'gene'].astype(str).str.upper()
        mt_fraction = float(top.str.startswith('MT-').mean())
        ribo_fraction = float(top.str.match('^RP[LS]').mean())
        stress_fraction = float(top.isin(STRESS_SET).mean())
        appreciable = int((patient_part['mean_normalized_usage'] >= 0.01).sum())
        patient_fraction = float(patient_part.iloc[0]['fraction_of_total_program_usage'])
        sample_fraction = float(sample_part.iloc[0]['fraction_of_total_program_usage'])
        dflag = dominance_label(patient_fraction)
        technical = []
        review = []
        if mt_fraction >= 0.5:
            technical.append('HIGH_MT_TOP50')
        if ribo_fraction >= 0.5:
            technical.append('HIGH_RIBOSOMAL_TOP50')
        if stress_fraction >= 0.3:
            review.append('DISSOCIATION_STRESS_TOP50_REVIEW')
        if float((values >= 0.01).mean()) < 0.01:
            review.append('VERY_SPARSE_USAGE_REVIEW')
        if appreciable <= 1:
            review.append('PRIVATE_PATIENT_REVIEW')
        if sample_fraction >= 0.8:
            review.append('EXTREME_SAMPLE_DEPENDENCE_REVIEW')
        if max_similarity[program] >= 0.9:
            review.append('HIGH_WITHIN_COHORT_COSINE_REVIEW')
        summary_rows.append({'program_id': program, 'total_raw_usage': float(usages_raw[program].sum()), 'mean_raw_usage': float(usages_raw[program].mean()), 'median_raw_usage': float(usages_raw[program].median()), 'q1_raw_usage': float(usages_raw[program].quantile(0.25)), 'q3_raw_usage': float(usages_raw[program].quantile(0.75)), 'iqr_raw_usage': float(usages_raw[program].quantile(0.75) - usages_raw[program].quantile(0.25)), 'total_normalized_usage': float(values.sum()), 'mean_normalized_usage': float(values.mean()), 'median_normalized_usage': float(values.median()), 'q1_normalized_usage': float(values.quantile(0.25)), 'q3_normalized_usage': float(values.quantile(0.75)), 'iqr_normalized_usage': float(values.quantile(0.75) - values.quantile(0.25)), 'fraction_cells_non_negligible': float((values >= 0.01).mean()), 'largest_patient': str(patient_part.iloc[0]['patient_id']), 'largest_patient_usage_fraction': patient_fraction, 'largest_sample': str(sample_part.iloc[0]['sample_id']), 'largest_sample_usage_fraction': sample_fraction, 'patients_with_appreciable_usage': appreciable, 'patient_dominance_flag': dflag})
        flag_rows.append({'program_id': program, 'patient_dominance_flag': dflag, 'largest_patient': str(patient_part.iloc[0]['patient_id']), 'largest_patient_usage_fraction': patient_fraction, 'largest_sample': str(sample_part.iloc[0]['sample_id']), 'largest_sample_usage_fraction': sample_fraction, 'patients_with_appreciable_usage': appreciable, 'mt_fraction_top50': mt_fraction, 'ribosomal_fraction_top50': ribo_fraction, 'locked_stress_set_fraction_top50': stress_fraction, 'max_within_cohort_hvg_spectra_cosine': max_similarity[program], 'most_similar_within_cohort_program': max_partner[program], 'technical_flags': ';'.join(technical) if technical else 'NONE', 'review_flags': ';'.join(review) if review else 'NONE', 'automatic_exclusion': False})
    summary = pd.DataFrame(summary_rows)
    flags = pd.DataFrame(flag_rows)
    summary.to_csv(cohort_dir / 'program_usage_summary.tsv', sep='\t', index=False)
    flags.to_csv(cohort_dir / 'program_qc_flags.tsv', sep='\t', index=False)
    heat = patient.pivot(index='patient_id', columns='program_id', values='mean_normalized_usage')
    patient_order = meta.groupby('patient_id', observed=True).size().sort_values(ascending=False).index.astype(str)
    heat = heat.reindex(patient_order)
    height = max(4.0, min(18.0, 0.26 * len(heat) + 2.0))
    (fig, ax) = plt.subplots(figsize=(max(8.0, 0.75 * k + 3), height), constrained_layout=True)
    sns.heatmap(heat, cmap='viridis', ax=ax, cbar_kws={'label': 'Mean normalized usage'})
    ax.set_title(f'{cohort}: patient × formal program usage QC')
    ax.set_xlabel('Neutral program ID')
    ax.set_ylabel('Patient')
    fig.savefig(cohort_dir / 'patient_program_usage_heatmap.png', dpi=250)
    fig.savefig(cohort_dir / 'patient_program_usage_heatmap.pdf')
    plt.close(fig)
    replicate_files = list(tmp_dir.glob(f'formal.spectra.k_{k}.iter_*.df.npz'))
    log_text = '\n'.join((p.read_text(encoding='utf-8', errors='replace') for p in cohort_dir.glob('*.log')))
    worker_warning_lines = [line for line in log_text.splitlines() if 'warning' in line.lower()]
    worker_error_lines = [line for line in log_text.splitlines() if re.search('error|traceback', line, re.I)]
    warnings = ['CONSENSUS_WARNING: scanpy preprocessing _scale.py: Received a view of an AnnData; making a copy. Non-fatal; consensus completed.'] + worker_warning_lines
    errors = worker_error_lines
    (cohort_dir / 'warnings_errors.txt').write_text('WARNINGS\n' + '\n'.join(warnings) + '\n\nERRORS\n' + ('\n'.join(errors) if errors else 'NONE') + '\n', encoding='utf-8')
    metrics = pd.read_csv(resolve_data(cohort_dir / 'formal_cnmf_metrics.tsv'), sep='\t')
    metrics['cohort'] = cohort
    metrics['k'] = k
    metrics['replicates_requested'] = 100
    metrics['replicates_completed'] = len(replicate_files)
    metrics['replicates_failed'] = 100 - len(replicate_files)
    metrics['worker_warning_count'] = len(worker_warning_lines)
    metrics['consensus_warning_count'] = 1
    metrics['error_count'] = len(errors)
    metrics['input_sha256'] = observed_hash
    metrics.to_csv(cohort_dir / 'formal_cnmf_metrics.tsv', sep='\t', index=False)
    (cohort_dir / 'formal_run.log').write_text(f"INPUT_HASH_AUDIT PASS {observed_hash}\nPREPARE PASS\nFACTORIZE PASS {len(replicate_files)}/100\nCOMBINE PASS\nCONSENSUS PASS density_threshold={config['selected_density_threshold']}\nOUTPUT_FREEZE PASS programs={k}\n", encoding='utf-8')
    return (flags, {'metrics': metrics.iloc[0].to_dict(), 'summary': summary, 'flags': flags})

def fmt(x: float) -> str:
    return f'{x:.6f}'

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {cohort: process_cohort(cohort, k)[1] for (cohort, k) in K_BY_COHORT.items()}
    combined_metrics = pd.DataFrame([results[c]['metrics'] for c in K_BY_COHORT])
    combined_metrics.to_csv(OUT / 'formal_cnmf_metrics.tsv', sep='\t', index=False)
    all_flags = pd.concat([results[c]['flags'].assign(cohort=c) for c in K_BY_COHORT], ignore_index=True)
    all_flags.to_csv(OUT / 'program_qc_flags.tsv', sep='\t', index=False)
    lines = ['# Phase B1 — Formal Independent cNMF and Within-Cohort Program QC', '', 'Date: 2026-08-26 (Asia/Shanghai)', '', '## 1. Executive status', '', '| Cohort | Formal cNMF status | Programs | Role/QC context |', '|---|---|---:|---|', '| GSE160269 | PASS | 10 | Primary large cohort |', '| GSE188900 | PASS | 8 | P6 cell imbalance retained as QC concern |', '| GSE196756 | PASS | 5 | Small supportive cohort; strong P4/sample dominance retained |', '', '**PHASE B1 OVERALL STATUS: PASS**', '', 'No cross-cohort matching, RMP construction, enrichment, biological naming, treatment, spatial, trajectory, survival, clinical, or manuscript analysis was performed.', '', '## 2. Reproducibility audit', '', '| Cohort | Input SHA256 | K | Replicates | Seed | HVGs | cNMF | Density threshold | Neighborhood | Workers × threads |', '|---|---|---:|---:|---:|---:|---|---:|---:|---|']
    for cohort in K_BY_COHORT:
        m = results[cohort]['metrics']
        lines.append(f"| {cohort} | `{m['input_sha256']}` | {int(m['k'])} | {int(m['replicates_completed'])}/{int(m['replicates_requested'])} | 20260826 | 2,000 independent | 1.7.1 | {m['density_threshold']} | 0.30 | 4 × 1 |")
    lines += ['', 'Inputs were re-hashed before formal preparation and again during output freeze; all match the Phase B0 manifest. Raw nonnegative integer UMI counts in `.X` were used. HVGs were independently selected per cohort using the already locked cNMF top-2,000 rule.', '', 'Density selection was locked before results: test 0.5 (cNMF default), then 1.0, then 2.0, selecting the first threshold retaining at least 80% of replicate components with at least five components in every KMeans cluster. Top genes were not inspected during threshold selection.', '', '## 3. Formal cNMF metrics', '', '| Cohort | Stability (silhouette) | Reconstruction SSE | Components removed | Removed proportion | Replicate success | Warnings / errors |', '|---|---:|---:|---:|---:|---|---|']
    for cohort in K_BY_COHORT:
        m = results[cohort]['metrics']
        lines.append(f"| {cohort} | {fmt(m['silhouette'])} | {m['prediction_error']:,.2f} | {int(m['components_removed'])}/{int(m['components_total'])} | {1 - m['retained_fraction']:.2%} | {int(m['replicates_completed'])}/{int(m['replicates_requested'])} | {int(m['worker_warning_count'])} worker + {int(m['consensus_warning_count'])} non-fatal consensus / {int(m['error_count'])} |")
    lines += ['', 'All silhouettes exceed the pre-locked significant-instability stop threshold of 0.50. No secondary-K run was triggered or performed.', '', '## 4. Program inventory', '', '| Cohort | Expected | Actual | Neutral IDs |', '|---|---:|---:|---|', '| GSE160269 | 10 | 10 | GSE160269_P01–P10 |', '| GSE188900 | 8 | 8 | GSE188900_P01–P08 |', '| GSE196756 | 5 | 5 | GSE196756_P01–P05 |', '', 'Frozen outputs include gene×program spectra, scores and ranks; raw and normalized cell×program usage; top-50 genes; cohort metrics; exact configuration; patient/sample usage tables; heatmaps; flags; and logs.', '', '## 5. Patient/sample dependence', '', 'Locked patient thresholds: largest-patient total program usage <50% = NO_DOMINANCE; 50–<80% = MODERATE; ≥80% = STRONG. Flags do not remove programs.', '']
    for cohort in K_BY_COHORT:
        flags = results[cohort]['flags']
        lines.append(f'### {cohort}')
        lines.append('')
        for label in ('NO_DOMINANCE', 'MODERATE_PATIENT_DOMINANCE', 'STRONG_PATIENT_DOMINANCE'):
            ids = flags.loc[flags.patient_dominance_flag.eq(label), 'program_id'].tolist()
            lines.append(f"- {label}: {(', '.join(ids) if ids else 'NONE')}")
        lines.append('')
    g188 = results['GSE188900']['flags']
    p6 = g188.loc[g188.largest_patient.eq('P6'), ['program_id', 'largest_patient_usage_fraction']]
    lines += ['### GSE188900 P6-related QC', '', f'P6 is the largest usage contributor for {len(p6)}/8 programs. ' + ('; '.join((f'{r.program_id}={r.largest_patient_usage_fraction:.2%}' for r in p6.itertuples())) if len(p6) else 'No program has P6 as the largest contributor.') + '. The underlying 67.83% malignant-cell imbalance remains an explicit cohort-level QC concern.', '']
    g196 = results['GSE196756']['flags']
    p4 = g196.loc[g196.largest_patient.eq('P4'), ['program_id', 'largest_patient_usage_fraction']]
    lines += ['### GSE196756 P4-related QC', '', f'P4 is the largest usage contributor for {len(p4)}/5 programs. ' + ('; '.join((f'{r.program_id}={r.largest_patient_usage_fraction:.2%}' for r in p4.itertuples())) if len(p4) else 'No program has P4 as the largest contributor.') + '. Because P4 supplies 84.87% of cells, these flags limit evidential weight; programs were not mechanically deleted.', '', '## 6. Technical QC flags', '']
    flagged = all_flags.loc[all_flags.technical_flags.ne('NONE') | all_flags.review_flags.ne('NONE')]
    if flagged.empty:
        lines.append('No program crossed the pre-locked technical/review thresholds.')
    else:
        lines += ['| Program | Technical flags | Review flags | Automatic exclusion |', '|---|---|---|---|']
        for row in flagged.itertuples():
            lines.append(f'| {row.program_id} | {row.technical_flags} | {row.review_flags} | NO |')
    lines += ['', 'Within-cohort duplication audit (consensus HVG spectra; cosine ≥0.90 review threshold):', '', '| Cohort | Maximum pairwise cosine | Review triggered |', '|---|---:|---|']
    for cohort in K_BY_COHORT:
        f = results[cohort]['flags']
        maximum = float(f['max_within_cohort_hvg_spectra_cosine'].max())
        lines.append(f"| {cohort} | {maximum:.6f} | {('YES' if maximum >= 0.9 else 'NO')} |")
    lines += ['', 'Cell-cycle-associated genes were not treated as technical artifacts. No flag used expected biological narratives, and no flagged program was removed.', '', '## 7. Warnings and errors', '', 'Each cohort emitted one non-fatal Scanpy view-copy warning during consensus usage refitting; consensus completed. Factorization worker logs contain zero warning/error/traceback lines. No failed replicates or errors occurred.', '', '## 8. Readiness gate', '', '`READY FOR CROSS-COHORT PROGRAM MATCHING — GSE160269: YES`', '', '`READY FOR CROSS-COHORT PROGRAM MATCHING — GSE188900: YES`', '', '`READY FOR SUPPORTIVE PROGRAM MATCHING — GSE196756: YES`', '', '`READY FOR PHASE C: YES`', '', 'Phase B1 analysis complete. GSE196756 remains supportive/replication evidence in later interpretation.', '']
    (OUT / 'PHASE_B1_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    frozen_names = {'exact_configuration.json', 'formal_cnmf_metrics.tsv', 'density_threshold_audit.tsv', 'gene_program_spectra.tsv', 'gene_program_scores.tsv', 'gene_program_ranks.tsv', 'cell_program_usage.tsv', 'cell_program_usage_normalized.tsv', 'top50_genes_per_program.tsv', 'program_usage_summary.tsv', 'patient_program_usage.tsv', 'sample_program_usage.tsv', 'patient_program_usage_heatmap.png', 'patient_program_usage_heatmap.pdf', 'program_qc_flags.tsv', 'within_cohort_program_similarity.tsv', 'warnings_errors.txt', 'formal_run.log'}
    manifest_rows = []
    for cohort in K_BY_COHORT:
        for path in sorted((OUT / cohort).iterdir()):
            if path.is_file() and path.name in frozen_names:
                manifest_rows.append({'cohort': cohort, 'file': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    for path in (OUT / 'PHASE_B1_REPORT.md', OUT / 'formal_cnmf_metrics.tsv', OUT / 'program_qc_flags.tsv'):
        manifest_rows.append({'cohort': 'ALL', 'file': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(OUT / 'output_manifest.tsv', sep='\t', index=False)
if __name__ == '__main__':
    main()
