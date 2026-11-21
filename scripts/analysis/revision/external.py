from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
from common import *
from coherence_engine import *
M = '01_external_coherence'
start(M)
sigs = signatures(M)
old = read('05_phase_E_external_validation/06_patient_level_validation/patient_program_scores.tsv', M)
metrics = read('05_phase_E_external_validation/10_tables/program_validation_metrics.tsv', M)
save(metrics, M, 'original_external_metrics_REUSED.tsv')
for p in ['06_patient_level_validation/patient_level_stability.tsv', '07_coherence_and_distinctness/patient_signature_rank_coherence.tsv', '08_sensitivity_top100/top50_top100_consistency.tsv']:
    df = read('05_phase_E_external_validation/' + p, M)
    save(df, M, Path(p).name)
for p in ['gene_overlap_all_metrics.tsv', 'patient_score_correlations_all.tsv']:
    save(read(R1 / '02_distinctness' / p, M), M, p)
patients = []
nulldraw = []
cover = []
score_null = []
for cohort in ['GSE199654', 'GSE203067']:
    a = load_h5(f'05_phase_E_external_validation/03_processed_external_data/{cohort}_malignant_ready.h5ad', M)
    X = sparse.csr_matrix(a.X)
    gn = list(a.var_names)
    lookup = {g: i for (i, g) in enumerate(gn)}
    nullsets = read(f'05_phase_E_external_validation/05_program_scoring/{cohort}_matched_null_gene_sets.tsv', M)
    pb = []
    pids = sorted(a.obs.patient_id.astype(str).unique())
    detection = np.asarray((X > 0).sum(0)).ravel()
    for pid in pids:
        mask = a.obs.patient_id.astype(str).eq(pid).to_numpy()
        block = X[mask]
        pb.append(np.asarray(block.sum(0)).ravel())
        dr = np.asarray((block > 0).sum(0)).ravel()
        for (key, g) in sigs.items():
            cover.append(dict(cohort=cohort, patient_id=pid, signature_key=key, n_cells=int(mask.sum()), platform_genes=sum((x in lookup for x in g)), globally_scored_genes=sum((x in lookup and detection[lookup[x]] > 0 for x in g)), patient_genes_detected=sum((x in lookup and dr[lookup[x]] > 0 for x in g)), patient_aggregation='median/mean cell score and sum-UMI pseudobulk rank score; original scores reused'))
    ranks = rankdata(np.asarray(pb), axis=1)
    for ((rmp, sig), g) in nullsets.groupby(['rmp_id', 'signature_type'], sort=False):
        rr = []
        for r in g.itertuples():
            idx = np.array([lookup[x] for x in r.genes.split(';')])
            scores = score_ranks(ranks, idx)
            rr.append(scores)
            score_null.extend((dict(cohort=cohort, rmp_id=rmp, signature=sig, replicate=r.permutation, patient_id=pid, matched_random_pseudobulk_score=s) for (pid, s) in zip(pids, scores)))
        rr = np.asarray(rr)
        o = old[(old.cohort == cohort) & (old.rmp_id == rmp) & (old.signature_type == sig)].set_index('patient_id').loc[pids]
        assert np.allclose(rr.mean(0), o.matched_null_mean, atol=1e-12) and np.allclose(rr.std(0, ddof=1), o.matched_null_sd, atol=1e-12)
    for pid in pids:
        logging.info('Within-patient coherence %s %s', cohort, pid)
        mask = a.obs.patient_id.astype(str).eq(pid).to_numpy()
        (rows, nulls, membership) = patient_coherence(X[mask], gn, sigs, cohort + pid)
        for row in rows:
            row.update(cohort=cohort, patient_id=pid)
            patients.append(row)
        for (key, ns) in nulls.items():
            nulldraw.extend((dict(cohort=cohort, patient_id=pid, signature_key=key, replicate=i + 1, null_statistic=v) for (i, v) in enumerate(ns)))
        if membership:
            pd.DataFrame(membership).to_csv(O / M / f'{cohort}_{pid}_coherence_random_gene_sets.tsv', sep='\t', index=False)
    save(pd.DataFrame(patients), M, 'patient_coherence_partial_checkpoint.tsv')
score_null = pd.DataFrame(score_null)
score_null.to_csv(O / M / 'original_matched_random_pseudobulk_scores_RECOVERED.tsv', sep='\t', index=False)
patient = pd.DataFrame(patients)
draws = pd.DataFrame(nulldraw)
draws.to_csv(O / M / 'patient_coherence_null_draws.tsv', sep='\t', index=False)
save(patient, M, 'patient_coherence_full.tsv')
summary = summarize_coherence(patient, draws, M)
save(summary, M, 'cohort_coherence_summary.tsv')
cov = pd.DataFrame(cover)
save(cov, M, 'per_patient_signature_coverage.tsv')
save(old.merge(cov.assign(rmp_id=cov.signature_key.str.rsplit('_', n=1).str[0], signature_type=cov.signature_key.str.rsplit('_', n=1).str[1]), on=['cohort', 'patient_id', 'rmp_id', 'signature_type'], validate='one_to_one'), M, 'complete_patient_scores_coverage_REUSED.tsv')
for cohort in ['GSE199654', 'GSE203067']:
    f = patient[(patient.cohort == cohort) & patient.signature_key.str.endswith('TOP50')].pivot(index='patient_id', columns='signature_key', values='effect')
    heat(f, O / M / f'{cohort}_individual_patient_coherence.png', f'{cohort}: patient effect above matched-null', cmap='coolwarm')
finish(M, 'DONE_VERIFIED_WITH_EVALUABILITY_LIMITS', dict(patient_signature_rows=len(patient), evaluated=int((patient.status == 'EVALUABLE').sum()), cohort_results=len(summary), new_coherence_definition=True, original_score_inference_reused=True))
