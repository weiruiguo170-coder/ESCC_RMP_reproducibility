from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
from common import *
from coherence_engine import *
import anndata as ad, ast
M = '02_malignant_qc_balance'
start(M)
sigs = signatures(M)
identity = [dict(cohort='GSE160269', basis='Author tumor epithelial annotation with original study CNV/WGS context; author epithelial boundary retained', direct_per_cell_evidence='author_celltype and malignant_flag_initial; not newly inferred here', label='AUTHOR_ANNOTATED_TUMOR_EPITHELIAL', high_confidence_subset='No independent per-cell CNV scores in retained analysis object; original annotation subset retained', doublet='doublet_like_qc_flag only; not author doublet calls'), dict(cohort='GSE188900', basis='Author malignant squamous-cell labels mapped from original metadata to raw object', direct_per_cell_evidence='author_celltype and malignant_flag_initial', label='AUTHOR_ANNOTATED_MALIGNANT_SQUAMOUS', high_confidence_subset='Original author malignant subset; no additional outcome-blind stricter confidence labels', doublet='doublet_like_qc_flag only; not author doublet calls'), dict(cohort='GSE196756', basis='infercnvpy autosomal sliding window; matched normal epithelial per patient >=50; window100 step10 lfc_clip3 dynamic_threshold1.5; score mean absolute X_cnv; >normal99pct', direct_per_cell_evidence='cnv_score and cnv_threshold_patient_99pct_normal', label='CNV_SUPPORTED_EPITHELIAL', high_confidence_subset='Existing frozen subset all exceed original paired-normal threshold; no new threshold or CNV run', doublet='No credible author labels'), dict(cohort='GSE199654', basis='Author epithelial expression matrix, tumor sampleType T, reproducible QC; no public per-cell CNV or DoubletScore', direct_per_cell_evidence='tumor origin and epithelial annotation only', label='TUMOR_ASSOCIATED_EPITHELIAL', high_confidence_subset='CONDITIONAL_BLOCKED: no independent malignant/CNV confidence labels', doublet='CONDITIONAL_BLOCKED: original DoubletScore unavailable'), dict(cohort='GSE203067', basis='Primary-tumor origin and frozen lineage-marker rule; published study CNV finding is not per-cell mapping in this object', direct_per_cell_evidence='lineage-marker scores and source sample; no original CNV labels', label='TUMOR_ASSOCIATED_EPITHELIAL', high_confidence_subset='CONDITIONAL_BLOCKED: no mapped original CNV/cancer labels', doublet='CONDITIONAL_BLOCKED: no credible author doublet calls')]
save(pd.DataFrame(identity), M, 'malignant_identity_and_conditional_subsets.tsv')
for p in ['04_scripts/gse196756_phase_a_cnv.py', '04_scripts/day2_phase_a.py', '05_phase_E_external_validation/12_scripts/phase_e_preprocess.py', '03_metadata/GSE188900_author_MSE_metadata.csv', '03_metadata/GSE188900_author_SE_metadata.csv']:
    record(p, M, 'identity method / author provenance')
ann = read('06_rmp_annotation/rmp_annotation_table.tsv', M)
balance = []
patientdraw = []
flagrows = []
originalcells = []
for cohort in ['GSE160269', 'GSE188900', 'GSE196756']:
    obj = ad.read_h5ad(record(f'02_data_processed/{cohort}_malignant_ready.h5ad', M), backed='r')
    obs = obj.obs.copy()
    obj.file.close()
    use = read(f'04_cnmf_formal/{cohort}/cell_program_usage_normalized.tsv', M, index_col=0).astype(np.float32)
    obs = obs.loc[use.index]
    groups = {str(p): np.flatnonzero(obs.patient_id.astype(str).to_numpy() == str(p)) for p in sorted(obs.patient_id.astype(str).unique())}
    m = min(200, min(map(len, groups.values())))
    arr = use.to_numpy(float)
    rng = np.random.default_rng(stable_seed(cohort + 'balanced'))
    if cohort == 'GSE196756':
        assert np.all(obs.cnv_score > obs.cnv_threshold_patient_99pct_normal)
    for (pid, ix) in groups.items():
        originalcells.append(dict(cohort=cohort, patient_id=pid, n_cells=len(ix), equal_cells_sampled=m, n_author_confident_cells=len(ix) if cohort != 'GSE160269' else np.nan))
    if 'doublet_like_qc_flag' in obs:
        exclude = obs.doublet_like_qc_flag.astype(str).str.lower().isin(['true', '1']).to_numpy()
        for prog in use:
            means = []
            for (pid, ix) in groups.items():
                keep = ix[~exclude[ix]]
                means.append(np.mean(arr[keep, list(use.columns).index(prog)]) if len(keep) else np.nan)
            val = np.asarray(means)
            frac = val / np.nansum(val)
            flagrows.append(dict(cohort=cohort, program_id=prog, mode='REMOVE_RETAINED_DOUBLET_LIKE_OUTLIERS_NOT_CONFIRMED_DOUBLETS', removed_cells=int(exclude.sum()), patients_retained=int(np.isfinite(val).sum()), equal_max_contribution=np.nanmax(frac)))
    progmap = {getattr(r, cohort + '_program'): r.rmp_id for r in ann.itertuples() if cohort in ['GSE160269', 'GSE188900']}
    for rep in range(300):
        means = np.array([arr[rng.choice(ix, m, replace=False)].mean(0) for ix in groups.values()])
        fracs = means / means.sum(0)
        for (j, prog) in enumerate(use.columns):
            balance.append(dict(cohort=cohort, program_id=prog, RMP=progmap.get(prog, 'SUPPORTIVE_OR_UNMATCHED'), replicate=rep + 1, cells_per_patient=m, n_patients=len(groups), max_contribution=fracs[:, j].max(), dominant_patient=list(groups)[int(fracs[:, j].argmax())], above_original_08=bool(fracs[:, j].max() >= 0.8), precision='LOW_CELL_PRECISION' if m < 20 else 'STANDARD_SENSITIVITY'))
        for (i, pid) in enumerate(groups):
            for (j, prog) in enumerate(use.columns):
                patientdraw.append(dict(cohort=cohort, patient_id=pid, program_id=prog, replicate=rep + 1, mean_normalized_usage=means[i, j], contribution=fracs[i, j]))
save(pd.DataFrame(originalcells), M, 'discovery_patient_cells_and_sampling_m.tsv')
save(pd.DataFrame(balance), M, 'balanced_300_repeat_program_results.tsv')
pd.DataFrame(patientdraw).to_csv(O / M / 'balanced_patient_usage_draws.tsv', sep='\t', index=False)
save(pd.DataFrame(flagrows), M, 'discovery_doublet_like_flag_sensitivity.tsv')
bb = pd.DataFrame(balance)
summary = bb.groupby(['cohort', 'program_id', 'RMP']).agg(cells_per_patient=('cells_per_patient', 'first'), median_max_contribution=('max_contribution', 'median'), lower_025=('max_contribution', lambda x: x.quantile(0.025)), upper_975=('max_contribution', lambda x: x.quantile(0.975)), fraction_ge_08=('above_original_08', 'mean')).reset_index()
save(summary, M, 'balanced_program_summary.tsv')
tiers = []
for r in ann.itertuples():
    a = bb[(bb.cohort == 'GSE160269') & (bb.program_id == r.GSE160269_program)].sort_values('replicate')
    b = bb[(bb.cohort == 'GSE188900') & (bb.program_id == r.GSE188900_program)].sort_values('replicate')
    t = np.maximum(a.max_contribution.to_numpy(), b.max_contribution.to_numpy()) >= 0.8
    tiers.append(dict(RMP=r.rmp_id, original_Tier=r.tier, balanced_fraction_Tier2_by_historical_rule=t.mean(), balanced_fraction_Tier1=1 - t.mean(), Tier_modified=False))
save(pd.DataFrame(tiers), M, 'seven_RMP_Tier_sensitivity.tsv')
src = record('05_phase_E_external_validation/12_scripts/phase_e_preprocess.py', M)
tree = ast.parse(src.read_text(encoding='utf-8'))
names = {'canonical', 'collapse_duplicate_genes', 'qc_metrics', 'read_gse199654'}
defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
for fn in defs:
    if fn.name == 'read_gse199654':
        for node in ast.walk(fn):
            if isinstance(node, ast.Return):
                node.value = ast.Name(id='adata', ctx=ast.Load())
namespace = dict(np=np, pd=pd, sparse=sparse, ad=ad, RAW=R / '05_phase_E_external_validation/02_raw_external_data')
exec(compile(ast.fix_missing_locations(ast.Module(body=defs, type_ignores=[])), str(src), 'exec'), namespace)
record('05_phase_E_external_validation/02_raw_external_data/GSE199654/GSE199654_scTDN_UMI_matrix_epithelial_cells.txt.gz', M)
record('05_phase_E_external_validation/02_raw_external_data/GSE199654/GSE199654_scTDN_METADATA_epithelial_cells.txt.gz', M)
g199 = namespace['read_gse199654']()
g199 = g199[g199.obs.sampleType.astype(str).eq('T')].copy()
g199.obs['patient_id'] = g199.obs.patientID.astype(str)
g199.obs['sample_id'] = g199.obs.sampleID.astype(str)
g203 = load_h5('05_phase_E_external_validation/03_processed_external_data/GSE203067_primary_all_qc.h5ad', M)
g203 = g203[g203.obs.malignant_inclusion.to_numpy()].copy()
oldpatients = read('05_phase_E_external_validation/06_patient_level_validation/patient_program_scores.tsv', M)
qcrows = []
qcsumm = []
qc_coh = []
qc_null = []
retention = []
for (cohort, obj) in [('GSE199654', g199), ('GSE203067', g203)]:
    X = sparse.csr_matrix(obj.X)
    obs = obj.obs
    genes = list(obj.var_names)
    look = {g: i for (i, g) in enumerate(genes)}
    original = load_h5(f'05_phase_E_external_validation/03_processed_external_data/{cohort}_malignant_ready.h5ad', M)
    basedetect = np.asarray((original.X > 0).sum(0)).ravel()
    basegenes = {g for (g, i) in zip(original.var_names, basedetect) if i > 0}
    sigidx = {k: np.array([look[g] for g in v if g in look and g in basegenes]) for (k, v) in sigs.items()}
    cells = np.zeros((X.shape[0], 14), np.float32)
    N = X.shape[1]
    for i in range(len(obs)):
        (st, en) = X.indptr[i:i + 2]
        idx = X.indices[st:en]
        values = X.data[st:en]
        zeros = N - len(idx)
        ranks = np.full(N, (zeros + 1) / 2, dtype=float)
        ranks[idx] = zeros + rankdata(values)
        for (j, (key, ix)) in enumerate(sigidx.items()):
            cells[i, j] = (ranks[ix].mean() - (len(ix) + 1) / 2) / (N - len(ix))
    nullsets = read(f'05_phase_E_external_validation/05_program_scoring/{cohort}_matched_null_gene_sets.tsv', M)
    if cohort == 'GSE199654':
        loose = (obs.n_genes >= 150) & (obs.n_genes < 7000) & (obs.pct_mito < 10)
        standard = (obs.n_genes >= 200) & (obs.n_genes < 6000) & (obs.pct_mito < 5)
        strict = (obs.n_genes >= 300) & (obs.n_genes < 5000) & (obs.pct_mito < 3)
    else:
        loose = (obs.n_counts >= 200) & (obs.n_genes >= 150) & (obs.log10_genes_per_umi > 0.75) & (obs.pct_mito < 25)
        standard = (obs.n_counts >= 300) & (obs.n_genes >= 200) & (obs.log10_genes_per_umi > 0.8) & (obs.pct_mito < 20)
        strict = (obs.n_counts >= 500) & (obs.n_genes >= 300) & (obs.log10_genes_per_umi > 0.85) & (obs.pct_mito < 15)
    assert set(obs.index[standard]) == set(original.obs_names)
    outlier = np.zeros(len(obs), bool)
    for (sample, loc) in obs.groupby('sample_id', observed=True).indices.items():
        for col in ['n_counts', 'n_genes']:
            v = obs.iloc[loc][col].to_numpy()
            med = np.median(v)
            mad = np.median(abs(v - med))
            outlier[loc] |= v > med + 5 * mad if mad > 0 else False
    masks = {'LOOSE': loose.to_numpy(), 'STANDARD': standard.to_numpy(), 'STRICT': strict.to_numpy(), 'STANDARD_REMOVE_DOUBLET_LIKE_OUTLIERS': standard.to_numpy() & ~outlier}
    allp = sorted(original.obs.patient_id.astype(str).unique())
    for (condition, mask) in masks.items():
        logging.info('QC %s %s cells=%s', cohort, condition, mask.sum())
        pids = [p for p in allp if ((obs.patient_id.astype(str) == p).to_numpy() & mask).any()]
        pb = np.array([np.asarray(X[(obs.patient_id.astype(str) == p).to_numpy() & mask].sum(0)).ravel() for p in pids])
        ranks = rankdata(pb, axis=1)
        for p in allp:
            sel = (obs.patient_id.astype(str) == p).to_numpy() & mask
            retention.append(dict(cohort=cohort, condition=condition, patient_id=p, n_cells=int(sel.sum()), relaxation_scope='TRUNCATED_RELAXATION_POST_STANDARD_QC' if cohort == 'GSE203067' and condition == 'LOOSE' else 'QC_PRESELECTION_COUNTS_AVAILABLE'))
        for (j, (key, ix)) in enumerate(sigidx.items()):
            (rmp, sig) = key.rsplit('_', 1)
            nr = nullsets[(nullsets.rmp_id == rmp) & (nullsets.signature_type == sig)]
            random = np.array([score_ranks(ranks, np.array([look[x] for x in rr.genes.split(';')])) for rr in nr.itertuples()])
            sc = score_ranks(ranks, ix)
            z = (sc - random.mean(0)) / random.std(0, ddof=1)
            for (k, p) in enumerate(pids):
                sel = (obs.patient_id.astype(str) == p).to_numpy() & mask
                qcrows.append(dict(cohort=cohort, condition=condition, rmp_id=rmp, signature=sig, patient_id=p, n_cells=int(sel.sum()), median_cell_score=np.median(cells[sel, j]), pseudobulk_score=sc[k], null_z=z[k]))
            qcsumm.append(dict(cohort=cohort, condition=condition, rmp_id=rmp, signature=sig, n_patients=len(pids), n_cells=int(mask.sum()), median_patient_z=np.median(z), positive_patient_fraction=(z > 0).mean(), P_patient_wilcoxon=wilcoxon(z).pvalue if np.any(z) else 1))
        if condition == 'STANDARD':
            continue
        if cohort == 'GSE203067' and np.array_equal(mask, standard.to_numpy()):
            continue
        for p in allp:
            sel = (obs.patient_id.astype(str) == p).to_numpy() & mask
            cache = O / M / 'coherence_checkpoints'
            cache.mkdir(exist_ok=True)
            cp = cache / f'{cohort}_{condition}_{p}.json'
            if cp.exists():
                stored = json.loads(cp.read_text())
                rows = stored['rows']
                ns = {k: np.asarray(v) for (k, v) in stored['nulls'].items()}
            else:
                (rows, ns, sets) = patient_coherence(X[sel], genes, sigs, cohort + condition + p)
                cp.write_text(json.dumps(dict(rows=rows, nulls={k: v.tolist() for (k, v) in ns.items()}), default=lambda x: float(x)), encoding='utf-8')
            for row in rows:
                row.update(cohort=cohort, condition=condition, patient_id=p)
                qc_coh.append(row)
            for (key, v) in ns.items():
                qc_null.extend((dict(cohort=cohort, condition=condition, patient_id=p, signature_key=key, replicate=i + 1, null_statistic=t) for (i, t) in enumerate(v)))
        save(pd.DataFrame(qcrows), M, 'QC_patient_scores_checkpoint.tsv')
q = pd.DataFrame(qcsumm)
for (_, idx) in q.groupby(['cohort', 'condition', 'signature']).groups.items():
    q.loc[idx, 'BH_FDR'] = bh(q.loc[idx, 'P_patient_wilcoxon'])
save(q, M, 'QC_score_matched_null_summary.tsv')
save(pd.DataFrame(qcrows), M, 'QC_patient_scores_full.tsv')
save(pd.DataFrame(retention), M, 'QC_patient_cell_retention.tsv')
save(pd.DataFrame(qc_coh), M, 'QC_patient_coherence.tsv')
pd.DataFrame(qc_null).to_csv(O / M / 'QC_coherence_null_draws.tsv', sep='\t', index=False)
out = []
for (condition, g) in pd.DataFrame(qc_coh).groupby('condition'):
    dr = pd.DataFrame(qc_null)
    s = summarize_coherence(g, dr[dr.condition == condition], M)
    s['condition'] = condition
    out.append(s)
save(pd.concat(out, ignore_index=True), M, 'QC_coherence_cohort_summary.tsv')
(fig, ax) = plt.subplots(figsize=(9, 4))
f = summary[summary.cohort == 'GSE188900'].set_index('RMP').reindex(IDS)
ax.errorbar(np.arange(7), f.median_max_contribution, yerr=[f.median_max_contribution - f.lower_025, f.upper_975 - f.median_max_contribution], fmt='o')
ax.set_xticks(range(7), [r[-3:] for r in IDS])
ax.axhline(0.8, ls='--', color='gray')
ax.set_ylabel('Balanced fixed-usage max patient contribution')
ax.set_title('GSE188900; 300 repetitions, all patients retained')
fig.tight_layout()
fig.savefig(O / M / 'balanced_patient_contribution_distribution.png', dpi=170)
plt.close(fig)
finish(M, 'PARTIAL_CONDITIONAL_SUBSETS_BLOCKED', dict(new_QC_scenarios=8, balanced_repeats_per_cohort=300, new_cNMF=0, high_confidence_external='CONDITIONAL_BLOCKED', confirmed_doublets='NOT_AVAILABLE', QC_scores=len(q)))
