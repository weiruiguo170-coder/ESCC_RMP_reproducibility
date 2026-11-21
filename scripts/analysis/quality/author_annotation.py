from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
from pathlib import Path
import sys, os, json, importlib.util
R = INPUT_ROOT
O = OUTPUT_ROOT/'quality/cnv'
O.mkdir(parents=True,exist_ok=True)
for _folder in ['analysis','logs','tables','figures']:(O/_folder).mkdir(exist_ok=True)
P = R / 'FIG_R1_5_Malignant_Identity_QC_Validation_20261114'
os.chdir(R)
sys.path.insert(0, str(REPO_ROOT/'scripts/analysis/revision'))
import numpy as np, pandas as pd, anndata as ad
from scipy import sparse
from scipy.stats import rankdata
from coherence_engine import patient_coherence, summarize_coherence
meta = pd.read_csv(resolve_data(INPUT_ROOT / 'references/quality/Supplemetary Data 1.txt'), sep='\t', low_memory=False).set_index('Cell_barcode')
assert meta.index.is_unique
a = ad.read_h5ad(R / '05_phase_E_external_validation/03_processed_external_data/GSE203067_malignant_ready.h5ad')
m = meta.reindex(a.obs_names)
m.insert(0, 'cell_id', a.obs_names)
m['project_patient_id'] = a.obs.patient_id.to_numpy()
m['author_mapped'] = m.sample_id.notna()
m['author_epithelial'] = m.cell_type.eq('Epithelial cells')
m['author_tumor_sample'] = m.sample_id.astype(str).str.startswith('ESCC_T')
m['author_normal_cluster'] = m.subtypes.isin(['C11_Epithelial', 'C18_Epithelial'])
m['retained'] = m.author_epithelial & m.author_tumor_sample & ~m.author_normal_cluster
m.to_csv(O / 'GSE203067_AUTHOR_BARCODE_MAPPING.tsv', sep='\t', index=False)
assert sum(m.author_mapped) == 4641 and sum(m.retained) == 4450
sigs = {}
for i in range(1, 8):
    for n in [50, 100]:
        sigs[f'RMP_C{i:02d}_TOP{n}'] = (R / f'06_rmp_annotation/RMP_C{i:02d}_signature_top{n}.txt').read_text().splitlines()
X = sparse.csr_matrix(a.X)
X.sort_indices()
N = a.n_vars
det = np.asarray((X > 0).sum(0)).ravel() > 0
idx = {k: np.array([j for (j, g) in enumerate(a.var_names) if g in s and det[j]]) for (k, s) in sigs.items()}
score = []
pat = []
nulls = []
member = []
ret = []
for pid in sorted(a.obs.patient_id.astype(str).unique()):
    old = a.obs.patient_id.astype(str).eq(pid).to_numpy()
    keep = old & m.retained.to_numpy()
    ret.append(dict(patient_id=pid, original_cells=int(old.sum()), author_epithelial_cells=int(keep.sum()), removed=int((old & ~keep).sum()), coherence_evaluable_min20=bool(keep.sum() >= 20)))
    for (cond, mask) in [('ORIGINAL', old), ('AUTHOR_CNV_CONTEXT_EPITHELIAL', keep)]:
        pb = np.asarray(X[mask].sum(0)).ravel()
        rr = rankdata(pb)
        medscores = []
        for i in np.flatnonzero(mask):
            (st, en) = X.indptr[i:i + 2]
            ix = X.indices[st:en]
            v = X.data[st:en]
            z = N - len(ix)
            ranks = np.full(N, (z + 1) / 2, dtype=float)
            ranks[ix] = z + rankdata(v)
            medscores.append([np.clip((ranks[ss].mean() - (len(ss) + 1) / 2) / (N - len(ss)), 0, 1) for ss in idx.values()])
        ms = np.median(medscores, axis=0) if len(medscores) else np.full(14, np.nan)
        for (j, (sk, ss)) in enumerate(idx.items()):
            score.append(dict(cohort='GSE203067', condition=cond, patient_id=pid, signature_key=sk, cell_N=int(mask.sum()), median_cell_score=ms[j], pseudobulk_score=np.clip((rr[ss].mean() - (len(ss) + 1) / 2) / (N - len(ss)), 0, 1) if mask.any() else np.nan, status='INSUFFICIENT_PATIENTS_FOR_Q'))
    cp = O / 'logs' / ('AUTHOR_GSE203067_' + pid + '.json')
    if cp.exists():
        stored = json.loads(cp.read_text())
        rows = stored['rows']
        ns = {k: np.array(v) for (k, v) in stored['nulls'].items()}
        assert stored['cell_ids'] == list(a.obs_names[keep])
    else:
        print('Author-supported subset coherence', pid, int(keep.sum()), flush=True)
        (rows, ns, membership) = patient_coherence(X[keep], list(a.var_names), sigs, 'GSE203067' + pid)
        cp.write_text(json.dumps(dict(rows=rows, nulls={k: v.tolist() for (k, v) in ns.items()}, cell_ids=list(a.obs_names[keep]))), encoding='utf-8')
        pd.DataFrame(membership).to_csv(O / 'logs' / ('AUTHOR_' + pid + '_matched_gene_sets.tsv'), sep='\t', index=False)
    for r in rows:
        r.update(cohort='GSE203067', patient_id=pid)
        pat.append(r)
    for (sk, v) in ns.items():
        nulls.extend((dict(cohort='GSE203067', patient_id=pid, signature_key=sk, replicate=j + 1, null_statistic=x) for (j, x) in enumerate(v)))
pd.DataFrame(score).to_csv(O / 'GSE203067_FIXED_SIGNATURE_PATIENT_SENSITIVITY.tsv', sep='\t', index=False)
pd.DataFrame(ret).to_csv(O / 'GSE203067_AUTHOR_SUBSET_RETENTION.tsv', sep='\t', index=False)
pd.DataFrame(pat).to_csv(O / 'GSE203067_AUTHOR_PATIENT_COHERENCE.tsv', sep='\t', index=False)
pd.DataFrame(nulls).to_csv(O / 'GSE203067_AUTHOR_MATCHED_NULL.tsv', sep='\t', index=False)
summary = summarize_coherence(pd.DataFrame(pat), pd.DataFrame(nulls), 'R1_5B')
summary['condition'] = 'AUTHOR_CNV_CONTEXT_EPITHELIAL'
summary['identity_limit'] = 'Author lineage/subcluster plus published CNV context, no per-cell genomic call'
summary.to_csv(O / 'GSE203067_AUTHOR_COHERENCE_SUMMARY.tsv', sep='\t', index=False)
spec = importlib.util.spec_from_file_location('edsc', REPO_ROOT / 'scripts/evidence/classification.py')
ed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ed)
records = json.loads(data_path('anonymous_records.json').read_text())
key = pd.read_csv(resolve_data(R / 'E1_FINAL_RESOLUTION/inputs/IDENTITY_KEY.tsv'), sep='\t').set_index('anonymous_id').RMP.to_dict()
out = []
for r in records:
    before = ed.classify_edsc(r)
    rid = key[r['anonymous_id']]
    v = summary[(summary.rmp_id == rid) & (summary.signature == 'TOP' + str(r['size']))].iloc[0].to_dict()
    for e in r['external']:
        if e['cohort'] == 'external_2':
            e['coherence'] = v
    after = ed.classify_edsc(r)
    out.append(dict(RMP=rid, size=r['size'], baseline_EDSC=before['EDSC_group'], author_subset_EDSC=after['EDSC_group'], impact='UNCHANGED' if before['EDSC_group'] == after['EDSC_group'] else 'CHANGED', reason='GSE203067 original N4 cannot qualify Q; identity evidence improved but not new adequate replication', protein='FROZEN_NOT_REEVALUATED'))
pd.DataFrame(out).to_csv(O / 'AUTHOR_SUBSET_EDSC_IMPACT.tsv', sep='\t', index=False)
print(summary.to_string(index=False), flush=True)
