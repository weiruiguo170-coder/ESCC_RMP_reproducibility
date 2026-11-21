"""Reconstruct and check the complete parameter grid without storing a large cache."""
from pathlib import Path
import sys,copy,csv,hashlib,importlib.util,itertools,json,io,argparse
_root=next(p for p in Path(__file__).resolve().parents if (p/'scripts/_paths.py').exists())
sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,OUTPUT_ROOT,data_path,read_table,resolve_data
spec=importlib.util.spec_from_file_location('grid_rules',ROOT/'scripts/evidence/decision_rules.py')
rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
AXES = {'match_q': [0.01, 0.05, 0.1], 'match_metrics': [2, 3], 'dominance': [0.65, 0.7, 0.8, 0.85, 0.9], 'rna_min_n': [3, 5, 8, 10], 'rna_patient_coverage': [0.5, 0.6, 0.7, 0.8], 'rna_positive': [0.6, 0.7, 0.8], 'protein_coverage': [0.5, 0.6, 0.7, 0.8], 'evidence_q': [0.01, 0.05, 0.1]}
QC = ['STANDARD', 'LOOSE', 'STRICT', 'STANDARD_REMOVE_DOUBLET_LIKE_OUTLIERS']
def generate(records, qc_rows, config, identity, write_full=None):
    """Yield the exact original CSV bytes and summary counts without storing a cache."""
    thresholds = config['thresholds']
    paths = list(itertools.product(*AXES.values()))
    default = tuple((thresholds[k] for k in AXES))
    qc = {(r['condition'], r['anonymous_id'], r['size'], r['cohort']): r for r in qc_rows}
    total = 0
    hasher = hashlib.sha256()
    summaries = []
    rows_total = 0
    state = lambda value: 'U' if value is None else 'YES' if value else 'NO'
    fields = ['path_id', 'RMP', 'size', 'QC', 'D', 'R_caution', 'Q', 'P', 'Q_n_cohorts', 'Q_detail', 'category', 'joint_only_change']
    import pandas as pd
    for record in records:
        rmp = identity[record['anonymous_id']]
        baseline = rules.classify(record, 'B', config)['category']
        for condition in QC:
            branch = copy.deepcopy(record)
            if condition != 'STANDARD':
                for e in branch['external']:
                    item = qc[condition, record['anonymous_id'], record['size'], e['cohort']]
                    e['coherence'] = item['coherence']
                    if not item['same_as_standard']:
                        e['old'] = None
            condition_base = rules.classify(branch, 'B', config)['category']
            cache = {k: {} for k in ['D', 'R', 'Q', 'P']}
            rows = []
            origin = {}
            for (pid, values) in enumerate(paths, 1):
                (md, mk, rr, nn, frac, qp, pc, eq) = values
                c = dict(thresholds, **dict(zip(AXES, values)))
                for (name, key, fn) in [('D', (md, mk), rules.discovery), ('R', rr, lambda r, c: rules.concentration(r, c, 'BOTH')), ('Q', (nn, frac, qp, eq), rules.rna), ('P', (pc, eq), rules.protein)]:
                    if key not in cache[name]:
                        cache[name][key] = fn(branch, c)
                d = cache['D'][md, mk]
                risk = cache['R'][rr]
                (q, detail, nc) = cache['Q'][nn, frac, qp, eq]
                p = cache['P'][pc, eq]
                if d is None or (d and risk is None):
                    category = 'INDETERMINATE'
                elif not d or risk:
                    category = 'Exploratory'
                elif q is None or (q and p is None):
                    category = 'INDETERMINATE'
                elif not q:
                    category = 'Exploratory'
                else:
                    category = 'Primary' if p else 'Secondary'
                origin[values] = category
                rows.append([pid, rmp, record['size'], condition, state(d), state(risk), state(q), state(p), nc, detail, category])
            one = {(i, v): origin[default[:i] + (v,) + default[i + 1:]] for (i, axis) in enumerate(AXES.values()) for v in axis}
            for (row, values) in zip(rows, paths):
                row.append(row[-1] != condition_base and all((one[i, v] == condition_base for (i, v) in enumerate(values))))
            frame = pd.DataFrame(rows, columns=fields)
            b = frame.to_csv(sep='\t', index=False, header=not total, lineterminator='\n').encode()
            hasher.update(b)
            total += len(b)
            rows_total += len(rows)
            if write_full:
                write_full.write(b)
            eq = frame.category.eq(baseline)
            det = frame.category.ne('INDETERMINATE')
            summaries.append(dict(RMP=rmp, size=record['size'], QC=condition, paths=len(rows), baseline_retained=int(eq.sum()), determinate=int(det.sum()), all_fraction=float(eq.mean()), determinate_fraction=float(eq.sum() / det.sum())))
    return (summaries, {'bytes': total, 'rows': rows_total, 'sha256': hasher.hexdigest()})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-full',type=Path,help='Optional 119-MiB full TSV; keep this generated cache outside Git')
    args=parser.parse_args()
    records=json.loads(data_path('anonymous_records.json').read_text())
    qc=json.loads(data_path('anonymous_QC_records.json').read_text())
    cfg=json.loads(data_path('E1_RULES_CONFIG.json').read_text())
    key=read_table('IDENTITY_KEY.tsv').set_index('anonymous_id').RMP.to_dict()
    stream=None
    if args.write_full:args.write_full.parent.mkdir(parents=True,exist_ok=True);stream=args.write_full.open('wb')
    try:summary,integrity=generate(records,qc,cfg,key,stream)
    finally:
        if stream:stream.close()
    expected=json.loads((ROOT/'data/metadata/evidence/parameter_grid_integrity.json').read_text())
    for k in ['bytes','rows','sha256']:assert integrity[k]==expected[k],(k,integrity[k],expected[k])
    import pandas as pd,numpy as np
    observed=pd.DataFrame(summary).sort_values(['RMP','size','QC']).reset_index(drop=True)
    frozen=read_table('parameter_grid_summary.tsv').sort_values(['RMP','size','QC']).reset_index(drop=True)
    assert observed[['RMP','size','QC']].equals(frozen[['RMP','size','QC']])
    for col in ['paths','baseline_retained','determinate','all_fraction','determinate_fraction']:
        assert np.allclose(observed[col],frozen[col],atol=1e-13)
    print('967680 grid rows reconstructed; exact original bytes and all 56 summaries verified.')
if __name__=='__main__':main()
