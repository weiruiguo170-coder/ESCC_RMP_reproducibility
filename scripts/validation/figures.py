"""Validate final figure inputs and load display metadata without rendering."""
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/_paths.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from _paths import data_path,read_table,dataset_bytes,OUTPUT_ROOT

import ast,pandas as pd
index=pd.read_csv(ROOT/'docs/figures.tsv',sep='\t')
assert set(index.Figure_ID)=={f'Fig{i}' for i in range(1,10)}|{f'FigS{i}' for i in range(1,14)}
for r in index.itertuples():
    for name in r.Source_Data.split(';'):assert (ROOT/name).is_file(),name
    ast.parse((ROOT/r.Plotting_Script).read_text());assert len(r.Submitted_PNG_SHA256)==64
sys.path.insert(0,str(ROOT/'scripts/plotting'))
import workflow,cohorts
x,parameters=workflow.load_data(ROOT/'data/processed/figures')
assert x['GSE188900'][:2]==(6,6801) and x['GSE160269'][:2]==(60,44547)
cells,fits,summary=cohorts.load_data(ROOT/'data/processed/figures');assert len(summary)==7
conditions=read_table('condition_summary.tsv')
threshold=conditions[conditions.family.eq('threshold alternatives')]
qc=conditions[conditions.family.str.contains('QC',case=False)]
assert threshold.changed.sum()==17 and threshold.rows.sum()==280
assert qc.changed.sum()==6 and qc.rows.sum()==126
mask=read_table('domain_masking_summary.tsv')
assert set(mask.domain)=={'PATIENT','Q','TECH','CONTEXT','PROTEIN','PDO'}
assert mask.cross_claim_status_transitions.eq(0).all()
print('All 22 final figure mappings and display inputs verified; no images rendered.')
