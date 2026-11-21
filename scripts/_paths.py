"""Resolve versioned scientific datasets and explicit input/output locations."""
from pathlib import Path
import io,json,os
import pandas as pd
ROOT=REPO_ROOT=Path(__file__).resolve().parents[1]
INPUT_ROOT=Path(os.environ.get('ESCC_INPUT_ROOT',str(ROOT/'data/raw')))
OUTPUT_ROOT=Path(os.environ.get('ESCC_OUTPUT_ROOT',str(ROOT/'results/reproduced')))
_CATALOG=json.loads((ROOT/'data/metadata/catalog.json').read_text(encoding='utf-8'))
def entry(name):
    s=str(name).replace('\\','/')
    for cohort in ['GSE160269','GSE188900']:
        if cohort in s:
            key='discovery/'+cohort+'/'+Path(s).name
            if key in _CATALOG:return _CATALOG[key]
    if s in _CATALOG:return _CATALOG[s]
    base=Path(s).name
    if base in _CATALOG:return _CATALOG[base]
    matches={tuple(v['files']):v for k,v in _CATALOG.items() if Path(k).name==base}
    if len(matches)==1:return next(iter(matches.values()))
    raise KeyError('Dataset is absent or ambiguous: '+s)
def dataset_bytes(name):
    e=entry(name);files=[ROOT/p for p in e['files']]
    return files[0].read_bytes()+b''.join(p.read_bytes().split(b'\n',1)[1] for p in files[1:])
def data_path(name):
    e=entry(name)
    if len(e['files'])!=1:raise ValueError('Partitioned table: use read_table or resolve_data')
    return ROOT/e['files'][0]
def resolve_data(value):
    if not isinstance(value,(str,os.PathLike)):return value
    s=str(value)
    if s.startswith(('https://','http://','ftp://')):return value
    p=Path(value)
    code={'frozen_classifier.py':'scripts/evidence/decision_rules.py','E1_RULES_EXECUTABLE.py':'scripts/evidence/classification.py','coherence_engine.py':'scripts/analysis/revision/coherence_engine.py'}
    if p.name in code:return ROOT/code[p.name]
    if p.exists():return p
    try:e=entry(value)
    except KeyError:return p
    return ROOT/e['files'][0] if len(e['files'])==1 else io.BytesIO(dataset_bytes(value))
def read_table(name,**kwargs):return pd.read_csv(resolve_data(name),sep='\t',**kwargs)
