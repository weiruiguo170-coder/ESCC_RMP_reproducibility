"""Check dataset integrity, frozen signatures, CNV counts and evidence rules."""
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/_paths.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from _paths import data_path,read_table,dataset_bytes,OUTPUT_ROOT

import csv,hashlib,io,itertools,json
import pandas as pd
from evidence.bridge import evidence_bridge
catalog=json.loads((ROOT/'data/metadata/catalog.json').read_text());seen=set();n=0
for key,e in catalog.items():
    identity=tuple(e['files'])
    if identity in seen:continue
    seen.add(identity);raw=dataset_bytes(key)
    assert hashlib.sha256(raw).hexdigest()==e['sha256'],key
    if str(e['files'][0]).endswith('.tsv'):
        rd=csv.reader(io.StringIO(raw.decode('utf-8-sig')),delimiter='\t');header=next(rd);rows=0
        for row in rd:assert len(row)==len(header),(key,rows);rows+=1
        assert (rows,len(header))==(e['rows'],e['columns']),key
    n+=1
sig=read_table('S2_frozen_signatures.tsv');assert len(sig)==1050
for r,g in sig.groupby('rmp_id'):
    ordered={size:x.sort_values('consensus_rank').gene.tolist() for size,x in g.groupby('signature_size')}
    for size in [50,100]:
        assert len(ordered[size])==len(set(ordered[size]))==size
        assert data_path(f'{r}_signature_top{size}.txt').read_text(encoding='utf-8-sig').splitlines()==ordered[size]
    assert ordered[50]==ordered[100][:50]
bridge=read_table('S5_recomputed_audit_bridge.tsv');frozen=read_table('S5_profiles.tsv')
for r in bridge.itertuples():
    expected=frozen[(frozen.RMP==r.RMP)&(frozen.signature_size==r.size)].EDSC_group.iloc[0]
    assert evidence_bridge(dict(D=r.D,R=r.R_warning,Q=r.Q_historical_sign_supported,P=r.P_qualified))==expected
for d,r,q,p in itertools.product([False,True],repeat=4):
    expected='Concentration-cautioned or limited support' if not d or r or not q else 'Joint RNA and proteomic contrast support' if p else 'RNA-domain support without qualified proteomic contrast support'
    assert evidence_bridge(dict(D=d,R=r,Q=q,P=p))==expected
for k in ['D','R','Q','P']:
    x=dict(D=True,R=False,Q=True,P=True);x[k]=None;assert evidence_bridge(x)=='INDETERMINATE'
cnv=read_table('S1_CNV_configurations_by_patient.tsv')
for cfg,counts in [('RAW_AUTO_C2000_W100',[44,30,415]),('LOG_ALL_FULL_W100',[72,6,914]),('LOG_AUTO_FULL_W100',[74,8,888])]:
    assert cnv[cnv.condition==cfg].set_index('patient_id').loc[['P1','P2','P4']].selected_N.tolist()==counts
print(f'{n} unique scientific datasets verified; 14 ordered signatures, CNV counts and complete bridge rules passed.')
