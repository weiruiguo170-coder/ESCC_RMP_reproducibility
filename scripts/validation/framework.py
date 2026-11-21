"""Reconstruct strategy coverage and domain-mask summaries from every claim record."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'data';checks=[]
claims=pd.read_csv(resolve_data(DATA/'S5_ALL7_STRATEGY_COMPARISON.tsv'),sep='\t')
for (size,strategy),g in claims.groupby(['signature_size','strategy']):
    expected={'M1':7,'M2':21,'M3':28,'M4':49}[strategy]
    assert len(g)==56 and g.RMP.nunique()==7 and g.claim.nunique()==8
    assert int(g.status.ne('NOT_EVALUABLE').sum())==expected
    checks.append([size,strategy,'evaluable_claims',expected])
mask=pd.read_csv(resolve_data(DATA/'S5_framework_leave_one_domain_out.tsv'),sep='\t')
summary=pd.read_csv(resolve_data(DATA/'S5_framework_ablation.tsv'),sep='\t').set_index(['signature_size','domain'])
for (size,domain),g in mask.groupby(['signature_size','masked_domain']):
    assert len(g)==56 and g.RMP.nunique()==7 and g.claim.nunique()==8
    actual={
      'A_own_claim_cells':int(g.increment_category.eq('A_OWN_CLAIM_COVERAGE').sum()),
      'B_other_claim_boundary_cells':int(g.increment_category.eq('B_OTHER_CLAIM_BOUNDARY').sum()),
      'C_unchanged_cells':int(g.increment_category.eq('C_NO_IDENTIFIABLE_CHANGE').sum()),
      'status_transitions':int(g.status_changed.sum()),
      'boundary_transitions':int(g.boundary_changed.sum()),
      'cross_claim_status_transitions':int(g.loc[g.increment_category.ne('A_OWN_CLAIM_COVERAGE'),'status_changed'].sum())}
    for field,value in actual.items():
        expected=int(summary.loc[(size,domain),field]);assert value==expected,(size,domain,field,value,expected)
        checks.append([size,domain,field,value])
pd.DataFrame(checks,columns=['size','strategy_or_domain','check','value']).to_csv(OUTPUT_ROOT/'framework_reconstruction_checks.tsv',sep='\t',index=False)
print(len(checks),'strategy and domain-mask summary checks passed from 56-cell claim ledgers.')
