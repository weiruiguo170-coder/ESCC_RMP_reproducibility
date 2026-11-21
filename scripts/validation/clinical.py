"""Verify the complete 21-test clinical family and the two seven-program Cox families."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table
from pathlib import Path
import numpy as np,pandas as pd
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/'data';rows=[]
x=pd.read_csv(resolve_data(DATA/'S6_clinical_endpoints.tsv'),sep='\t')
g=x[x.status.eq('EVALUATED_EXPLORATORY')]
assert len(g)==21 and g.RMP.nunique()==7 and g.endpoint.nunique()==3
assert np.allclose(multipletests(g.P,method='fdr_bh')[1],g.BH_FDR,atol=1e-13)
rows.append(['clinical BH21',len(g),float(g.BH_FDR.min())])
blocked=x[x.status.eq('CONDITIONAL_BLOCKED')]
assert len(blocked)==7 and blocked.P.isna().all() and blocked.BH_FDR.isna().all()
rows.append(['Unavailable metastasis retained',len(blocked),np.nan])
os=pd.read_csv(resolve_data(DATA/'S6_OS_Cox_models.tsv'),sep='\t')
for model,g in os.groupby('model'):
    assert len(g)==7 and g.n_patients.eq(179).all() and g.events.eq(106).all()
    assert np.allclose(multipletests(g.P,method='fdr_bh')[1],g.BH_FDR,atol=1e-13)
    assert g.BH_FDR.ge(.05).all()
    rows.append([model+' BH7',len(g),float(g.BH_FDR.min())])
pd.DataFrame(rows,columns=['check','rows','minimum_q']).to_csv(OUTPUT_ROOT/'clinical_family_checks.tsv',sep='\t',index=False)
print(len(rows),'clinical availability and complete-family checks passed.')
