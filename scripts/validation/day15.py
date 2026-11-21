"""Recount exact 9!/12! fixed-dose tails by dynamic programming over labeled assignments."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table
from pathlib import Path
from collections import Counter
from functools import lru_cache
from math import factorial
import numpy as np,pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parent;ROOT=REPO_ROOT
@lru_cache(maxsize=None)
def histogram(left,right):
    n=len(left);states=[Counter() for _ in range(1<<n)];states[0][0]=1
    for assigned,counts in enumerate(states[:-1]):
        value=left[assigned.bit_count()]
        for j,other in enumerate(right):
            bit=1<<j
            if assigned & bit:continue
            target=states[assigned|bit];inc=value*other
            for total,count in counts.items():target[total+inc]+=count
    assert sum(states[-1].values())==factorial(n)
    return states[-1]
scores=pd.read_csv(resolve_data(ROOT/'data/S8_PDO_day15_membership_scores.tsv'),sep='\t')
source=pd.read_csv(resolve_data(ROOT/'data/S8_PDO_day15_exact.tsv'),sep='\t');rows=[]
for (scope,endpoint,size),group in source.groupby(['scope','endpoint','size']):
    selected=scores[scores[endpoint].notna()]
    if group.n.iloc[0]==9:selected=selected[selected.author_AUC.notna()]
    n=len(selected);tail=[]
    for r in group.itertuples():
        x=(2*rankdata(selected[r.RMP.removeprefix('RMP_')+f'_Top{size}'])-n-1).astype(int)
        y=(2*rankdata(selected[endpoint])-n-1).astype(int)
        # Sorting left and permuting the labeled right vector preserves the full null.
        hist=histogram(tuple(sorted(x)),tuple(sorted(y)))
        observed=abs(int(x@y));extreme=sum(v for dot,v in hist.items() if abs(dot)>=observed)
        assert extreme==r.extreme,(scope,endpoint,size,r.RMP,extreme,r.extreme)
        p=extreme/factorial(n);tail.append(p)
        rows.append([scope,endpoint,size,r.RMP,n,extreme,factorial(n),p])
    assert np.allclose(multipletests(tail,method='fdr_bh')[1],group.q7,atol=1e-13)
out=OUTPUT_ROOT;out.mkdir(exist_ok=True)
pd.DataFrame(rows,columns=['scope','endpoint','size','RMP','n','extreme_recounted','permutations','P_exact_recounted']).to_csv(out/'day15_exact_recount.tsv',sep='\t',index=False)
print(len(rows),'Day-15 exact tails recounted; all family q values verified.')
