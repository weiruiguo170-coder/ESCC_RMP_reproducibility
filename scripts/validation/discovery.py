"""Reconstruct all 80 observed matching statistics and source shares from frozen fits."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent;ROOT=REPO_ROOT
checks=[]
def ck(name,a,b,tol=1e-10):
    a,b=np.broadcast_arrays(np.asarray(a),np.asarray(b))
    err=float(np.max(abs(a-b)));assert err<=tol,(name,err,tol)
    checks.append([name,a.size,err,tol])
tables=ROOT/'data/processed';fitted={}
for co in ['GSE160269','GSE188900']:
    d=ROOT/'data/processed/discovery'/co
    mapping=pd.read_csv(resolve_data(d/'gene_symbol_map.tsv'),sep='\t').set_index('source_gene').gene_symbol
    fitted[co]={}
    for name,mode in [('scores','mean'),('spectra','sum')]:
        x=pd.read_csv(resolve_data(d/f'gene_program_{name}.tsv'),sep='\t',index_col=0)
        original=x.index.astype(str)
        names=mapping.reindex(original).fillna(pd.Series(original,index=original)).astype(str).str.strip().str.upper()
        x.index=names.to_numpy();x=x[x.index!='']
        fitted[co][name]=x.groupby(level=0,sort=True).agg(mode)
    usage=pd.read_csv(resolve_data(d/'cell_program_usage_normalized.tsv'),sep='\t',index_col=0)
    obs=pd.read_csv(resolve_data(d/'cell_source_labels.tsv'),sep='\t').set_index('cell_id').loc[usage.index]
    ck(co+' normalized usage',usage.sum(axis=1),1,2e-6)
    source=pd.read_csv(resolve_data(tables/'S3_unit_level_contributions.tsv'),sep='\t').query('cohort==@co')
    for mode,aggregator,col in [('RAW','sum','exact_RAW_contribution'),('EQUAL','mean','EQUAL_contribution')]:
        aggregate=usage.groupby(obs.patient_id).agg(aggregator)
        shares=aggregate/aggregate.sum(axis=0)
        observed=np.array([shares.loc[r.unit_id,r.program_id] for r in source.itertuples()])
        ck(co+' '+mode+' source shares',observed,source[col],3e-7)
a,b=fitted.values()
genes=sorted(set(a['scores'].index)&set(a['spectra'].index)&set(b['scores'].index)&set(b['spectra'].index))
ck('Primary shared universe',len(genes),16143,0)
sa,sb=a['scores'].loc[genes].to_numpy(),b['scores'].loc[genes].to_numpy()
ra,rb=rankdata(sa,axis=0),rankdata(sb,axis=0)
za,zb=(ra-ra.mean(0))/ra.std(0),(rb-rb.mean(0))/rb.std(0)
rho=za.T@zb/len(genes)
pa,pb=np.maximum(a['spectra'].loc[genes].to_numpy(),0),np.maximum(b['spectra'].loc[genes].to_numpy(),0)
pa=pa/pa.sum(0);pb=pb/pb.sum(0)
pa=pa/np.sqrt((pa*pa).sum(0));pb=pb/np.sqrt((pb*pb).sum(0))
cos=pa.T@pb
def top(x,n):
    out=np.zeros(x.shape,dtype=np.int16)
    for j in range(x.shape[1]):out[np.argpartition(x[:,j],-n)[-n:],j]=1
    return out
ov=top(sa,50).T@top(sb,50);s=(rho.clip(min=0)+cos+ov/50)/3
reference=pd.read_csv(resolve_data(tables/'S3_all_80_pairs.tsv'),sep='\t')
rows=[]
for r in reference.itertuples():
    i=a['scores'].columns.get_loc(r.program_A);j=b['scores'].columns.get_loc(r.program_B)
    rows.append([r.program_A,r.program_B,rho[i,j],cos[i,j],ov[i,j],s[i,j]])
rows=pd.DataFrame(rows,columns=['program_A','program_B','spearman_rho','cosine_similarity','top50_overlap','composite_similarity'])
for c in rows.columns[2:]:ck('80 fitted '+c,rows[c],reference[c],2e-10)
margins=pd.read_csv(resolve_data(tables/'S3_matching_margins.tsv'),sep='\t')
for r in margins.itertuples():
    i=a['scores'].columns.get_loc(r.program_A);j=b['scores'].columns.get_loc(r.program_B)
    ck(r.rmp_id+' margin A',s[i,j]-np.max(np.delete(s[i,:],j)),r.matching_margin_A)
    ck(r.rmp_id+' margin B',s[i,j]-np.max(np.delete(s[:,j],i)),r.matching_margin_B)
out=OUTPUT_ROOT;out.mkdir(exist_ok=True)
rows.to_csv(out/'reconstructed_80_matches.tsv',sep='\t',index=False)
pd.DataFrame(checks,columns=['check','values','maximum_absolute_error','tolerance']).to_csv(out/'discovery_reconstruction_checks.tsv',sep='\t',index=False)
print(len(checks),'discovery matrix, source-share and matching-margin checks passed.')
