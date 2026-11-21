"""Recompute revision endpoints from the shipped patient, protein, permutation and parameter records."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
from itertools import permutations, product, islice
import json, math, importlib.util, sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
TAB=ROOT/'results/tables'

C=B=D=E=ROOT/'data/processed/evidence'
out=Path(sys.argv[1]) if len(sys.argv)>1 else OUTPUT_ROOT
out.mkdir(parents=True,exist_ok=True)
def read(p):return pd.read_csv(resolve_data(p),sep='\t')
def save(x,name):x.to_csv(out/(name+'.tsv'),sep='\t',index=False,float_format='%.15g')
checks=[]
def verify(name,actual,expected,tol=1e-9):
    actual,expected=np.broadcast_arrays(np.asarray(actual,float),np.asarray(expected,float))
    assert np.array_equal(np.isnan(actual),np.isnan(expected)),name
    err=float(np.nanmax(abs(actual-expected)))
    assert err<=tol,(name,err)
    checks.append([name,actual.size,err,tol])
def adj(values,mode):
    p=np.array(values);i=np.argsort(p);n=len(p);v=p[i]
    z=(np.maximum.accumulate(v*np.arange(n,0,-1)) if mode=='holm' else np.minimum.accumulate((v*n/np.arange(1,n+1))[::-1])[::-1])
    return np.minimum(1,z)[np.argsort(i)]
prof=read(C/'seven_RMP_profiles.tsv').set_index(['RMP','size']).sort_index()
pairs=read(TAB/'S3_all_80_pairs.tsv')
verify('Discovery BH80',adj(pairs.permutation_p,'bh'),pairs.permutation_FDR)
r=pairs[pairs.RBH & pairs.statistically_supported_edge]
verify('Reciprocal best match magnitudes',np.sort(r.composite_similarity),np.sort(prof.xs(50,level=1).D_match_similarity))
units=read(TAB/'S3_unit_level_contributions.tsv')
units=units[units.rmp_id.isin(prof.index.get_level_values('RMP'))]
for co,g in units.groupby('cohort'):
    for mode,col in [('RAW','aggregate_normalized_usage'),('EQUAL','mean_normalized_usage')]:
        share=g[col]/g.groupby('rmp_id')[col].transform('sum')
        exact=share.groupby(g.rmp_id).max()
        key=co+('_alias' if co=='GSE188900' else '')+'_'+mode
        verify(key,exact,prof.xs(50,level=1)[key],2e-7)
grid=read_table('parameter_grid_summary.tsv').query("QC=='STANDARD'")
bridge=read_table('table_B1_grid_reconciliation_exact.tsv').set_index(['RMP','size']).sort_index()
ret=[]
for k,row in grid.set_index(['RMP','size']).sort_index().iterrows():
    ret.append([*k,row.paths,row.baseline_retained,row.determinate,row.all_fraction,row.determinate_fraction])
ret=pd.DataFrame(ret,columns=['RMP','size','all','retained','determinate','all_fraction','determinate_fraction']).set_index(['RMP','size'])
verify('All paths denominator',ret.all_fraction,bridge.reference_retention_all)
verify('Determinate denominator',ret.determinate_fraction,bridge.reference_retention_determinate_only)
e1=ROOT/'data/metadata/evidence'
module=importlib.util.spec_from_file_location('baseline',ROOT/'scripts/evidence/decision_rules.py')
frozen=importlib.util.module_from_spec(module);module.loader.exec_module(frozen)
cfg=json.loads(data_path('E1_RULES_CONFIG.json').read_text())
key=read(data_path('IDENTITY_KEY.tsv')).set_index('anonymous_id').RMP
common=[]
for item in json.loads(data_path('anonymous_records.json').read_text()):
    item['external']=[x for x in item['external'] if x['cohort']=='external_1']
    k=(key[item['anonymous_id']],item['size']);g=item['discovery'];q=item['external'][0]['coherence'];p=item['protein']
    for mq,dom,pos,pc,eq in product([.01,.05,.1],[.7,.8,.85,.9],[.6,.7],[.5,.6,.7],[.01,.05,.1]):
        ts=dict(cfg['thresholds'],match_q=mq,match_metrics=2,dominance=dom,rna_min_n=5,rna_patient_coverage=.5,rna_positive=pos,protein_coverage=pc,evidence_q=eq)
        y=frozen.classify(item,'B',dict(cfg,thresholds=ts))['category']
        gate=g['q']<mq and g['rbh'] and g['no_technical'] and g['metric_count']>=2
        concentrated=max(g['raw_max'],g['equal_max'])>=dom
        rna=(q['n_patients_evaluable']>=max(5,.5*q['n_patients_original']) and q['positive_patient_fraction']>=pos-1e-12 and q['BH_FDR']<eq and q['CI_low']>0 and q['mean_patient_effect']>0)
        tissue=p['coverage']>=pc-1e-12 and p['matched_null_FDR']<eq and (p['CI_low']>0 or p['CI_high']<0)
        other=('Primary' if tissue else 'Secondary') if gate and not concentrated and rna else 'Exploratory'
        common.append([*k,mq,eq,dom,pos,pc,mq==eq,y,other,y==other])
grid2=pd.DataFrame(common,columns=['RMP','size','match_q','evidence_q','dominance','positive','protein_cov','shared_cutoff','frozen','reimplementation','agree'])
verify('216 harmonized paths x 14',grid2.agree,1);save(grid2,'harmonized_grid');save(ret.reset_index(),'retention')
obs=read(C/'real_patient_measurements.tsv')
draws=read(E/'real_q_background_draws.tsv')
gid=['cohort','patient_id','signature_key','method','background']
z=draws.merge(obs[gid+['target_corr']],on=gid,validate='many_to_one')
z=z[np.isfinite(z.statistic)].copy()
z['extreme']=z.statistic>=z.target_corr-1e-12
by=z.groupby(gid).agg(B=('statistic','size'),above=('extreme','sum'),nullmean=('statistic','mean')).reset_index().merge(obs,on=gid)
by['tail']=(by.above+1)/(by.B+1)
verify('Patient empirical tail',by['tail'],by.patient_p)
verify('Patient excess',by.target_corr-by.nullmean,by.relative_q)
gids=['cohort','signature_key','method','background'];rec=[]
for k,g in by.groupby(gids):
    N=14 if k[0]=='GSE199654' else 4
    row=dict(zip(gids,k));row.update(n=len(g),N=N,Q=g.relative_q.mean(),size=int(k[1].split('TOP')[1]))
    order=np.sort(np.r_[g['tail'],np.ones(N-len(g))])
    for u,label in [(2,'repeated'),(N//2+1,'majority')]:
        if len(g)<max(5,math.ceil(N/2),u):continue
        row['p_'+label]=min(1,(N-u+1)*order[u-1])
        h=order[u-1:];t=-np.log(h).sum()
        row['p_fisher_'+label]=np.exp(-t)*sum(t**j/math.factorial(j) for j in range(len(h)))
    rec.append(row)
pc=pd.DataFrame(rec); fam=['cohort','size','method','background']
for field in [x for x in pc.columns if x.startswith('p_')]:
    pc[field.replace('p_','holm_',1)]=pc.groupby(fam)[field].transform(lambda p:adj(p,'holm'))
expected=read(C/'real_Q_claims_by_spec.tsv')
check=pc.merge(expected,on=gids,suffixes=('_replay','_source'))
verify('Cohort Q',check.Q,check.Q_effect)
for fld in [x for x in pc.columns if x.startswith(('p_','holm_'))]: verify(fld,check[fld+'_replay'],check[fld+'_source'])
save(pc,'patient_conjunction')
prot=read(TAB/'S7_paired_proteomic_scores.tsv')
genes=read(TAB/'S7_Liu_gene_effects.tsv')
for (r,s),g in prot.groupby(['RMP','signature']):
    k=(r,int(s[3:]));di=g.raw_paired_delta
    verify(r+s+' paired dz',di.mean()/di.std(),prof.loc[k,'P_dz'])
    v=genes[(genes.RMP==r)&(genes.signature==s)].mean_delta_raw_T_minus_N.dropna()
    q=read(D/'table_D_final_7RMP_claim_conditioned_profiles.tsv').set_index(['RMP','signature_size'])
    verify(r+s+' DCI',abs(v.sum())/v.abs().sum(),q.loc[k,'P_DCI'])
    verify(r+s+' coverage',len(v)/k[1],prof.loc[k,'P_coverage'])
pdo=read(TAB/'S8_PDO_scores_endpoints.tsv')
res=read(TAB/'S8_PDO_exact_results.tsv').query("subset=='All10_author_AUC' and family.str.startswith('RMP_7')",engine='python').sort_values(['signature','program_or_control'])
# No ties in these 14 rankings. The distribution of rank dot products is invariant to relabeling.
ranks=np.arange(1,11)*2-11;it=permutations(ranks.tolist());hist=np.zeros(331,dtype=np.int64)
while chunk:=list(islice(it,40000)):
    hist+=np.bincount(np.abs(np.array(chunk)@ranks),minlength=len(hist))
for size,grp in res.groupby('signature'):
    tail=[]
    for row in grp.itertuples():
        x=pdo[row.score_column].rank().to_numpy()*2-11;y=pdo.author_AUC.rank().to_numpy()*2-11
        assert len(set(x))==len(set(y))==10
        dot=int(x@y);verify(row.score_column+' rho',dot/330,row.rho);tail.append(hist[abs(dot):].sum()/math.factorial(10))
    verify(size+' exact enumeration',tail,grp.P_exact_two_sided);verify(size+' PDO BH7',adj(tail,'bh'),grp.FDR_seven_exact)
sim=read(C/'controlled_Q_claims_by_world.tsv').query("method=='RAW_RANK' and missingness=='STANDARD_SCOPE'")
assert len(sim[['mechanism','world']].drop_duplicates())==480
sim=sim.assign(pc8=sim.MAJORITY_bonferroni_999_holm.le(.05),majority_truth=sim.truth_finite_n_positive.ge(8))
performance=sim.groupby(['mechanism','program','majority_truth']).agg(worlds=('world','size'),old=('FROZEN_Q_supported','mean'),effect=('effect_decision','mean'),pc8=('pc8','mean')).reset_index()
save(performance,'simulation_performance')
paired=read(C/'controlled_sensitivity_depth_pairs.tsv')
rates=paired.groupby(['mechanism','sensitive_at1x']).raw_changed_at4x.mean().unstack()
published=read(C/'controlled_sensitivity_depth_predictiveness.tsv').set_index('mechanism')
verify('Acquisition sensitivity',rates[True],published.raw_switch_given_sensitive);verify('Stable acquisition',rates[False],published.raw_switch_given_stable)
save(pd.DataFrame(checks,columns=['test','values','maximum_absolute_error','tolerance']),'verification')
print(f'{len(checks)} checks passed. {len(by)} patient measurements, {len(pc)} cohort specifications, {len(grid2)} grid comparisons.')


