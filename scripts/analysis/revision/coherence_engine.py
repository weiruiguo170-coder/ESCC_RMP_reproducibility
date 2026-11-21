
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from common import *
def stable_seed(key):return (SEED+int(hashlib.sha256(str(key).encode()).hexdigest()[:8],16))%(2**32)
def joint_matched_sets(target,b1,b2,n=1000,seed=SEED):
 rng=np.random.default_rng(seed);ex=set(target);groups={}
 for i in range(len(b1)):
  if i not in ex:groups.setdefault((int(b1[i]),int(b2[i])),[]).append(i)
 counts=collections.Counter((int(b1[i]),int(b2[i])) for i in target)
 pools=[];fallback=0
 for loc,k in sorted(counts.items()):
  pool=groups.get(loc,[])
  if len(pool)<k:
   fallback+=1;pool=[]
   for other in sorted(groups,key=lambda v:(abs(v[0]-loc[0])+abs(v[1]-loc[1]),v)):
    pool.extend(groups[other])
    if len(pool)>=k:break
  pools.append((np.asarray(pool,int),k,loc))
 sets=[]
 for _ in range(n):
  chosen=[]
  for pool,k,loc in pools:
   use=np.setdiff1d(pool,chosen,assume_unique=False) if fallback else pool
   if len(use)<k:
    expanded=[]
    for other in sorted(groups,key=lambda v:(abs(v[0]-loc[0])+abs(v[1]-loc[1]),v)):
     expanded.extend(i for i in groups[other] if i not in chosen)
     if len(expanded)>=k:break
    use=np.asarray(expanded,int)
   if len(use)<k:raise ValueError('Matched pool insufficient after neighbor expansion')
   chosen.extend(rng.choice(use,k,replace=False))
  sets.append(chosen)
 return np.asarray(sets,int),fallback
import collections
def bins5(v):return np.searchsorted(np.unique(np.quantile(v,[.2,.4,.6,.8])),v,side='right')
def patient_coherence(X,genes,sigs,key):
 X=sparse.csr_matrix(X);n_original=X.shape[0];rng=np.random.default_rng(stable_seed(key))
 if n_original>500:X=X[np.sort(rng.choice(n_original,500,replace=False))]
 n=X.shape[0];gnames=pd.Index(genes);found={g:i for i,g in enumerate(gnames)};rows=[];nulls={};membership=[]
 if n<20:
  for sig,g in sigs.items():rows.append(dict(signature_key=sig,n_cells_original=n_original,n_cells_used=n,status='NOT_EVALUABLE_LT20_CELLS',effective_genes=0,platform_genes=sum(x in found for x in g)))
  return rows,nulls,membership
 totals=np.asarray(X.sum(1)).ravel();log=X.multiply(np.divide(1e4,totals,out=np.zeros_like(totals,dtype=float),where=totals>0)[:,None]).toarray();log=np.log1p(log)
 detection=np.asarray((X>0).sum(0)).ravel();rate=detection/n;sd=log.std(0);eligible=(detection>=3)&(rate>=.05)&(sd>0)
 indices=np.flatnonzero(eligible);lookup={genes[i]:j for j,i in enumerate(indices)}
 ranks=rankdata(log[:,indices],axis=0);ranks-=ranks.mean(0);ranks/=ranks.std(0);Z=ranks.astype(np.float32)
 b1=bins5(rate[indices]);b2=bins5(log[:,indices].mean(0))
 for sig,sg in sigs.items():
  target=np.array([lookup[g] for g in sg if g in lookup]);m=len(target);platform=sum(g in found for g in sg)
  base=dict(signature_key=sig,n_cells_original=n_original,n_cells_used=n,effective_genes=m,platform_genes=platform,coverage_fraction=m/platform if platform else 0)
  if m<10 or m<.2*platform:rows.append(dict(**base,status='NOT_EVALUABLE_GENE_COVERAGE'));continue
  zsum=Z[:,target].sum(1,dtype=np.float64);obs=(np.dot(zsum,zsum)-m*n)/(m*(m-1)*n)
  try:sets,fallback=joint_matched_sets(target,b1,b2,1000,stable_seed(key+'_'+sig))
  except ValueError:
   rows.append(dict(**base,status='NOT_EVALUABLE_INSUFFICIENT_MATCHED_POOL'));continue
  null=np.empty(1000)
  for offset in range(0,1000,50):
   sums=Z[:,sets[offset:offset+50]].sum(2,dtype=np.float64);null[offset:offset+50]=((sums*sums).sum(0)-m*n)/(m*(m-1)*n)
  effect=obs-null.mean();p=(1+(null>=obs-1e-12).sum())/1001
  rows.append(dict(**base,status='EVALUABLE',observed_mean_pairwise_spearman=obs,null_mean=null.mean(),null_sd=null.std(ddof=1),effect=effect,z=effect/null.std(ddof=1) if null.std(ddof=1)>0 else np.nan,P_empirical_positive=p,neighbor_expansion_strata=fallback))
  nulls[sig]=null
  # Preserve exact gene-set identities compactly without duplicating expression.
  for i,ss in enumerate(sets):membership.append(dict(signature_key=sig,replicate=i+1,genes=';'.join(gnames[indices[ss]])))
 return rows,nulls,membership
def summarize_coherence(patient,drawtable,m):
 summaries=[]
 for (cohort,sig),g in patient.groupby(['cohort','signature_key']):
  v=g[g.status=='EVALUABLE'];d=v.effect.to_numpy() if len(v) else np.array([]);alln=g.patient_id.nunique()
  base=dict(cohort=cohort,signature_key=sig,rmp_id=sig.rsplit('_',1)[0],signature=sig.rsplit('_',1)[1],n_patients_original=alln,n_patients_evaluable=len(v))
  if not len(v):summaries.append(dict(**base,coherence_status='NOT_EVALUABLE'));continue
  draw=drawtable[(drawtable.cohort==cohort)&(drawtable.signature_key==sig)].pivot(index='replicate',columns='patient_id',values='null_statistic')[v.patient_id]
  centered=draw.to_numpy()-draw.to_numpy().mean(0);null=centered.mean(1);obs=d.mean();p=(1+(null>=obs-1e-12).sum())/(len(null)+1)
  rng=np.random.default_rng(stable_seed(cohort+sig+'bootstrap'));boot=rng.choice(d,size=(2000,len(d)),replace=True).mean(1);lo,hi=np.quantile(boot,[.025,.975]);loo=np.array([np.delete(d,i).mean() for i in range(len(d))]) if len(d)>1 else np.array([np.nan])
  summaries.append(dict(**base,mean_patient_effect=obs,median_patient_effect=np.median(d),positive_patient_fraction=(d>0).mean(),CI_low=lo,CI_high=hi,P_empirical=p,LOPO_effect_min=np.nanmin(loo),LOPO_effect_max=np.nanmax(loo),LOPO_sign_consistency=np.nanmean(np.sign(loo)==np.sign(obs)),coherence_status='PENDING_FDR'))
 out=pd.DataFrame(summaries)
 for _,idx in out.groupby(['cohort','signature']).groups.items():out.loc[idx,'BH_FDR']=bh(out.loc[idx,'P_empirical'])
 for i,r in out.iterrows():
  if r.coherence_status=='NOT_EVALUABLE':continue
  if r.n_patients_evaluable<5:label='EXPLORATORY_SMALL_N'
  elif r.n_patients_evaluable<.5*r.n_patients_original:label='INSUFFICIENT_PATIENT_COVERAGE'
  elif r.BH_FDR<.05 and r.mean_patient_effect>0 and r.CI_low>0 and r.positive_patient_fraction>=.7:label='REVISION_COHERENCE_SUPPORTED'
  elif r.mean_patient_effect>0:label='MIXED_POSITIVE_EFFECT'
  else:label='NOT_SUPPORTED'
  out.loc[i,'coherence_status']=label
 return out
