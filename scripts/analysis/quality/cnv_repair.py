
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
import os,json,importlib.util,hashlib
R = INPUT_ROOT;O=OUTPUT_ROOT/'quality/cnv';P=R/'FIG_R1_5_Malignant_Identity_QC_Validation_20261114';os.chdir(R)
O.mkdir(parents=True,exist_ok=True)
for _folder in ['analysis','logs','tables','figures']:(O/_folder).mkdir(exist_ok=True)
import numpy as np,pandas as pd,anndata as ad,scanpy as sc,infercnvpy as cnv
from scipy import sparse
from scipy.stats import spearmanr
from infercnvpy.tl._infercnv import _infercnv_chunk,_get_reference
spec=importlib.util.spec_from_file_location('old',REPO_ROOT/'scripts/analysis/preprocessing/cnv_original.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
# Fixed, bounded design recorded before any new CNV output. No quantile sweep.
configs=[dict(name='RAW_AUTO_C2000_W100',scale='RAW',normgenes='AUTO',chunk=2000,window=100),dict(name='LOG_AUTO_C2000_W100',scale='LOG',normgenes='AUTO',chunk=2000,window=100),dict(name='LOG_AUTO_FULL_W100',scale='LOG',normgenes='AUTO',chunk=100000,window=100),dict(name='LOG_ALL_FULL_W100',scale='LOG',normgenes='ALL',chunk=100000,window=100),dict(name='LOG_ALL_FULL_W50',scale='LOG',normgenes='ALL',chunk=100000,window=50),dict(name='LOG_ALL_FULL_W150',scale='LOG',normgenes='ALL',chunk=100000,window=150)]
param=dict(configs=configs,main='LOG_ALL_FULL_W100',quantile=.99,lfc_clip=3,step=10,dynamic_threshold=1.5,reference='patient-matched normal epithelial >=50',version=cnv.__version__,execution='installed official _get_reference/_infercnv_chunk called serially, same algorithm, no process-map overhead',interpretation='CNV-supported candidate only; unmatched cells undetermined')
pf=O/'analysis/PREDEFINED_CNV_CONFIG.json'
if pf.exists():assert json.loads(pf.read_text())==param
else:pf.write_text(json.dumps(param,indent=2),encoding='utf-8')
a=ad.read_h5ad(R/'02_data_processed/GSE196756_preflight.h5ad');data=a.X.data
raw=dict(shape=list(a.shape),nonzero_entries=len(data),minimum=float(data.min()),maximum=float(data.max()),all_nonnegative=bool((data>=0).all()),all_integral=bool(np.equal(data,np.floor(data)).all()),layers=list(a.layers.keys()),raw_slot=a.raw is not None,uns={k:str(v) for k,v in a.uns.items()});(O/'logs/RAW_INPUT_NUMERIC_AUDIT.json').write_text(json.dumps(raw,indent=2),encoding='utf-8');assert raw['all_integral'] and raw['all_nonnegative']
qc=(a.obs.n_genes_by_counts>=500)&(a.obs.pct_counts_mt<=20);mc=old.marker_count(a);a=a[qc.to_numpy()&(mc>=2)].copy();coords=old.parse_gtf(R/'10_reference/Homo_sapiens.GRCh38.112.gtf.gz',set(a.var.original_gene_id.astype(str).str.split('.').str[0])).set_index('original_gene_id');ids=a.var.original_gene_id.astype(str).str.split('.').str[0]
for c in ['chromosome','start','end']:a.var[c]=ids.map(coords[c]).to_numpy()
mask=(a.var.chromosome.isin(old.AUTOSOMES)&a.var.start.notna()).to_numpy();fulltot=np.asarray(a.X.sum(1)).ravel();autotot=np.asarray(a.X[:,mask].sum(1)).ravel();a=a[:,mask].copy();a.var['start']=a.var.start.astype(int);a.var['end']=a.var.end.astype(int);a.obs['cnv_reference']=np.where(a.obs.tissue.astype(str).eq('adjacent_normal'),'normal_epithelial','tumor_epithelial');a.obs['total_allgenes']=fulltot;a.obs['total_autosomal']=autotot
hist=ad.read_h5ad(R/'02_data_processed/GSE196756_malignant_ready.h5ad',backed='r');hobs=hist.obs.copy();hist.file.close();prev=pd.read_csv(resolve_data(P/'GSE196756_NORMALIZED_CNV_CELL_AUDIT.tsv'),sep='\t').set_index('cell_id');summary=[];allcells=[];chromrows=[];rel=[]
for cfg in configs:
 for pid in sorted(a.obs.patient_id.astype(str).unique()):
  name=cfg['name']+'_'+pid;cp=O/'logs'/(name+'.tsv');npz=O/'logs'/(name+'.npz');b=a[a.obs.patient_id.astype(str).eq(pid)].copy();normal=b.obs.cnv_reference.eq('normal_epithelial').to_numpy();assert normal.sum()>=50;X=sparse.csr_matrix(b.X,dtype=np.float32)
  if cfg['scale']=='LOG':
   totals=b.obs['total_allgenes' if cfg['normgenes']=='ALL' else 'total_autosomal'].to_numpy();X=X.multiply((1e4/totals)[:,None]).tocsr();X.data=np.log1p(X.data)
  b.X=X
  if cp.exists():
   d=pd.read_csv(resolve_data(cp),sep='\t');mat=sparse.load_npz(npz);pos=json.loads((O/'logs'/(name+'_chromosomes.json')).read_text());assert list(d.cell_id)==list(b.obs_names);assert np.allclose(np.asarray(abs(mat).mean(1)).ravel(),d.cnv_score);print('REUSED CHECKED',name,flush=True)
  else:
   print('RUN',name,b.shape,'references',normal.sum(),flush=True);reference=_get_reference(b,'cnv_reference','normal_epithelial',None,None);chunks=[]
   for st in range(0,len(b),cfg['chunk']):
    pos,m,_=_infercnv_chunk(X[st:st+cfg['chunk']],b.var[['chromosome','start','end']],reference,3,cfg['window'],10,1.5,False);chunks.append(m)
   mat=sparse.vstack(chunks).tocsr();scores=np.asarray(abs(mat).mean(1)).ravel();threshold=float(np.quantile(scores[normal],.99));d=b.obs[['patient_id','sample_id','tissue','total_counts','n_genes_by_counts','pct_counts_mt']].copy();d['cell_id']=b.obs_names;d['condition']=cfg['name'];d['reference']=normal;d['cnv_score']=scores;d['threshold_normal99']=threshold;d['selected']=(~normal)&(scores>threshold);d['historical_selected']=d.cell_id.isin(hobs.index);d['R1_5_selected']=prev.loc[d.cell_id,'normalized_cnv_selected'].to_numpy();d.to_csv(cp,sep='\t',index=False);sparse.save_npz(npz,mat);(O/'logs'/(name+'_chromosomes.json')).write_text(json.dumps({k:int(v) for k,v in pos.items()}),encoding='utf-8')
  if cfg['name']=='RAW_AUTO_C2000_W100':
   check=d.set_index('cell_id').loc[hobs.index.intersection(d.cell_id)];err=float(np.max(abs(check.cnv_score-hobs.loc[check.index,'cnv_score'])));assert err<2e-6;assert set(d.cell_id[d.selected])==set(hobs.index[hobs.patient_id.astype(str).eq(pid)]);print('HISTORICAL REPRODUCTION',pid,err,flush=True)
  hs=d.historical_selected;ns=d.selected;pr=d.R1_5_selected;union=(hs|ns).sum();summary.append(dict(condition=cfg['name'],patient_id=pid,reference_N=int(normal.sum()),tumor_candidates_N=int((~normal).sum()),historical_N=int(hs.sum()),selected_N=int(ns.sum()),intersection_historical=int((hs&ns).sum()),jaccard_historical=float((hs&ns).sum()/union) if union else np.nan,intersection_R1_5=int((pr&ns).sum()),threshold=float(d.threshold_normal99.iloc[0]),reference_score_median=float(d.loc[normal,'cnv_score'].median()),tumor_score_median=float(d.loc[~normal,'cnv_score'].median()),candidate_identity='INFERRED_NOT_GENOMICALLY_VALIDATED'))
  for label,sel in [('reference',normal),('tumor',~normal),('historical',hs.to_numpy()),('selected',ns.to_numpy())]:
   if sel.sum()>2:
    rho=spearmanr(d.loc[sel,'total_counts'],d.loc[sel,'cnv_score']).statistic;rel.append(dict(condition=cfg['name'],patient_id=pid,group=label,cell_N=int(sel.sum()),rho_library_size_score=rho,scope='descriptive cell-level technical association; NOT patient-replicated test'))
  keys=list(pos);starts=list(pos.values());ends=starts[1:]+[mat.shape[1]]
  for chrom,st,en in zip(keys,starts,ends):
   vals=np.asarray(mat[:,st:en].mean(1)).ravel()
   for label,sel in [('reference',normal),('tumor',~normal),('historical',hs.to_numpy()),('selected',ns.to_numpy())]:
    chromrows.append(dict(condition=cfg['name'],patient_id=pid,chromosome=chrom,group=label,cell_N=int(sel.sum()),mean_signed_window_signal=float(vals[sel].mean()) if sel.any() else np.nan,median_signed_window_signal=float(np.median(vals[sel])) if sel.any() else np.nan,positive_fraction=float((vals[sel]>0).mean()) if sel.any() else np.nan,scope='expression-derived chromosome windows, not DNA calls'))
  allcells.extend(d.to_dict('records'))
pd.DataFrame(summary).to_csv(O/'GSE196756_CNV_PATIENT_COMPARISON.tsv',sep='\t',index=False);pd.DataFrame(allcells).to_csv(O/'GSE196756_CNV_CELL_COMPARISON.tsv',sep='\t',index=False);pd.DataFrame(chromrows).to_csv(O/'CNV_CHROMOSOME_PATTERN_SUMMARY.tsv',sep='\t',index=False);pd.DataFrame(rel).to_csv(O/'CNV_LIBRARY_SIZE_ASSOCIATION.tsv',sep='\t',index=False);a.obs.to_csv(O/'CNV_CANDIDATE_METADATA.tsv',sep='\t')
print(pd.DataFrame(summary).to_string(index=False),flush=True)
