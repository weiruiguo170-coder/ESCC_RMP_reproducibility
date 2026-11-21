
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from common import *
import itertools,math,ast
from scipy.stats import kendalltau
M='04_ifn_nrf2_pdo';start(M)
manifest=read('06_rmp_annotation/gene_sets/gene_set_resource_manifest.tsv',M);save(manifest,M,'gene_set_resource_provenance_REUSED.tsv')
def gmt(path):
 p=record(path,M);out={}
 for line in p.read_text(encoding='utf-8').splitlines():
  t=line.split('\t');out[t[0]]=dict(description=t[1],genes=list(dict.fromkeys(x.strip().upper() for x in t[2:] if x.strip())))
 return out
react=gmt('06_rmp_annotation/gene_sets/Reactome_Pathways_2024.gmt');hall=gmt('06_rmp_annotation/gene_sets/MSigDB_Hallmark_2020.gmt')
logging.info('IFN Hallmark names %s',[x for x in hall if 'INTERFERON' in x.upper()])
def find(name,db):
 hits=[k for k in db if k.upper().replace('_',' ')==name.upper().replace('_',' ')];assert len(hits)==1,(name,hits);return hits[0]
sets={};prov=[]
for key,name in [('REACTOME_IFN_ALPHA_BETA','Interferon Alpha Beta Signaling'),('REACTOME_IFN_GAMMA','Interferon Gamma Signaling'),('REACTOME_NFE2L2_NUCLEAR','Nuclear Events Mediated by NFE2L2')]:
 n=find(name,react);sets[key]=react[n]['genes'];prov.append(dict(score=key,library='Reactome_Pathways_2024',formal_set_name=n,source_definition=react[n]['description'],role='primary comparator; independently curated pathway, not proven original PDO author score'))
for key,name in [('HALLMARK_IFN_ALPHA','Interferon Alpha Response'),('HALLMARK_IFN_GAMMA','Interferon Gamma Response')]:
 n=find(name,hall);sets[key]=hall[n]['genes'];prov.append(dict(score=key,library='MSigDB_Hallmark_2020',formal_set_name=n,source_definition=hall[n]['description'],role='sensitivity; response rather than Reactome signaling'))
src=REPO_ROOT/'scripts/analysis/pdo/scoring.py';tree=ast.parse(src.read_text(encoding='utf-8'))
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NRF2_TARGETS' for t in node.targets):sets['LEGACY_NRF2_SELECTED10']=ast.literal_eval(node.value)
prov.append(dict(score='LEGACY_NRF2_SELECTED10',library='retained original PDO analysis script',formal_set_name='NRF2_TARGETS',source_definition='10 selected reported targets; source-study formal scoring set not established',role='sensitivity only; not synonymous with all NRF2 pathways'))
save(pd.DataFrame(prov),M,'gene_set_provenance.tsv');save(pd.DataFrame([dict(score=k,rank=i+1,gene=g) for k,v in sets.items() for i,g in enumerate(v)]),M,'all_control_gene_membership.tsv')
expr=read('08_PDO_functional_validation/processed/GSE269447_TPM_canonicalized.tsv',M,index_col=0);expr=expr[[c for c in expr if c.endswith('_Tumor')]];assert len(expr.columns)==15;vals=expr.to_numpy(float).T;ranks=rankdata(vals,axis=1);detect=(vals>0).any(0);lookup={g:i for i,g in enumerate(expr.index)};scores=pd.DataFrame(index=[x.removesuffix('_Tumor') for x in expr.columns]);coverage=[]
for key,genes in sets.items():
 idx=np.array([lookup[g] for g in genes if g in lookup and detect[lookup[g]]]);scores[key]=score_ranks(ranks,idx)
 for pid,v in zip(scores.index,vals):coverage.append(dict(PDO_ID=pid,score=key,set_size=len(genes),effective_genes=len(idx),coverage_fraction=len(idx)/len(genes),nonzero_genes_this_PDO=int((v[idx]>0).sum()),missing_or_never_detected=';'.join(g for g in genes if g not in lookup or not detect[lookup[g]])))
old=read(R1/'05_pdo/all_10_PDO_AUC_plot_data.tsv',M).set_index('PDO_ID');data=old[['patient_ID','GSM','CDDP_AUC','C07_Top50','C07_Top100']].join(scores,validate='one_to_one');assert len(data)==10 and not data.isna().any().any();save(data.reset_index(),M,'same_10_PDO_control_scores_and_AUC.tsv');save(pd.DataFrame(coverage),M,'per_PDO_control_coverage.tsv')
keys=list(sets);xr=np.column_stack([rankdata(data[k]) for k in keys]);xr-=xr.mean(0);xr/=np.linalg.norm(xr,axis=0);yr=rankdata(data.CDDP_AUC);ym=yr.mean();yn=np.linalg.norm(yr-ym);observed=(yr-ym)@xr/yn;counts=np.zeros(len(keys),int);it=itertools.permutations(yr);total=0
while True:
 block=list(itertools.islice(it,50000))
 if not block:break
 a=np.asarray(block);dist=(a-ym)@xr/yn;counts+=(abs(dist)>=abs(observed)-1e-12).sum(0);total+=len(a)
assert total==3628800
res=[]
for k,r,p in zip(keys,observed,counts/total):res.append(dict(score=k,n_PDO=10,rho=r,P_exact=p,P_asymptotic=spearmanr(data[k],data.CDDP_AUC).pvalue,source='NEW control exact calibration'))
r1=read(R1/'05_pdo/seven_program_spearman_exact_and_FDR.tsv',M)
for sig in ['Top50','Top100']:
 row=r1[(r1.RMP=='RMP_C07')&(r1.signature==sig)].iloc[0];res.append(dict(score='C07_'+sig,n_PDO=10,rho=row.rho,P_exact=row.P_exact_two_sided,P_asymptotic=row.P_scipy_asymptotic,Round1_FDR7_exact=row.FDR_seven_exact,source='REUSED Round1 exact P; not re-enumerated'))
res=pd.DataFrame(res);primary=['C07_Top50','REACTOME_IFN_ALPHA_BETA','REACTOME_IFN_GAMMA','REACTOME_NFE2L2_NUCLEAR'];secondary=['C07_Top100','HALLMARK_IFN_ALPHA','HALLMARK_IFN_GAMMA','LEGACY_NRF2_SELECTED10']
for family,ks in [('REVISION_PRIMARY_FOUR',primary),('SIGNATURE_AND_DEFINITION_SENSITIVITY_FOUR',secondary)]:
 ix=res.index[res.score.isin(ks)];assert len(ix)==4;res.loc[ix,'test_family']=family;res.loc[ix,'BH_FDR_exact']=bh(res.loc[ix,'P_exact'])
save(res,M,'control_AUC_exact_correlations_and_FDR.tsv')
allkeys=['C07_Top50','C07_Top100']+keys;corr=[];over=[];loo=[];sigs=signatures(M)
for a in allkeys:
 for b in allkeys:
  rho,p=spearmanr(data[a],data[b]);tau,tp=kendalltau(data[a],data[b]);corr.append(dict(score_A=a,score_B=b,n_PDO=10,spearman_rho=rho,kendall_tau=tau,identical_rank_order=bool(np.array_equal(rankdata(data[a]),rankdata(data[b]))),mean_absolute_rank_displacement=np.mean(abs(rankdata(data[a])-rankdata(data[b])))))
 for drop in data.index:
  sub=data.drop(index=drop);rho,p=spearmanr(sub[a],sub.CDDP_AUC);loo.append(dict(score=a,omitted_PDO=drop,n_PDO=9,rho=rho,P_asymptotic_descriptive=p))
for sig in ['TOP50','TOP100']:
 a=set(sigs['RMP_C07_'+sig])
 for key,v in sets.items():
  b=set(v);n=len(a&b);over.append(dict(C07_signature=sig,control=key,C07_n=len(a),control_n=len(b),intersection=n,Jaccard=n/len(a|b),overlap_coefficient=n/min(len(a),len(b)),shared_genes=';'.join(sorted(a&b))))
save(pd.DataFrame(corr),M,'score_spearman_and_rank_agreement.tsv');save(pd.DataFrame(over),M,'C07_control_gene_overlap.tsv');save(pd.DataFrame(loo),M,'all_control_LOO_sensitivity.tsv')
mat=pd.DataFrame(corr).pivot(index='score_A',columns='score_B',values='spearman_rho');heat(mat,O/M/'control_score_correlations.png','Same 10 PDOs; score Spearman',vmin=-1,vmax=1,cmap='coolwarm')
fig,axs=plt.subplots(2,4,figsize=(17,8),layout='constrained')
for ax,k in zip(axs.flat,allkeys):
 ax.scatter(data[k],data.CDDP_AUC);ax.margins(.15)
 for pid,row in data.iterrows():ax.annotate(pid.replace('ESCCO',''),(row[k],row.CDDP_AUC),xytext=(4,4),textcoords='offset points',fontsize=7)
 ax.set_title(k,fontsize=10);ax.set_xlabel('Frozen rank score');ax.set_ylabel('Author CDDP AUC')
fig.savefig(O/M/'same_PDO_AUC_scatter_all_controls.png',dpi=170);plt.close(fig)
