
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
import json,os,xml.etree.ElementTree as ET
R = INPUT_ROOT;O=OUTPUT_ROOT/'quality/cnv';P=R/'FIG_R1_5_Malignant_Identity_QC_Validation_20261114'
O.mkdir(parents=True,exist_ok=True)
for _folder in ['analysis','logs','tables','figures']:(O/_folder).mkdir(exist_ok=True)
import pandas as pd,numpy as np
from scipy import sparse
from scipy.stats import spearmanr
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
cells=pd.read_csv(resolve_data(O/'GSE196756_CNV_CELL_COMPARISON.tsv'),sep='\t');main=cells[cells.condition=='LOG_ALL_FULL_W100'].copy();hist=cells[cells.condition=='RAW_AUTO_C2000_W100'].set_index('cell_id');main['raw_CNV_score']=main.cell_id.map(hist.cnv_score);main['raw_threshold']=main.cell_id.map(hist.threshold_normal99);prev=pd.read_csv(resolve_data(P/'GSE196756_NORMALIZED_CNV_CELL_AUDIT.tsv'),sep='\t').set_index('cell_id');main['R1_5_score']=main.cell_id.map(prev.cnv_score_log_normalized)
main['identity_state']=np.select([main.reference,main.selected&main.historical_selected,main.selected&~main.historical_selected,~main.selected&main.historical_selected],['REFERENCE_TISSUE_EPITHELIAL','CONSISTENT_CNV_CANDIDATE','REPAIRED_ONLY_CNV_CANDIDATE','HISTORICAL_ONLY_UNCERTAIN'],default='UNSELECTED_IDENTITY_UNCERTAIN');main.to_csv(O/'REPAIRED_CNV_CELL_IDENTITY_STATES.tsv',sep='\t',index=False)
reason=[];quant=[];selectedids=[]
fig,axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
for ax,pid in zip(axes,['P1','P2','P4']):
 g=main[main.patient_id==pid].copy();mat=sparse.load_npz(O/'logs'/('LOG_ALL_FULL_W100_'+pid+'.npz'));pos=json.loads((O/'logs'/('LOG_ALL_FULL_W100_'+pid+'_chromosomes.json')).read_text());idx=[]
 order=['REFERENCE_TISSUE_EPITHELIAL','HISTORICAL_ONLY_UNCERTAIN','CONSISTENT_CNV_CANDIDATE','REPAIRED_ONLY_CNV_CANDIDATE','UNSELECTED_IDENTITY_UNCERTAIN'];cuts=[]
 for label in order:
  loc=np.flatnonzero(g.identity_state.eq(label));loc=loc[np.argsort(g.cnv_score.to_numpy()[loc])];loc=loc[np.linspace(0,len(loc)-1,min(100,len(loc)),dtype=int)] if len(loc) else loc
  idx.extend(loc);cuts.append(len(idx));selectedids.extend(dict(patient_id=pid,plot_row=len(idx)-len(loc)+j,cell_id=g.iloc[i].cell_id,identity_state=label) for j,i in enumerate(loc))
 v=mat[idx].toarray();im=ax.imshow(v,aspect='auto',cmap='coolwarm',vmin=-.25,vmax=.25,interpolation='nearest');keys=list(pos);starts=list(pos.values());ends=starts[1:]+[mat.shape[1]];ax.set_xticks([(a+b)/2 for a,b in zip(starts,ends)],[k.replace('chr','') for k in keys],fontsize=8)
 labels=['Reference','Old only','Old + repaired','Repaired only','Other uncertain'];starts_y=[0]+cuts[:-1];present=[(st,en,lab) for st,en,lab in zip(starts_y,cuts,labels) if en>st]
 large=[(st,en,lab) for st,en,lab in present if en-st>=10];small=[f'{lab}: {en-st} plotted rows' for st,en,lab in present if en-st<10]
 ax.set_yticks([(st+en-1)/2 for st,en,lab in large],[lab for st,en,lab in large],fontsize=7)
 if small:ax.text(.995,.98,'; '.join(small),ha='right',va='top',transform=ax.transAxes,fontsize=7,bbox=dict(facecolor='white',alpha=.9,edgecolor='gray'))
 for cut in cuts[:-1]:ax.axhline(cut-.5,color='black',lw=.5)
 for st in starts:ax.axvline(st-.5,color='gray',lw=.3)
 ax.set_ylabel('Actual cells, grouped');ax.set_title(pid+': repaired log-normalized CNV windows (display sample <=100 per identity group)');ax.set_xlabel('Chromosome')
 groups=g[~g.reference];rho=spearmanr(groups.raw_CNV_score,groups.cnv_score).statistic
 reason.append(dict(patient_id=pid,raw_vs_repaired_score_rho_tumor=rho,historical_cell_N=int(groups.historical_selected.sum()),repaired_cell_N=int(groups.selected.sum()),overlap=int((groups.selected&groups.historical_selected).sum()),raw_selected_library_median=float(groups.loc[groups.historical_selected,'total_counts'].median()),new_selected_library_median=float(groups.loc[groups.selected,'total_counts'].median()),tumor_library_median=float(groups.total_counts.median()),reference99_threshold=float(g.threshold_normal99.iloc[0]),normal_reference_N=int(g.reference.sum()),interpretation='technical score reordering and patient-relative threshold, NOT proof of true malignant/normal status'))
 ref=g.loc[g.reference,'cnv_score'].to_numpy();rng=np.random.default_rng(2026111406);draw=np.quantile(rng.choice(ref,size=(2000,len(ref)),replace=True),.99,axis=1);nsel=np.array([(groups.cnv_score>t).sum() for t in draw]);quant.append(dict(patient_id=pid,conditional_reference_quantile025=np.quantile(draw,.025),conditional_reference_quantile975=np.quantile(draw,.975),selected_N025=np.quantile(nsel,.025),selected_N975=np.quantile(nsel,.975),scope='conditional resampling of saved reference scores; no refitting of reference mean/CNV; not biological patient CI',seed=2026111406))
fig.colorbar(im,ax=axes,label='Centered expression-derived CNV signal',shrink=.5);fig.savefig(O/'figures/CNV_CHROMOSOME_HEATMAP.png',dpi=180);plt.close(fig);pd.DataFrame(selectedids).to_csv(O/'figures/CNV_HEATMAP_ROW_IDS.tsv',sep='\t',index=False);pd.DataFrame(reason).to_csv(O/'CNV_DIAGNOSTIC_EXPLANATION.tsv',sep='\t',index=False);pd.DataFrame(quant).to_csv(O/'REFERENCE_THRESHOLD_CONDITIONAL_SENSITIVITY.tsv',sep='\t',index=False)
fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,pid in zip(axs,['P1','P2','P4']):
 g=main[(main.patient_id==pid)&~main.reference];colors={'HISTORICAL_ONLY_UNCERTAIN':'orange','CONSISTENT_CNV_CANDIDATE':'purple','REPAIRED_ONLY_CNV_CANDIDATE':'red','UNSELECTED_IDENTITY_UNCERTAIN':'gray'}
 for state,gg in g.groupby('identity_state'):ax.scatter(gg.raw_CNV_score,gg.cnv_score,s=6,alpha=.5,label=state,color=colors.get(state,'black'))
 ax.axhline(g.threshold_normal99.iloc[0],ls='--',color='red');ax.axvline(g.raw_threshold.iloc[0],ls='--',color='orange');ax.set_title(pid);ax.set_xlabel('Historical raw-count score');ax.set_ylabel('Repaired log-normalized score')
axs[-1].legend(fontsize=5);fig.savefig(O/'figures/CNV_RAW_VS_REPAIRED_PATIENTS.png',dpi=180);plt.close(fig)
s=pd.read_csv(resolve_data(O/'GSE196756_CNV_PATIENT_COMPARISON.tsv'),sep='\t');pivot=s.pivot(index='condition',columns='patient_id',values='selected_N');fig,ax=plt.subplots(figsize=(11,4),layout='constrained');pivot.plot.bar(ax=ax);ax.set_ylabel('CNV-supported tumor candidates');ax.tick_params(axis='x',rotation=20,labelsize=7);fig.savefig(O/'figures/CNV_PREDEFINED_PARAMETER_COUNTS.png',dpi=180);plt.close(fig)
for name in ['PMC9008022','PMC9982558']:
 root=ET.parse(INPUT_ROOT/'references/quality'/(name+'.xml')).getroot();texts=[' '.join(e.itertext()) for e in root.findall('.//p') if any(w in ' '.join(e.itertext()).lower() for w in ['infercnv','copy number variation','five patients','doublet'])];(INPUT_ROOT/'references/quality'/(name+'_relevant_paragraphs.txt')).write_text('\n\n'.join(texts),encoding='utf-8')
print(pd.DataFrame(reason).to_string(index=False));print(pd.DataFrame(quant).to_string(index=False))
