
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
import os,json,copy,importlib.util
R = INPUT_ROOT;O=OUTPUT_ROOT/'quality/grouping_sensitivity';os.chdir(R)
O.mkdir(parents=True,exist_ok=True)
for _folder in ['analysis','logs','tables','figures']:(O/_folder).mkdir(exist_ok=True)
import numpy as np,pandas as pd,anndata as ad

def rd(p,**kw):return pd.read_csv(resolve_data(p),sep='\t',**kw)
def save(d,n):pd.DataFrame(d).to_csv(O/n,sep='\t',index=False)
config=json.loads(data_path('PREDEFINED_SCENARIOS.json').read_text(encoding='utf-8'));ann=rd(R/'06_rmp_annotation/rmp_annotation_table.tsv');historical=rd(R/'FIG_R1_5_Malignant_Identity_QC_Validation_20261114/PATIENT_BALANCE_SENSITIVITY.tsv');checks=[]
def ck(n,v):checks.append(dict(test=n,passed=bool(v)));assert v,n
cohorts={}
for ds in ['GSE160269','GSE188900']:
 u=rd(R/f'04_cnmf_formal/{ds}/cell_program_usage_normalized.tsv',index_col=0);a=ad.read_h5ad(R/f'02_data_processed/{ds}_malignant_ready.h5ad',backed='r');obs=a.obs.loc[u.index].copy();a.file.close();ck('IDs_unique_'+ds,u.index.is_unique and len(u)==len(obs));ck('nonnegative_row_normalized_'+ds,(u.to_numpy()>=0).all() and np.allclose(u.sum(1),1,atol=2e-6));cohorts[ds]=(u,obs)
u160,m160=cohorts['GSE160269'];sum160=u160.groupby(m160.patient_id.astype(str)).sum();mean160=u160.groupby(m160.patient_id.astype(str)).mean();raw160=sum160/sum160.sum();eq160=mean160/mean160.sum();unch=[]
for row in ann.itertuples():
 for pid in raw160.index:unch.append(dict(RMP=row.rmp_id,patient_id=pid,RAW=raw160.loc[pid,row.GSE160269_program],EQUAL=eq160.loc[pid,row.GSE160269_program],n_cells=int(m160.patient_id.astype(str).eq(pid).sum())))
save(unch,'UNCHANGED_GSE160269_CONTRIBUTIONS.tsv')
sp=importlib.util.spec_from_file_location('edsc',REPO_ROOT/'scripts/evidence/classification.py');ed=importlib.util.module_from_spec(sp);sp.loader.exec_module(ed);records=json.loads(data_path('anonymous_records.json').read_text());key=rd(R/'E1_FINAL_RESOLUTION/inputs/IDENTITY_KEY.tsv').set_index('anonymous_id').RMP.to_dict();recordmap={(key[r['anonymous_id']],r['size']):r for r in records};cfg=ed.load_config();before={k:ed.classify_edsc(v) for k,v in recordmap.items()};matrix=[];units=[];tier=[];edrows=[];registry=[]
u,obs=cohorts['GSE188900'];alias=obs.patient_id.astype(str);originals=u.groupby(alias).sum();counts=alias.value_counts().sort_index();baseline_metrics={}
for sc in config['scenarios']:
 pair=sc['merged_aliases'];mapping={p:('MERGED_'+'_'.join(pair) if p in pair else p) for p in counts.index};labels=alias.map(mapping);sums=u.groupby(labels).sum();n=labels.value_counts().sort_index();means=sums.div(n,axis=0);raw=sums/sums.sum();equal=means/means.sum();ck('unitN_'+sc['scenario'],len(n)==sc['hypothetical_units']);ck('cellN_'+sc['scenario'],n.sum()==len(u));ck('contribution_sums_'+sc['scenario'],np.allclose(raw.sum(),1) and np.allclose(equal.sum(),1));ck('P6_not_split_'+sc['scenario'],labels[alias=='P6'].nunique()==1)
 registry.append(dict(scenario=sc['scenario'],merged_aliases='+'.join(pair) if pair else 'NONE',included=True,hypothetical_units=len(n),exclusion_evidence='NONE; no evidence-based exclusion',status=sc['status'],true_donor_mapping='DONOR_MAPPING_UNRESOLVED'))
 for a in ann.itertuples():
  rid=a.rmp_id;prog=a.GSE188900_program;rs=raw[prog];es=equal[prog];rmax=float(rs.max());emax=float(es.max());r160=float(raw160[a.GSE160269_program].max());e160=float(eq160[a.GSE160269_program].max());rr=max(rmax,r160);ee=max(emax,e160);risk=rr>=.8 or ee>=.8;tt='TIER_2_DOMINANCE_CAUTIOUS_RECURRENT_PROGRAM' if rr>=.8 else 'TIER_1_CORE_RECURRENT_PROGRAM'
  ck('RAW_cannot_decrease_'+sc['scenario']+rid,rmax+1e-12>=float((originals[prog]/originals[prog].sum()).max()))
  if not pair:
   h=historical[(historical.dataset=='GSE188900')&(historical.RMP==rid)].iloc[0];ck('baseline_RAW_EQUAL_'+rid,abs(rmax-h.RAW)<2e-6 and abs(emax-h.EQUAL)<2e-6);ck('historical_Tier_'+rid,tt==a.tier)
  for pid in rs.index:units.append(dict(scenario=sc['scenario'],RMP=rid,program_id=prog,hypothetical_unit=pid,member_aliases='+'.join(p for p,g in mapping.items() if g==pid),n_cells=int(n[pid]),normalized_usage_sum=sums.loc[pid,prog],mean_normalized_usage=means.loc[pid,prog],RAW=rs[pid],EQUAL=es[pid],interpretation='Actual cell usage, hypothetical grouping; not true donor identity'))
  metric=dict(scenario=sc['scenario'],RMP=rid,hypothetical_units=len(n),merged_aliases='+'.join(pair) if pair else 'NONE',RAW_188900=rmax,EQUAL_188900=emax,RAW_max_unit=rs.idxmax(),EQUAL_max_unit=es.idxmax(),RAW_160269=r160,EQUAL_160269=e160,RAW_two_cohort=rr,EQUAL_two_cohort=ee,RAW_caution=rr>=.8,EQUAL_caution=ee>=.8,BOTH_caution=risk,conditional_Tier=tt,historical_Tier=a.tier,Tier_changed=tt!=a.tier,status='HISTORICAL_RESULT' if not pair else 'HYPOTHETICAL_SCENARIO',true_donor_mapping='DONOR_MAPPING_UNRESOLVED')
  matrix.append(metric);tier.append({k:metric[k] for k in ['scenario','RMP','merged_aliases','RAW_two_cohort','RAW_caution','historical_Tier','conditional_Tier','Tier_changed','true_donor_mapping']})
  for size in [50,100]:
   old=recordmap[(rid,size)];r=copy.deepcopy(old);r['discovery']['raw_max']=rr;r['discovery']['equal_max']=ee
   ck('fixed_Q_P_other_inputs_'+sc['scenario']+rid+str(size),all(r[k]==old[k] for k in old if k!='discovery') and all(r['discovery'][k]==old['discovery'][k] for k in old['discovery'] if k not in ['raw_max','equal_max']))
   result=ed.classify_edsc(r,cfg);orig=before[(rid,size)]
   if not pair:ck('baseline_EDSC_'+rid+str(size),result['EDSC_group']==orig['EDSC_group'] and abs(old['discovery']['raw_max']-rr)<2e-6 and abs(old['discovery']['equal_max']-ee)<2e-6)
   edrows.append(dict(scenario=sc['scenario'],RMP=rid,size=size,historical_EDSC=orig['EDSC_group'],scenario_EDSC=result['EDSC_group'],category=result['category'],changed=result['EDSC_group']!=orig['EDSC_group'],concentration_caution=result['concentration_caution'],RNA_supported=result['RNA_supported'],protein_supported=result['protein_paired_abundance_supported'],reason=result['reason'],RAW_two_cohort=rr,EQUAL_two_cohort=ee,status='HISTORICAL_RESULT' if not pair else 'CONDITIONAL_SCENARIO_RESULT',other_inputs='FROZEN_REUSED_NOT_NEWLY_VALIDATED',donor_status='DONOR_MAPPING_UNRESOLVED'))
save(matrix,'ALL_16_SCENARIOS_PROGRAM_METRICS.tsv');save(units,'SCENARIO_UNIT_CONTRIBUTIONS.tsv');save(tier,'DISCOVERY_TIER_SCENARIOS.tsv');save(edrows,'EDSC_SCENARIOS.tsv');save(registry,'SCENARIO_REGISTRY.tsv');save(checks,'logs/ANALYSIS_CHECKS.tsv')
m=pd.DataFrame(matrix);e=pd.DataFrame(edrows);summ=[]
for rid,g in m.groupby('RMP'):
 z=g[g.scenario!='S00_ORIGINAL6'];er=e[e.RMP==rid];summ.append(dict(RMP=rid,RAW_min=g.RAW_188900.min(),RAW_max=g.RAW_188900.max(),EQUAL_min=g.EQUAL_188900.min(),EQUAL_max=g.EQUAL_188900.max(),RAW_caution_N16=int(g.RAW_caution.sum()),EQUAL_caution_N16=int(g.EQUAL_caution.sum()),BOTH_caution_N16=int(g.BOTH_caution.sum()),RAW_caution_N15=int(z.RAW_caution.sum()),EQUAL_caution_N15=int(z.EQUAL_caution.sum()),Tier_changed_N15=int(z.Tier_changed.sum()),Top50_EDSC_changed_N15=int(er[(er['size']==50)&(er.scenario!='S00_ORIGINAL6')].changed.sum()),Top100_EDSC_changed_N15=int(er[(er['size']==100)&(er.scenario!='S00_ORIGINAL6')].changed.sum()),scenario_status='SCENARIO_ROBUST' if er.scenario_EDSC.nunique()==1 and g.conditional_Tier.nunique()==1 else 'SCENARIO_DEPENDENT',donor_status='DONOR_UNRESOLVED',N_scenarios=16,interpretation='Ranges/counts over enumerated hypotheses, not probabilities or confidence intervals'))
save(summ,'RAW_EQUAL_RANGE_AND_STABILITY.tsv');print(pd.DataFrame(summ).to_string(index=False));print('Completed',len(matrix),'program-scenario metrics and',len(edrows),'EDSCrecords')
