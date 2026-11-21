
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from common import *
from scipy.integrate import simpson
import re,xml.etree.ElementTree as ET
M='05_pdo_metadata';start(M)
source=record('08_PDO_functional_validation/raw/paper/supplementary/42003_2025_7869_MOESM3_ESM.xlsx',M)
x=workbook_cells(source);dose=x['Fig. 6a'];reported={row[0]:float(row[1]) for row in x['Fig. 6e'] if isinstance(row[0],str) and re.fullmatch(r'ESCCO\d+',row[0])}
concs=np.array([row[0] for row in dose[4:]],float);assert np.all(np.diff(concs)>0) and np.array_equal(concs,[0,.1,.4,1.6,6.4,25,100,400]);rows=[];points=[]
for j,pid in enumerate(dose[2]):
 if not isinstance(pid,str) or not re.fullmatch(r'ESCCO\d+',pid):continue
 block=np.array([row[j:j+3] for row in dose[4:]],float);means=block.mean(1);assert np.isfinite(block).all();auc=reported[pid]
 methods={'linear_concentration_trapezoid_percent':np.trapezoid(means,concs),'linear_concentration_simpson_percent':simpson(means,x=concs),'log10_concentration_trapezoid_excluding_vehicle':np.trapezoid(means[1:],np.log10(concs[1:])),'linear_trapezoid_fraction_viability':np.trapezoid(means/100,concs),'linear_trapezoid_vehicle_normalized_percent':np.trapezoid(means/means[0]*100,concs),'linear_trapezoid_exclude_highest_dose':np.trapezoid(means[:-1],concs[:-1]),'linear_trapezoid_zero_clip_negative_wells':np.trapezoid(np.maximum(block,0).mean(1),concs)}
 for method,v in methods.items():rows.append(dict(PDO_ID=pid,method=method,n_doses=len(concs) if 'excluding_vehicle' not in method else len(concs)-1,n_wells_per_dose=3,author_AUC=auc,reconstructed=v,absolute_difference=abs(v-auc),relative_difference=(v-auc)/auc,method_interpretation='diagnostic alternative; not evidence of author original integration method'))
 for i,c in enumerate(concs):
  for well in range(3):points.append(dict(PDO_ID=pid,cisplatin_uM=c,technical_well=well+1,relative_viability_percent=block[i,well],mean_three_wells=means[i]))
df=pd.DataFrame(rows);save(df,M,'all_AUC_definition_diagnostics.tsv');save(pd.DataFrame(points),M,'original_dose_response_well_values.tsv')
old=read(R1/'05_pdo/all_10_PDO_AUC_plot_data.tsv',M);assert all(reported[r.PDO_ID]==r.CDDP_AUC for r in old.itertuples())
clinical=x['Supplementary Table 1'];ct=pd.DataFrame(clinical[4:],columns=['patient_ID','sex','age','type','pTNM','pStage','therapy','drug','response','normal_organoid','tumor_organoid','chemo_resistance','NRF2_mutation','unused']);ct['patient_ID']=pd.to_numeric(ct.patient_ID,errors='coerce');ct=ct[ct.patient_ID.notna()].copy();ct.patient_ID=ct.patient_ID.astype(int);assert ct.patient_ID.is_unique
joined=old[['PDO_ID','patient_ID','GSM','CDDP_AUC']].merge(ct,on='patient_ID',how='left',validate='one_to_one');assert len(joined)==10 and joined.sex.notna().all() and joined.PDO_ID.str.removeprefix('ESCCO').astype(int).eq(joined.patient_ID).all();joined['drug_response_batch']='NOT_AVAILABLE';joined['matched_expression_assay_batch']='NOT_AVAILABLE';joined['patient_independence_evidence']='distinct clinical patient IDs and one analyzed tumor PDO per ID; no extra wells counted';save(joined,M,'10_PDO_source_and_independence_audit.tsv')
paper=record('08_PDO_functional_validation/raw/paper/PMC11961618.xml',M);root=ET.fromstring(paper.read_bytes());paras=[]
for el in root.iter():
 if el.tag in ['p','title']:
  t=''.join(el.itertext())
  if any(k in t.lower() for k in ['batch','passag','celltiter','matrigel','dose-response','area under','interferon','nrf2']):paras.append(t)
(O/M/'source_method_evidence.txt').write_text('\n\n'.join(paras),encoding='utf-8')
f=df[(df.PDO_ID=='ESCCO15')&(df.method=='linear_concentration_trapezoid_percent')].iloc[0]
finish(M,'PARTIAL_AUC_METHOD_AND_BATCH_UNRESOLVED',dict(PatientIDs_verified=10,well_values=len(points),AUC_methods=len(methods),ESCCO15_cause='UNRESOLVED_AUTHOR_IMPLEMENTATION',batch='NOT_AVAILABLE'))
