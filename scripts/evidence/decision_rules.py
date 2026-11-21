"""Frozen, identity-independent candidate decision functions; no file I/O."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from copy import deepcopy
import math

def known(x):return x is not None and not (isinstance(x,float) and math.isnan(x))
def discovery(r,c):
 d=r.get('discovery')
 if not d or not all(known(d.get(k)) for k in ['q','rbh','metric_count','no_technical']):return None
 return d['q']<c['match_q'] and d['rbh'] and d['metric_count']>=c['match_metrics'] and d['no_technical']

def concentration(r,c,mode):
 d=r.get('discovery')
 if not d:return None
 keys={'RAW':['raw_max'],'EQUAL':['equal_max'],'BOTH':['raw_max','equal_max'],'BALANCED':['balanced_max']}[mode]
 if not all(known(d.get(k)) for k in keys):return None
 return max(d[k] for k in keys)>=c['dominance']

def old_cohort(e,c):
 o=e.get('old')
 if not o:return None
 keys=['coverage_percent','patients_evaluable','median_patient_z','positive_patient_fraction','major_patient_instability','median_rank_structure_rho','positive_patient_structure_fraction','median_gene_gene_rho','top50_top100_patient_rho']
 if not all(known(o.get(k)) for k in keys):return None
 if o['patients_evaluable']<c['old_min_n'] or o['coverage_percent']<60:return None
 det=o['median_patient_z']>0 and o['positive_patient_fraction']>=c['old_positive'] and o['coverage_percent']>=70
 mixed=o['median_patient_z']>0 or o['positive_patient_fraction']>=.5
 coherent=o['median_rank_structure_rho']>=c['old_coherence_rho'] and o['positive_patient_structure_fraction']>.5 and o['median_gene_gene_rho']>0
 low=o['median_rank_structure_rho']<=0 and o['median_gene_gene_rho']<=0
 size=o['top50_top100_patient_rho']>=c['old_size_rho']
 if not det and not mixed and low:return 0
 if o['coverage_percent']>=80 and det and not o['major_patient_instability'] and coherent and size:return 3
 if o['coverage_percent']>=70 and not o['major_patient_instability'] and sum([det,coherent,size])>=2:return 2
 return 1

def old_overall(r,c):
 vals=[old_cohort(e,c) for e in r['external']]
 # Missing requested cohort cannot create a stronger grade.
 if any(v is None for v in vals):return None
 if not vals:return None
 if all(v==0 for v in vals):return 0
 if len(vals)==2 and min(vals)>=2 and max(vals)>=3:return 3
 if len(vals)==2 and min(vals)>=1 and max(vals)>=2:return 2
 if len(vals)==1 and vals[0]>=3:return 2
 return 1

def rna(r,c):
 states=[];why=[];pending=False
 for e in r['external']:
  # Cohort biological replication, not resampling draw count.
  if e['n_original']<c['rna_min_n']:
   why.append(e['cohort']+':SMALL_ORIGINAL_N');continue
  a=e.get('coherence')
  if not a or not all(known(a.get(k)) for k in ['n_patients_evaluable','n_patients_original','mean_patient_effect','CI_low','BH_FDR','positive_patient_fraction']):
   pending=True;why.append(e['cohort']+':MISSING_COHERENCE');continue
  if a['n_patients_evaluable']<c['rna_min_n'] or a['n_patients_evaluable']<c['rna_patient_coverage']*a['n_patients_original']:
   pending=True;why.append(e['cohort']+':INSUFFICIENT_EVALUABLE_PATIENTS');continue
  criteria={'q':a['BH_FDR']<c['evidence_q'],'effect':a['mean_patient_effect']>0,'CI':a['CI_low']>0,'positive_patients':a['positive_patient_fraction']>=c['rna_positive']}
  states.append(all(criteria.values()));why.append(e['cohort']+':'+('SUPPORTED' if all(criteria.values()) else 'NOT_ESTABLISHED('+','.join(k for k,v in criteria.items() if not v)+')'))
 if any(v is False for v in states):return False,';'.join(why),len(states)
 if pending or not states:return None,';'.join(why),len(states)
 return True,';'.join(why),len(states)

def protein(r,c):
 p=r.get('protein')
 if not p or not all(known(p.get(k)) for k in ['N_pairs','coverage','matched_null_FDR','CI_low','CI_high']):return None
 if p['N_pairs']<1:return None
 return p['coverage']>=c['protein_coverage'] and p['matched_null_FDR']<c['evidence_q'] and (p['CI_low']>0 or p['CI_high']<0)

def classify(record,model,config,mode=None):
 r=record;c=config['thresholds'];mode=mode or config['models'][model]['dominance_mode'];d=discovery(r,c);risk=concentration(r,c,mode);rn,rwhy,nc=rna(r,c);pr=protein(r,c);old=old_overall(r,c);ann=r.get('annotation_rank')
 out=dict(anonymous_id=r['anonymous_id'],size=r['size'],model=model,dominance_mode=mode,discovery_supported=d,concentration_caution=risk,old_external_grade=old,RNA_supported=rn,RNA_evaluable_cohorts=nc,RNA_detail=rwhy,protein_paired_abundance_supported=pr,category=None,reason=None,scope=config['models'][model]['target'])
 def done(cat,reason):out.update(category=cat,reason=reason);return out
 if d is None:return done('INDETERMINATE','Discovery input missing; no upgrade')
 if not d:return done('Exploratory','Frozen robust discovery criterion not met; not proof of biological absence')
 if risk is None:return done('INDETERMINATE','Patient-concentration evidence missing; no upgrade')
 if risk:return done('Exploratory','Concentration caution under '+mode+'; historical Tier retained separately')
 if model=='A':
  if ann is None or old is None:return done('INDETERMINATE','Historical annotation/external support unavailable')
  if ann<2 or old==0:return done('Exploratory','Historical interpretation/support gate not met')
  if old==3:return done('Primary','Historical STRONG RNA support with discovery and original concentration gates')
  return done('Secondary','Historical evaluable external evidence below STRONG')
 if model=='B':
  if rn is None:return done('INDETERMINATE','Adequate external patient coherence unresolved')
  if not rn:return done('Exploratory','Patient-level RNA coordination support not established; not biological absence')
  if pr is None:return done('INDETERMINATE','RNA supported; protein missing leaves Primary/Secondary unresolved')
  if pr:return done('Primary','RNA coordination plus paired tissue-protein abundance evidence; no preservation claim')
  return done('Secondary','RNA support without qualified paired-protein abundance support')
 if model=='C':
  if ann is None:return done('INDETERMINATE','Biological interpretation missing')
  if ann<2:return done('Exploratory','Interpretation remains insufficient for focal eligibility')
  if rn is None:return done('INDETERMINATE','Adequate patient-level RNA evidence unresolved')
  if rn:return done('Primary','FOCAL_ELIGIBLE only; no automatic representative selection or overall rank')
  return done('Secondary','COMPARATIVE_ROLE; RNA coordination not established')
 raise ValueError('Unknown model')

def ablate(record,dimension):
 r=deepcopy(record)
 if dimension=='discovery':r['discovery']=None
 elif dimension=='patient_balance':
  for key in ['raw_max','equal_max','balanced_max']:r['discovery'][key]=None
 elif dimension=='old_external':
  for e in r['external']:e['old']=None
 elif dimension=='coherence':
  for e in r['external']:e['coherence']=None
 elif dimension=='protein':r['protein']=None
 elif dimension=='annotation':r['annotation_rank']=None
 elif dimension in ['PDO','bulk_context','clinical','pathology']:r.get('auxiliary',{}).pop(dimension,None)
 else:raise ValueError(dimension)
 return r
