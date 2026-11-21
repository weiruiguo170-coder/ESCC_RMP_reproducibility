
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
from itertools import combinations
import math
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from statsmodels.stats.multitest import multipletests

here=Path(__file__).resolve().parent
out=here.parent
tables=out/'data'
root=out.parent
checks=[]
def read(name):
    return pd.read_csv(resolve_data(tables/name),sep='\t')
def equal(title,actual,expected,tol=0):
    if isinstance(actual,(float,int,np.number)):
        assert math.isclose(actual,expected,rel_tol=0,abs_tol=tol),(title,actual,expected,tol)
    else:
        assert actual==expected,(title,actual,expected)
    checks.append((title,str(actual),str(expected),tol))

donor=read('S1_bounded_donor_sensitivity.tsv').set_index('RMP')
for prog,raw,equal_weight,either in [('01',0,0,0),('02',2,0,2),('03',16,0,16),('04',1,13,14),('05',1,3,4),('06',16,0,16),('07',1,0,1)]:
    x=donor.loc['RMP_C'+prog]
    equal('Donor scenarios C'+prog,(x.RAW_caution_N16,x.EQUAL_caution_N16,x.BOTH_caution_N16),(raw,equal_weight,either))
pairs=read('S3_all_80_pairs.tsv')
equal('80 matches at original unit',len(pairs),80)
equal('Exact nominal floor count',int(np.isclose(pairs.permutation_p,1/10001).sum()),71)
for q in ('S3_all_80_pairs.tsv','S3_matching_margins.tsv','S4_Q_summary.tsv','S4_original_expression_summary.tsv','S6_bulk_context.tsv','S6_bulk_patient_scores.tsv','S6_ordered_state_models.tsv','S6_paired_state_contrasts.tsv','S6_patient_state_scores.tsv','S7_Liu_gene_effects.tsv','S7_paired_proteomic_scores.tsv','S7_proteomic_contrasts.tsv','S8_PDO_exact_results.tsv','S8_PDO_scores_endpoints.tsv'):
    from hashlib import sha256
    expected_hash=json.loads(data_path('figure_numeric_hashes.json').read_text())[q]
    equal('Frozen figure input hash '+q,sha256(resolve_data(tables/q).read_bytes()).hexdigest(),expected_hash)
ctx=read('S6_bulk_context.tsv')
for sz,g in ctx.groupby('signature_size'):
    equal('Bulk precision '+sz,np.max(abs(multipletests(g.nominal_p,method='fdr_bh')[1]-g.BH_FDR)),0,1e-11)
bulk=read('S6_bulk_patient_scores.tsv')
equal('Bulk independent patient count',bulk.patient_id.nunique(),179)
equal('Bulk C01 median paired difference',bulk.RMP_C01_DELTA.median(),.187,.001)
path=read('S6_patient_state_scores.tsv').query("signature_type=='TOP50'")
for prog,g in path.groupby('rmp_id'):
    counts=g.groupby('ordered_state').size()
    equal(prog+' pathology state counts',tuple(counts),(9,11,14))
    equal(prog+' epithelial cell tally',int(g.retained_cells.sum()),3240)
mods=read('S6_ordered_state_models.tsv')
for (size,fam),g in mods.groupby(['signature','family']):
    equal('Ordered-state BH '+size+' '+fam,np.max(abs(multipletests(g.permutation_P,method='fdr_bh')[1]-g.BH_FDR)),0,1e-13)
for size,g in mods.groupby('signature'):
    equal('Ordered-state BH across seven '+size,np.max(abs(multipletests(g.permutation_P,method='fdr_bh')[1]-g.BH_FDR_all_seven_sensitivity)),0,1e-13)
outcome=read('S8_PDO_scores_endpoints.tsv')
equal('AUC sample ranks invariant',bool(outcome.author_AUC.rank().equals(outcome.linear_AUC.rank())),True)
wells=read('S8_PDO_public_dose_wells.tsv')
for pdo,g in wells.groupby('PDO_ID'):
    points=g.groupby('cisplatin_uM').relative_viability_percent.mean()
    equal('AUC ranks source well reconstruction '+pdo,np.trapezoid(points.to_numpy(),points.index.to_numpy()),outcome.set_index('PDO_ID').loc[pdo,'linear_AUC'],1e-7)
loo=read('S8_PDO_robustness_LOO.tsv').query("program_or_control=='RMP_C07' and signature=='Top50'")
expected=loo.set_index('excluded_PDO').rho
for name in outcome.PDO_ID:
    other=outcome.loc[outcome.PDO_ID!=name]
    equal('C07 leave one out '+name,spearmanr(other.C07_Top50,other.author_AUC).statistic,expected.loc[name],1e-13)
equal('C07 LOO corrected positives',int((loo.FDR_seven_exact<.05).sum()),4)
l2=read('S8_PDO_robustness_L2O.tsv')
for r in l2.loc[l2.program_or_control=='RMP_C07'].itertuples(index=False):
    g=outcome.loc[~outcome.PDO_ID.isin(r.excluded_pair.split('+'))]
    equal('C07 two deletions '+r.excluded_pair+' '+r.signature,spearmanr(g['C07_'+r.signature],g.author_AUC).statistic,r.rho,1e-13)
exact=read('S8_PDO_exact_results.tsv')
controls=exact.query("subset=='All10_author_AUC' and signature=='Canonical_Control'")
equal('Six-control BH',np.max(abs(multipletests(controls.P_exact_two_sided,method='fdr_bh')[1]-controls.FDR_control_family_exact)),0,1e-14)
day=read('S8_PDO_day15_membership_scores.tsv')
for (scope,endpoint,size),g in read('S8_PDO_day15_exact.tsv').groupby(['scope','endpoint','size']):
    members=day.loc[day[endpoint].notna()]
    if g.n.iloc[0]==9: members=members.loc[members.author_AUC.notna()]
    equal(scope+' '+endpoint+' n',len(members),int(g.n.iloc[0]))
    for r in g.itertuples():
        col=r.RMP.removeprefix('RMP_')+f'_Top{r.size}'
        equal(scope+' '+endpoint+' '+col,spearmanr(members[col],members[endpoint]).statistic,r.rho,1e-13)
    equal(scope+' '+endpoint+' '+str(size)+' exact q from archived counts',np.max(abs(multipletests(g.extreme/g.permutations,method='fdr_bh')[1]-g.q7)),0,1e-13)
mix=read('S7_Liu_gene_effects.tsv').query("RMP=='RMP_C07' and signature=='Top50'")
v=mix.mean_delta_raw_T_minus_N.dropna()
equal('Liu C07 signs',(int(v.gt(0).sum()),int(v.lt(0).sum())),(24,16))
equal('Liu C07 mean member delta',float(v.mean()),-.00551,.000005)
z=read('S7_cross_cohort_gene_effects.tsv').query("RMP=='RMP_C07' and signature=='Top50'").dropna(subset=['Zhao_dz','Liu_dz'])
equal('Zhao C07 finite effects',len(z),35)
e=z.loc[z.fixed80]
equal('Zhao C07 fixed80 genes',len(e),21)
equal('Zhao C07 fixed80 effect rank',spearmanr(e.Zhao_dz,e.Liu_dz).statistic,.7961038961,1e-9)
equal('Zhao C07 fixed80 RNA/protein positive',int(e.rho_tumor.gt(0).sum()),19)
pd.DataFrame(checks,columns=['check','observed','expected','absolute_tolerance']).to_csv(OUTPUT_ROOT/'extended_statistical_checks.tsv',sep='\t',index=False)
print(len(checks),'additional numeric and figure-input checks passed.')


