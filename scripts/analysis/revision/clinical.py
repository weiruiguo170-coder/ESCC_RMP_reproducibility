
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from common import *
from scipy.stats import mannwhitneyu
from statsmodels.duration.hazard_regression import PHReg
M='06_clinical_conditional';start(M)
tables=[];provenance=[]
for cohort,name in [('GSE53622','GSE53622_clinical_data_of_patients_independent_set.xlsx.gz'),('GSE53624','GSE53624_clinical_data_of_patients_orignial_set.xlsx.gz')]:
 source=R/'01_data_raw'/name
 if not source.exists():source=source.with_suffix('')
 sheets=workbook_cells(source);nonempty=[(k,v) for k,v in sheets.items() if v];assert len(nonempty)==1
 sheet,raw=nonempty[0];df=pd.DataFrame(raw[1:],columns=raw[0]);df['subseries']=cohort
 df['patient_id']=df['Patient ID'].astype(str).str.strip().str.lower();tables.append(df)
 provenance.append(dict(cohort=cohort,official_GEO=f'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={cohort}',source_file=source.name,sha256=sha(source),sheet=sheet,patients=len(df)))
clinical=pd.concat(tables,ignore_index=True);assert len(clinical)==179 and not clinical.duplicated(['subseries','patient_id']).any();save(pd.DataFrame(provenance),M,'clinical_source_provenance.tsv')
scores=read('06_phase_F_spatial_context/05_spatial_scoring/GSE53625_sample_RMP_scores.tsv',M);scores=scores[(scores.tissue=='TUMOR')&(scores.signature_size=='TOP50')];assert len(scores)==179
df=scores.merge(clinical,on=['subseries','patient_id'],how='outer',indicator=True,validate='one_to_one');assert (df._merge=='both').all()
for col in ['Tumor grade','TNM stage','N stage','Sex','Death at FU']:df[col]=df[col].astype('string').str.strip().str.lower()
stage={'i':1,'ii':2,'iii':3,'iv':4,'ia':1,'ib':1,'iia':2,'iib':2,'iiia':3,'iiib':3,'iiic':3};grade={'well':1,'well differentiated':1,'well-differentiated':1,'moderately':2,'moderate':2,'moderately differentiated':2,'poorly':3,'poor':3,'poorly differentiated':3}
df['TNM_stage_ordinal']=df['TNM stage'].map(stage);df['differentiation_ordinal']=df['Tumor grade'].map(grade);df['N_stage_ordinal']=pd.to_numeric(df['N stage'].str.extract(r'^n([0-3])$')[0],errors='coerce');df['M_stage_ordinal']=np.nan;df['OS_months']=pd.to_numeric(df['Survival time(months)'],errors='coerce');df['OS_event']=df['Death at FU'].map({'yes':1,'no':0});df['age_years']=pd.to_numeric(df.Age,errors='coerce');df['sex_male']=df.Sex.map({'male':1,'female':0})
save(df.drop(columns=['_merge']),M,'179_patient_verified_clinical_score_linkage.tsv')
miss=[];categories=[];results=[]
for variable in ['TNM_stage_ordinal','differentiation_ordinal','N_stage_ordinal','M_stage_ordinal','OS_months','OS_event','age_years','sex_male']:
 s=df[variable];miss.append(dict(variable=variable,total=179,n_valid=s.notna().sum(),n_missing=s.isna().sum(),missing_fraction=s.isna().mean()))
 for val,n in s.value_counts(dropna=False).items():categories.append(dict(variable=variable,category=str(val),n=n))
for variable in ['TNM_stage_ordinal','differentiation_ordinal','N_stage_ordinal','M_stage_ordinal']:
 for rmp in IDS:
  sub=df[[variable,rmp]].dropna();levels=sorted(sub[variable].unique());base=dict(RMP=rmp,endpoint=variable,n_patients=len(sub),n_categories=len(levels),signature='Top50',family='all evaluable clinicopathologic associations')
  if len(sub)<20 or len(levels)<2:results.append(dict(**base,status='CONDITIONAL_BLOCKED',reason='M not provided in source table; never inferred from TNM stage' if variable=='M_stage_ordinal' else 'insufficient complete records/categories'));continue
  if len(levels)==2:
   a=sub[sub[variable]==levels[1]][rmp];b=sub[sub[variable]==levels[0]][rmp]
   if min(len(a),len(b))<10:results.append(dict(**base,status='CONDITIONAL_BLOCKED',reason='binary group fewer than10'));continue
   u,p=mannwhitneyu(a,b,alternative='two-sided');effect=2*u/(len(a)*len(b))-1;test='Mann-Whitney two-sided; rank-biserial high-minus-low'
  else:effect,p=spearmanr(sub[variable],sub[rmp]);test='Spearman two-sided asymptotic; ordinal endpoint'
  results.append(dict(**base,status='EVALUATED_EXPLORATORY',effect=effect,P=p,test=test))
res=pd.DataFrame(results);res['BH_FDR']=bh(res.get('P',pd.Series(np.nan,index=res.index)));save(res,M,'all_clinicopathologic_associations.tsv');save(pd.DataFrame(miss),M,'clinical_missingness.tsv');save(pd.DataFrame(categories),M,'clinical_category_counts.tsv')
survival=[];diagnostics=[]
for rmp in IDS:
 sub=df[[rmp,'OS_months','OS_event','age_years','sex_male','TNM_stage_ordinal','subseries']].copy();sub=sub[(sub.OS_months>0)&sub.OS_event.isin([0,1])].dropna(subset=[rmp,'OS_months','OS_event']);n=len(sub);events=int(sub.OS_event.sum());censored=n-events
 if events<20 or censored<20 or n<40:survival.append(dict(RMP=rmp,model='UNIVARIATE',status='CONDITIONAL_BLOCKED',n_patients=n,events=events,reason='insufficient events/censoring per frozen rule'));continue
 sd=sub[rmp].std(ddof=1);sub['score_z']=(sub[rmp]-sub[rmp].mean())/sd
 model=PHReg(sub.OS_months,sub[['score_z']].to_numpy(),status=sub.OS_event,ties='efron').fit(disp=0)
 beta=float(model.params[0]);se=float(model.bse[0]);survival.append(dict(RMP=rmp,model='UNIVARIATE',status='EVALUATED_EXPLORATORY',n_patients=n,events=events,censored=censored,beta_per_SD=beta,HR_per_SD=np.exp(beta),CI_low=np.exp(beta-1.95996398454*se),CI_high=np.exp(beta+1.95996398454*se),P=model.pvalues[0],test='Cox PH partial likelihood; Efron ties; no data-driven cut point'))
 try:
  scho=model.schoenfeld_residuals[:,0];ok=np.isfinite(scho)&(sub.OS_event.to_numpy()==1);rho,p=spearmanr(np.log(sub.OS_months.to_numpy()[ok]),scho[ok]);diagnostics.append(dict(RMP=rmp,model='UNIVARIATE',diagnostic='descriptive Schoenfeld residual vs log-time Spearman; not formal global PH test',rho=rho,P=p,n_events=ok.sum()))
 except Exception as e:diagnostics.append(dict(RMP=rmp,model='UNIVARIATE',diagnostic='Schoenfeld diagnostic unavailable',reason=str(e)))
 adj=sub.dropna(subset=['age_years','sex_male','TNM_stage_ordinal']);stage_dummies=pd.get_dummies(adj.TNM_stage_ordinal.astype(int).astype(str),prefix='stage',drop_first=True,dtype=float);design=pd.concat([adj[['score_z','age_years','sex_male']],stage_dummies],axis=1);required_params=design.shape[1]+(adj.subseries.nunique()-1);ev=int(adj.OS_event.sum())
 if len(adj)<50 or ev<10*required_params:survival.append(dict(RMP=rmp,model='ADJUSTED_SENSITIVITY',status='CONDITIONAL_BLOCKED',n_patients=len(adj),events=ev,reason='complete records/events per parameter gate'));continue
 fitted=PHReg(adj.OS_months,design.to_numpy(float),status=adj.OS_event,strata=adj.subseries,ties='efron').fit(disp=0);beta=float(fitted.params[0]);se=float(fitted.bse[0]);survival.append(dict(RMP=rmp,model='ADJUSTED_SENSITIVITY',status='EVALUATED_EXPLORATORY',n_patients=len(adj),events=ev,censored=len(adj)-ev,beta_per_SD=beta,HR_per_SD=np.exp(beta),CI_low=np.exp(beta-1.95996398454*se),CI_high=np.exp(beta+1.95996398454*se),P=fitted.pvalues[0],covariates='age + sex + categorical TNM stage; baseline hazards stratified by source subseries',parameters_plus_strata=required_params))
s=pd.DataFrame(survival)
for _,ix in s.groupby('model').groups.items():s.loc[ix,'BH_FDR']=bh(s.loc[ix,'P'])
save(s,M,'all_OS_Cox_models.tsv');save(pd.DataFrame(diagnostics),M,'OS_model_diagnostics.tsv')
sig=res[(res.status=='EVALUATED_EXPLORATORY')&(res.BH_FDR<.05)];os_sig=s[(s.status=='EVALUATED_EXPLORATORY')&(s.BH_FDR<.05)]
finish(M,'DONE_VERIFIED_WITH_M_ENDPOINT_BLOCKED',dict(linked_patients=179,clinical_tests=len(res[res.status=='EVALUATED_EXPLORATORY']),Cox_model_rows=len(s),M_status='NOT_AVAILABLE',clinical_q05=len(sig),OS_q05=len(os_sig)))
