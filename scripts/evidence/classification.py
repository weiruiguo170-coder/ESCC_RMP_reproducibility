"""Apply the frozen model-B rule with the final descriptive group names."""
from pathlib import Path
import sys,json,importlib.util
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/_paths.py').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from _paths import data_path
spec=importlib.util.spec_from_file_location('edsc_rules',ROOT/'scripts/evidence/decision_rules.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
GROUPS={'Primary':'Joint RNA and proteomic contrast support','Secondary':'RNA-domain support without qualified proteomic contrast support','Exploratory':'Concentration-cautioned or limited support','INDETERMINATE':'Indeterminate'}
def load_config():return json.loads(data_path('E1_RULES_CONFIG.json').read_text())
def classify_edsc(record,config=None,mode=None):
    result=core.classify(record,'B',config or load_config(),mode)
    result['frozen_B_category']=result['category'];result['EDSC_group']=GROUPS[result['category']]
    result['framework']='Frozen evidence-domain support bridge';return result
