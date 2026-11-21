"""Deterministic evidence-domain support bridge used in the submitted study."""
import pandas as pd
def evidence_bridge(record):
    if any(pd.isna(record[k]) for k in ['D','R','Q','P']):return 'INDETERMINATE'
    if not record['D'] or record['R'] or not record['Q']:return 'Concentration-cautioned or limited support'
    if record['P']:return 'Joint RNA and proteomic contrast support'
    return 'RNA-domain support without qualified proteomic contrast support'
