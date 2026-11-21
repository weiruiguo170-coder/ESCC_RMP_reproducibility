
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
import json,hashlib,datetime,sys,platform,logging,shutil,os
import numpy as np,pandas as pd
from scipy import sparse
from scipy.stats import rankdata,spearmanr,wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = INPUT_ROOT;O=OUTPUT_ROOT/'revision';R1=R/'FIG_REVISION_ROUND1';SEED=2026111402
IDS=[f'RMP_C{i:02d}' for i in range(1,8)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def record(p,module,purpose='analysis input'):
 p=Path(p);p=p if p.is_absolute() else R/p
 value=resolve_data(p)
 if isinstance(value,Path) and not value.is_file():raise FileNotFoundError(value)
 return value

def read(p,m,**kw):
 suffix=Path(p).suffix
 return pd.read_csv(record(p,m),sep=',' if suffix=='.csv' else '\t',**kw)

def save(df,m,name):
 target=O/m/name;target.parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(target,sep='\t',index=False,float_format='%.15g')

def bh(p):
 p=np.asarray(p,float);q=np.full(p.shape,np.nan);ok=np.isfinite(p)
 if ok.any():q[ok]=multipletests(p[ok],method='fdr_bh')[1]
 return q
def start(m):
 (O/m).mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(message)s',force=True)
 logging.info('%s; seed=%s',m,SEED)

def finish(m,status,details):print(m,status,details,flush=True)

def signatures(m):
 out={}
 for r in IDS:
  for n in [50,100]:
   p=record(f'06_rmp_annotation/{r}_signature_top{n}.txt',m);g=p.read_text().splitlines();assert len(set(g))==n;out[f'{r}_TOP{n}']=g
 return out
def load_h5(p,m):
 import anndata as ad
 return ad.read_h5ad(record(p,m))
def score_ranks(ranks,idx):
 n=ranks.shape[1];k=len(idx)
 return np.clip((ranks[:,idx].mean(1)-(k+1)/2)/(n-k),0,1) if k else np.full(ranks.shape[0],np.nan)
def heat(df,p,title,vmin=None,vmax=None,cmap='viridis'):
 fig,ax=plt.subplots(figsize=(max(7,len(df.columns)*.8),max(4,len(df)*.25)),layout='constrained');im=ax.imshow(df.to_numpy(),aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap);ax.set_xticks(range(len(df.columns)),df.columns,rotation=45,ha='right');ax.set_yticks(range(len(df)),df.index,fontsize=7);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7);fig.savefig(p,dpi=170);plt.close(fig)

def workbook_cells(path):
 """Read complete source-sheet cell grids, including blank leading rows."""
 import openpyxl,gzip,io
 p=Path(path);raw=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
 wb=openpyxl.load_workbook(io.BytesIO(raw),read_only=True,data_only=True)
 result={ws.title:[list(row) for row in ws.iter_rows(values_only=True)] for ws in wb.worksheets}
 wb.close();return result
