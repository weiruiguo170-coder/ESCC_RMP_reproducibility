"""Render the historical domain masking panel from its archived display tables."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
OUT=Path(__file__).resolve().parent
DATA=ROOT/'data/processed/figures'
def data(name):return pd.read_csv(resolve_data(DATA/name),sep='\t')
P=data('program_profiles.tsv').query('signature_size==50').sort_values('RMP')
def test(name,value):
    if not value:raise ValueError(name)
INK='#172b3a'; GRAY='#52636d'; GRID='#dce3e7'
BLUE='#2a648d'; TEAL='#24776c'; ORANGE='#b55b2a'; PURPLE='#72578c'
LIGHT='#f2f6f8'
COLS=['#2a648d','#ae5729','#3e8571','#735d93','#6e7e46','#697983','#a55178']
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'axes.labelcolor':INK,
 'text.color':INK,'xtick.color':INK,'ytick.color':INK,'xtick.labelsize':7.5,'ytick.labelsize':8,
 'axes.edgecolor':GRAY,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,
 'grid.color':GRID,'grid.linewidth':.5,'lines.linewidth':1.2,'svg.fonttype':'none','pdf.fonttype':42,
 'savefig.facecolor':'white','figure.facecolor':'white','legend.frameon':False,'legend.fontsize':7.5})

def panel(ax,letter,title):
    ax.text(-.14,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=11,va='bottom')
    ax.text(0,1.10,title,transform=ax.transAxes,fontsize=9,va='bottom')
def figs8():
    counts=data('strategy_counts.tsv');counts=counts[counts.signature_size==50].copy()
    mask=data('domain_masking_summary.tsv');mask=mask[mask.signature_size==50].copy()
    cond=data('condition_summary.tsv')
    thresh=cond[cond.family=='threshold alternatives'];qc=cond[cond.family.str.contains('QC',case=False)]
    test('threshold sensitivity 17 of 280',thresh.changed.sum()==17 and thresh.rows.sum()==280)
    test('QC sensitivity 6 of 126',qc.changed.sum()==6 and qc.rows.sum()==126)
    fig,axs=plt.subplots(2,2,figsize=(174/25.4,170/25.4))
    fig.subplots_adjust(left=.105,right=.97,top=.925,bottom=.145,wspace=.36,hspace=.76)
    ax=axs[0,0];panel(ax,'a','Questions made evaluable')
    fields=[('supported_within_scope','Supported',BLUE),('limited_support','Limited',ORANGE),('not_established','Not established','#9cabb4'),('not_evaluable','Unevaluable','#e9eef1')]
    left=np.zeros(4)
    for field,label,col in fields:
        vals=counts[field].to_numpy();ax.barh(range(4),vals,left=left,color=col,label=label,height=.62);left+=vals
    for i,r in enumerate(counts.itertuples()):ax.text(57,i,f'{r.evaluable}/56',va='center',fontsize=7.3)
    ax.set_xlim(0,70);ax.set_xticks([0,14,28,42,56]);ax.set_yticks(range(4),counts.strategy);ax.set_ylim(3.6,-.6);ax.set_xlabel('Program-claim cells (Top50)');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(loc='lower left',bbox_to_anchor=(-.04,-.50),ncol=2,columnspacing=.8,handlelength=1.1,handletextpad=.5,fontsize=7.1)
    ax=axs[0,1];panel(ax,'b','Information exposed by masking')
    order=['PATIENT','Q','TECH','CONTEXT','PROTEIN','PDO'];mask=mask.set_index('domain').loc[order]
    ax.barh(range(6),mask.empirical_B_cells,color=TEAL,height=.58,label='Measured qualification')
    ax.barh(range(6),mask.generic_B_cells,left=mask.empirical_B_cells,color='#dbe4e9',height=.58,label='General qualification')
    for i,(m,g) in enumerate(zip(mask.empirical_B_cells,mask.generic_B_cells)):
        if m:ax.text(m/2,i,str(m),ha='center',va='center',color='white',fontsize=7.5)
        if g:ax.text(m+g/2,i,str(g),ha='center',va='center',color=GRAY,fontsize=7.5)
        if not m and not g:ax.text(.3,i,'0',va='center',fontsize=7.5,color=GRAY)
    ax.set_yticks(range(6),['Patient','Coordination','Technical','Context','Protein','PDO']);ax.set_ylim(5.6,-.6);ax.set_xlim(0,22);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('Cross-claim qualifications');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.legend(loc='lower left',bbox_to_anchor=(-.04,-.47),fontsize=7.1,handlelength=1.1)
    ax=axs[1,0];panel(ax,'c','Cell yield and unit activity')
    for k in ('C03','C04','C06'):
        r=P[P.RMP=='RMP_'+k].iloc[0];i=int(k[-2:])-1;a=r.RAW_max_contribution*100;b=r.EQUAL_max_contribution*100
        ax.plot([0,1],[a,b],color=COLS[i],marker='o',ms=4)
        ax.text(-.08,a,f'{k}  {a:.1f}',ha='right',va='center',fontsize=7.5,color=COLS[i])
        label_y={'C03':46,'C06':37}.get(k,b)
        if label_y!=b:ax.plot([1.01,1.07],[b,label_y],color=COLS[i],lw=.65)
        ax.text(1.08,label_y,f'{b:.1f}  {k}',ha='left',va='center',fontsize=7.5,color=COLS[i])
    ax.axhline(80,color=ORANGE,ls=(0,(3,2)),lw=.7);ax.set_xlim(-.55,1.55);ax.set_ylim(32,100);ax.set_xticks([0,1],['Cell-weighted','Equal-unit']);ax.set_ylabel('Maximum contribution (%)');ax.set_yticks([40,60,80,100]);ax.spines['bottom'].set_visible(False)
    ax=axs[1,1];panel(ax,'d','Technical condition dependence')
    tech=data('technical_conditions.tsv')
    d=tech[(tech.RMP=='C04')&(tech.Signature_Size==50)&(tech.Perturbation_Type=='depth')].copy().sort_values('Perturbation_Level')
    test('C04 depth values',len(d)==3 and np.allclose(d.Recovery_Rate,[.15,.35,.7]))
    ax.plot(d.Perturbation_Level,d.Recovery_Rate,color=COLS[3],marker='o',ms=4)
    for r in d.itertuples():ax.text(r.Perturbation_Level,r.Recovery_Rate+.07,f'{round(r.Recovery_Rate*20)}/20',ha='center',fontsize=7.5,color=COLS[3])
    ax.set_xlim(.12,1.12);ax.set_ylim(0,1.04);ax.set_xticks([.25,.5,1],['25%','50%','100%']);ax.set_yticks([0,.25,.5,.75,1]);ax.set_xlabel('Retained sequencing depth');ax.set_ylabel('C04 identity recovery');ax.grid(axis='y');ax.set_axisbelow(True)
    fig.text(.105,.079,'EDSC condition changes: 17 / 280 threshold records; 6 / 126 QC records.',fontsize=8,fontweight='bold')
    fig.text(.105,.050,'a-b quantify evaluability and interpretation, not accuracy. Cross-claim status changes after masking: 0.',fontsize=7.2,color=GRAY)
    fig.text(.105,.028,'c uses fixed usages and original sample aliases. d uses 20 technical repeats per condition, not new patients.',fontsize=7.2,color=GRAY)
    save(fig,'FigS8_Framework_Necessity_and_Conditions')

