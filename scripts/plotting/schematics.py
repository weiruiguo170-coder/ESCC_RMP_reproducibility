"""Original framework/contribution drawing bodies, with portable paths."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=Path(__file__).resolve().parent
DATA=ROOT/'results/tables'
BLUE,ORANGE,TEAL,PURPLE='#205C84','#A75430','#176C65','#655186'
def data(name):return pd.read_csv(resolve_data(DATA/name),sep='\t')
def title(ax,letter,text):ax.set_title(letter+'  '+text,loc='left',fontsize=9,fontweight='bold',pad=9)
def save(fig,name):raise RuntimeError('Caller must capture the figure; never write original locations')

def framework():
    fig=plt.figure(figsize=(6.85,6.05))
    ax=fig.add_axes([0,0,1,1],xlim=(0,174),ylim=(154,0));ax.axis('off')
    items=[
    ('Structural recurrence','Independent fits → reciprocal best match','80 pairs · BH80 · alternative margins'),
    ('Source representation','Cell-weighted and equal-source contribution','Patient IDs or specimen aliases · concentration'),
    ('External RNA','Patient excess correlation → cohort mean','Eligibility · matched background · ≥8/14 test'),
    ('Tissue and clinical context','Tumor-only correlation or paired tissue shift','179 pairs · 34 histology observations · Cox'),
    ('Paired tissue proteins','Signed abundance change among measured genes','124 pairs · coverage · member directions'),
    ('Organoid association','RNA score versus author cisplatin AUC','10 PDOs · exact permutation · BH7'),
    ]
    ax.text(7,6,'a  Evidence targets and observation units',fontsize=10,fontweight='bold',va='center')
    for i,(h,t,foot) in enumerate(items):
        x=7 if i%2==0 else 91;y=13+(i//2)*23
        ax.add_patch(FancyBboxPatch((x,y),76,19,boxstyle='round,pad=0,rounding_size=1.5',fc='#F1F6F8',ec='#ACBFCA',lw=.7))
        ax.text(x+3,y+4.1,h,fontsize=9,color=BLUE,fontweight='bold')
        ax.text(x+3,y+9.8,t,fontsize=7.8)
        ax.text(x+3,y+15,foot,fontsize=7.5,color='#425A66')
    ax.text(7,88,'b  Decision for each claim',fontsize=10,fontweight='bold')
    for x,h,t,c in [(7,'Supported within scope','Eligible data and stated\ninferential conditions met',TEAL),
                     (63,'Not established','Eligible data, with the\nspecified criterion unmet',ORANGE),
                     (119,'Not evaluable','Required input, patient count\nor attainable precision absent',PURPLE)]:
        ax.add_patch(FancyBboxPatch((x,94),48,18,boxstyle='round,pad=0,rounding_size=1.5',fc='white',ec=c,lw=.9))
        ax.text(x+24,99,h,ha='center',fontsize=8.5,color=c,fontweight='bold')
        ax.text(x+24,106,t,ha='center',va='center',fontsize=7.5)
    ax.text(7,122,'c  Program profile and descriptive audit bridge',fontsize=10,fontweight='bold')
    ax.text(7,128,'Continuous effects retain units, comparators, uncertainty and measurement conditions.',fontsize=8.2)
    ax.text(7,135,'The historical D/R/Q/P map first checks missing inputs, then D, R and Q, then P.',fontsize=8.2)
    ax.text(7,142,'C01/C02/C05 · joint RNA/protein      C07 · RNA domain      C03/C04/C06 · cautioned',fontsize=8.2)
    ax.text(7,149,'The bridge is conditional on the frozen signatures, reference RNA setting and six source aliases.',fontsize=7.8,color='#425A66')
    save(fig,'FigS10')

def contributions():
    d=data('S1_bounded_donor_sensitivity.tsv')
    fig,axes=plt.subplots(1,2,figsize=(6.85,3.8))
    fig.subplots_adjust(left=.09,right=.96,top=.83,bottom=.23,wspace=.4)
    y=np.arange(7)
    for ax,field,title_text in zip(axes,['RAW','EQUAL'],['Cell-weighted share','Equal-source share']):
        title(ax,'a' if field=='RAW' else 'b',title_text)
        ax.hlines(y,d[field+'_min']*100,d[field+'_max']*100,color=BLUE,lw=2)
        ax.scatter(d[field+'_min']*100,y,c=BLUE,s=10)
        ax.scatter(d[field+'_max']*100,y,c=BLUE,s=10)
        ax.axvline(80,ls='--',lw=.8,color=ORANGE)
        ax.set(yticks=y,yticklabels=d.RMP.str[-3:],xlim=(0,101),ylim=(6.7,-.7),xlabel='Maximum contribution across 16 groupings (%)')
        for yy,n in zip(y,d[field+'_caution_N16']):
            ax.text(101.5,yy,f'{n}/16',va='center',fontsize=7.5)
    fig.text(.5,.045,'Ranges cover the original grouping and 15 single-pair merges of six specimen aliases.',ha='center',fontsize=8)
    save(fig,'FigS12')
