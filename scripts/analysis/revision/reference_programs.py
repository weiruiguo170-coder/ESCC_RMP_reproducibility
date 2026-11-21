"""Recalculate measured-universe overlaps with official Zhang and Gavish lists.

Reference memberships are external author resources, obtained under their terms.
This calculation uses the frozen study signatures and shared measured background.
"""
from pathlib import Path
import sys,argparse,collections
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
sys.path.insert(0,str(ROOT/'scripts'))
from _paths import read_table,data_path,OUTPUT_ROOT
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

def calculate(zhang_workbook,gavish_tsv,hgnc):
    h=pd.read_csv(hgnc,sep='\t',dtype=str);approved=set(h.symbol.dropna());aliases=collections.defaultdict(set)
    for row in h.itertuples():
        for col in ['alias_symbol','prev_symbol']:
            value=getattr(row,col)
            if isinstance(value,str):
                for a in value.split('|'):aliases[a.strip().upper()].add(row.symbol)
    def canon(g):
        x=str(g).strip().upper()
        return x if x in approved or len(aliases.get(x,set()))!=1 else next(iter(aliases[x]))
    raw=pd.read_excel(zhang_workbook,sheet_name='Genes',header=None,dtype=object)
    lists=[]
    for j,col in enumerate('ABCDEFGH'):
        header=raw.iloc[2,j];name=f'{header} [column {col}]' if header=='Epi1/Epi2' else header
        genes=raw.iloc[3:,j].dropna().astype(str).tolist()
        assert len(genes)==len(set(genes))==30,(col,len(genes))
        lists.append(('Zhang2021_ESCC8',name,{canon(g) for g in genes}))
    gavish=pd.read_csv(gavish_tsv,sep='\t');gavish=gavish[gavish.cell_type.eq('Cancer')]
    assert gavish.reference_program.nunique()==41
    for name,g in gavish.groupby('reference_program'):lists.append(('Gavish2023_author_Cancer41',name,{canon(x) for x in g.gene}))
    universe=set(read_table('S9_overlap_gene_background.tsv').gene);assert len(universe)==16131
    out=[]
    for size in [50,100]:
        for i in range(1,8):
            rmp=f'RMP_C{i:02d}'
            original=data_path(f'{rmp}_signature_top{size}.txt').read_text().splitlines();assert len(original)==size
            full_a={canon(g) for g in original};a=full_a&universe
            for collection,name,full_b in lists:
                b=full_b&universe;k=len(a&b)
                out.append(dict(collection=collection,signature_size=size,RMP=rmp,reference_program=name,
                    background_size=len(universe),frozen_original_n=size,original_reference_n=len(full_b),
                    RMP_background_n=len(a),reference_background_n=len(b),intersection=k,
                    Jaccard=k/len(a|b),overlap_coefficient=k/min(len(a),len(b)),
                    P_hypergeom=hypergeom.sf(k-1,len(universe),len(a),len(b)),shared_genes=';'.join(sorted(a&b))))
    df=pd.DataFrame(out)
    for _,ix in df.groupby(['collection','signature_size']).groups.items():df.loc[ix,'BH_FDR']=multipletests(df.loc[ix,'P_hypergeom'],method='fdr_bh')[1]
    return df

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['zhang-workbook','gavish-tsv','hgnc']:p.add_argument('--'+flag,required=True,type=Path)
    p.add_argument('--output',type=Path,default=OUTPUT_ROOT/'reference_programs');args=p.parse_args()
    actual=calculate(args.zhang_workbook,args.gavish_tsv,args.hgnc)
    expected=read_table('S9_published_program_overlap.tsv');keys=['collection','signature_size','RMP','reference_program']
    expected=expected.set_index(keys).sort_index();actual=actual.set_index(keys).sort_index();assert actual.index.equals(expected.index)
    for c in ['background_size','frozen_original_n','original_reference_n','RMP_background_n','reference_background_n','intersection','Jaccard','overlap_coefficient','P_hypergeom','BH_FDR']:
        assert np.allclose(actual[c],expected[c],rtol=1e-10,atol=1e-14),c
    assert actual.shared_genes.fillna('').equals(expected.shared_genes.fillna(''))
    args.output.mkdir(parents=True,exist_ok=True);actual.reset_index().to_csv(args.output/'published_program_overlap.tsv',sep='\t',index=False)
    print('All 686 official-list overlaps and complete collection/size BH families reproduced.')
if __name__=='__main__':main()
