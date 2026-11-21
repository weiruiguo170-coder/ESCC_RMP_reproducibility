"""Render one final numbered figure from frozen source tables."""
from pathlib import Path
import argparse,sys,importlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'scripts/plotting'))
from _paths import data_path,resolve_data
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('figure');p.add_argument('--output',type=Path,default=ROOT/'results/figures');args=p.parse_args()
    name=args.figure;allowed=[f'Fig{i}' for i in range(1,10)]+[f'FigS{i}' for i in range(1,14)]
    if name not in allowed:p.error('Use Fig1–Fig9 or FigS1–FigS13')
    args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,*unused):
        for ext in ['pdf','svg','png']:fig.savefig(args.output/f'{name}.{ext}',dpi=600,facecolor='white')
        plt.close(fig)
    if name in ['Fig1','Fig2','FigS4','FigS8','FigS10','FigS12']:
        m=importlib.import_module('final_display');save(m.create(name))
    elif name=='Fig3':
        m=importlib.import_module('cohorts');cells,fits,summary=m.load_data(ROOT/'data/processed/figures')
        import workflow
        warning=float(workflow.load_data(ROOT/'data/processed/figures')[1]['warning_fraction'])
        save(m.draw(cells,fits,m.values_table(summary,warning),warning))
    elif name=='Fig9':
        m=importlib.import_module('profile');m.prepare_fig9_inputs=lambda:data_path('Fig9_input_data.tsv')
        m.export_figure=save;m.draw_fig9()
    elif name.startswith('FigS'):
        n=int(name[4:])
        if n in [1,2,3,5]:
            m=importlib.import_module('discovery' if n==1 else 'evidence');save(getattr(m,f'figure_s{n}')(ROOT/'results/tables'))
        elif n in [6,7]:
            m=importlib.import_module('protein_pdo');m.TABLES=ROOT/'results/tables';m.DATA=args.output;m.export=save;getattr(m,'protein' if n==6 else 'fixed_dose')()
        else:
            m=importlib.import_module('supplements');m.save=save;getattr(m,{9:'simulation',11:'cnv',13:'pdo_robustness'}[n])()
    else:
        n=int(name[3:]);m=importlib.import_module('evidence' if n==8 else 'tissue');save(getattr(m,f'figure_{n}')(ROOT/'results/tables'))
    print('Saved',name,'to',args.output)
if __name__=='__main__':main()
