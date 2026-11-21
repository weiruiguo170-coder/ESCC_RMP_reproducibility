"""Plot the Task B and C supplements and the corrected Task A audit panel."""
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from common import BLUE, GRAY, INK, ORANGE, TEAL, WIDTH, mm, panel, use_style
HERE = Path(__file__).resolve().parent
DATA = HERE / 'Source_Data'
REPRO = HERE.parent / 'Reproducibility_Update_V2'
TABLES = HERE.parent / 'Supplementary_Tables/Tables'

def read(path):
    return pd.read_csv(resolve_data(path), sep='\t')

def export(fig, path):
    for ext in ('.pdf', '.png'):
        fig.savefig(path.with_suffix(ext), dpi=600)
    with Image.open(path.with_suffix('.png')) as png:
        png.convert('RGB').save(path.with_suffix('.tiff'), dpi=(600, 600), compression='tiff_lzw')
    plt.close(fig)

def protein():
    gene = read(TABLES / 'S7_cross_cohort_gene_effects.tsv').query("RMP == 'RMP_C07'")
    stat = read(TABLES / 'S7_cross_cohort_gene_concordance.tsv').query("RMP == 'RMP_C07'").set_index('scope')
    (fig, ax) = plt.subplots(figsize=(WIDTH, mm(108)))
    fig.subplots_adjust(left=0.13, right=0.77, top=0.94, bottom=0.13)
    ax.axhline(0, color=GRAY, lw=0.7, ls='--')
    ax.axvline(0, color=GRAY, lw=0.7, ls='--')
    for (eligible, color, marker, label) in ((True, TEAL, 'o', 'Meets fixed80'), (False, ORANGE, 'D', 'Other finite effects')):
        selected = gene[gene.fixed80 == eligible]
        ax.scatter(selected.Liu_dz, selected.Zhao_dz, s=27, color=color, marker=marker, edgecolor='white', linewidth=0.4, label=label, zorder=3)
    ax.set(xlabel='Liu et al. paired protein effect ($d_z$)', ylabel='Zhao et al. paired protein effect ($d_z$)')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    (primary, all_genes) = (stat.loc['fixed80'], stat.loc['finite_dz'])
    for (y, label, row) in ((0.71, 'Fixed80 overlap', primary), (0.46, 'All finite effects', all_genes)):
        ax.text(1.02, y, f'{label}\n{row.n_genes:g} genes\nSpearman ρ = {row.rho:.3f}', transform=ax.transAxes, ha='left', va='top', linespacing=1.45)
    export(fig, HERE / 'FigS6')

def fixed_dose():
    table = read(TABLES / 'S8_PDO_day15_exact.tsv').query("RMP == 'RMP_C07'")
    (fig, axes) = plt.subplots(1, 2, figsize=(WIDTH, mm(82)), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.19, top=0.79, wspace=0.26)
    labels = (('CDDP_fixed_dose_viability', '11 µM cisplatin'), ('FU_fixed_dose_viability', '2.46 µM 5-FU'))
    scopes = ('matched_AUC_day15_9', 'all_day15_available_12')
    for (letter, ax, (endpoint, title)) in zip('ab', axes, labels, strict=True):
        panel(ax, letter, title)
        ax.axvline(0, color=GRAY, lw=0.7, ls='--')
        ax.grid(axis='y')
        ax.set_axisbelow(True)
        for (y, scope) in enumerate(scopes):
            selected = table[(table.scope == scope) & (table.endpoint == endpoint)].set_index('size')
            for (size, shift, color, marker) in ((50, -0.13, BLUE, 'o'), (100, 0.13, ORANGE, 's')):
                row = selected.loc[size]
                ax.scatter(row.rho, y + shift, s=25, color=color, marker=marker, zorder=3)
        ax.set(yticks=[0, 1], yticklabels=['AUC overlap\n9 PDOs', 'All measured\n12 PDOs'], ylim=(1.55, -0.55), xlim=(-1.05, 1.05), xlabel='Spearman ρ')
    handles = [plt.Line2D([], [], ls='', marker=m, color=c, label=f'Top{k}') for (k, c, m) in ((50, BLUE, 'o'), (100, ORANGE, 's'))]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.99), ncol=2)
    export(fig, HERE / 'FigS7')
