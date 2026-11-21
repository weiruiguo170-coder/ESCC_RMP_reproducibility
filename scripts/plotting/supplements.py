"""Recreate Figures S1 to S13 using the shipped numerical records."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table
from pathlib import Path
import sys, importlib, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA = ROOT / 'results/tables'
FIG = BASE / 'FIG_SUPPLEMENTARY_FIGURES'
INPUT = HERE / 'figure_inputs'
sys.path.insert(0, str(HERE / 'analysis/figure_code/04_Figure_Source_Code'))
from common import use_style
use_style()
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'escc-supplement-rebuild', 'axes.spines.right': False, 'axes.spines.top': False})
(BLUE, ORANGE, TEAL, PURPLE) = ('#205C84', '#A75430', '#176C65', '#655186')

def data(name):
    return pd.read_csv(resolve_data(DATA / name), sep='\t')

def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.canvas.draw()
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine(None)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=600, facecolor='white')
    with Image.open(FIG / f'{name}.png') as im:
        im.convert('RGB').save(FIG / f'{name}.tiff', dpi=(600, 600), compression='tiff_lzw')
    plt.close(fig)

def title(ax, letter, text):
    ax.set_title(letter + '  ' + text, loc='left', fontsize=9, fontweight='bold', pad=9)

def simulation():
    perf = data('S10_world_level_rates_recomputed.tsv')
    order = ['GAMMA_EXOGENOUS', 'LOGNORMAL_ENDOGENOUS', 'LOGNORMAL_EXOGENOUS']
    (fig, axes) = plt.subplots(1, 3, figsize=(6.85, 3.25))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.31, top=0.83, wspace=0.47)
    for (i, ax) in enumerate(axes[:2]):
        title(ax, chr(97 + i), ['False majority calls', 'True majority detection'][i])
        sel = perf[(perf.program == ('FOCAL' if i == 0 else 'BROAD')) & (perf.majority_truth == bool(i))]
        for (offset, rule, col) in [(-0.16, 'Historical mean-and-sign', BLUE), (0.16, 'Direct PC8', ORANGE)]:
            g = sel[sel.rule == rule].set_index('mechanism').loc[order]
            ax.bar(np.arange(3) + offset, g.rate * 100, width=0.28, color=col, label=rule)
            ax.errorbar(np.arange(3) + offset, g.rate * 100, yerr=[(g.rate - g.CI_low) * 100, (g.CI_high - g.rate) * 100], fmt='none', ecolor='#263C46', lw=0.7, capsize=2)
        ax.set(xticks=range(3), xticklabels=['G', 'LN-E', 'LN-X'], ylim=(0, 110), ylabel='Simulated worlds (%)')
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=7)
    s = data('S10_controlled_sensitivity_depth_predictiveness.tsv').set_index('mechanism').loc[order]
    ax = axes[2]
    title(ax, 'c', 'Depth-related switches')
    for (off, col, field, label) in [(-0.16, BLUE, 'raw_switch_given_sensitive', 'Sensitive at 1×'), (0.16, ORANGE, 'raw_switch_given_stable', 'Stable at 1×')]:
        ax.bar(np.arange(3) + off, s[field] * 100, width=0.28, color=col, label=label)
    ax.set(xticks=range(3), xticklabels=['G', 'LN-E', 'LN-X'], ylim=(0, 110), ylabel='Program–world records (%)')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), fontsize=7)
    save(fig, 'FigS9')

def cnv():
    d = data('S1_CNV_configurations_by_patient.tsv')
    configs = ['RAW_AUTO_C2000_W100', 'LOG_AUTO_FULL_W100', 'LOG_ALL_FULL_W100', 'LOG_ALL_FULL_W50', 'LOG_ALL_FULL_W150']
    configs = [c for c in configs if c in set(d.condition)]
    (fig, axes) = plt.subplots(1, 2, figsize=(6.85, 3.9))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.36, top=0.84, wspace=0.4)
    a = axes[0]
    title(a, 'a', 'Candidate counts by configuration')
    for (j, (pid, c)) in enumerate(zip(['P1', 'P2', 'P4'], [BLUE, ORANGE, TEAL])):
        y = d[d.patient_id == pid].set_index('condition').loc[configs].selected_N
        a.plot(range(len(configs)), y, 'o-', lw=1, ms=3, label=pid, color=c)
    a.set(xticks=range(len(configs)), xticklabels=['Raw\nchunked', 'Log\nautosomal', 'Log\nall genes', 'Window\n50', 'Window\n150'][:len(configs)], ylabel='Operational candidates')
    a.legend(frameon=False, fontsize=8)
    a = axes[1]
    title(a, 'b', 'Patient concentration')
    counts = np.array([[44, 30, 415], [72, 6, 914], [20, 2, 603]])
    baseline = np.zeros(3)
    for (i, (pid, c)) in enumerate(zip(['P1', 'P2', 'P4'], [BLUE, ORANGE, TEAL])):
        a.bar(range(3), counts[:, i] / counts.sum(1) * 100, bottom=baseline, color=c, label=pid)
        baseline += counts[:, i] / counts.sum(1) * 100
    a.set(xticks=range(3), xticklabels=['Historical\n489', 'Repaired\n992', 'Constrained\n625'], ylim=(0, 105), ylabel='Candidate fraction (%)')
    fig.text(0.5, 0.06, 'Selections use different operational definitions. Malignant identity remains unresolved.', ha='center', fontsize=8)
    save(fig, 'FigS11')

def pdo_robustness():
    loo = data('S8_PDO_robustness_LOO.tsv').query("program_or_control=='RMP_C07'")
    two = data('S8_PDO_robustness_L2O.tsv').query("program_or_control=='RMP_C07'")
    (fig, axes) = plt.subplots(1, 2, figsize=(6.85, 4.1))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.82, wspace=0.35)
    ax = axes[0]
    title(ax, 'a', 'Single deletion')
    names = sorted(loo.excluded_PDO.unique(), key=lambda t: int(t.replace('ESCCO', '')))
    for (shift, size, col) in [(-0.1, 'Top50', BLUE), (0.1, 'Top100', ORANGE)]:
        g = loo[loo.signature == size].set_index('excluded_PDO').loc[names]
        ax.scatter(g.rho, np.arange(10) + shift, c=col, s=16, label=size)
        sig = g.FDR_seven_exact < 0.05
        ax.scatter(g.rho[sig], np.arange(10)[sig] + shift, s=38, facecolor='none', edgecolor=col, lw=0.8)
    ax.set(yticks=range(10), yticklabels=[x.replace('ESCCO', '') for x in names], xlabel='Spearman ρ', ylabel='Omitted PDO identifier', xlim=(-1, -0.4))
    ax.invert_yaxis()
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    ax = axes[1]
    title(ax, 'b', 'Double deletion')
    for (shift, size, col) in [(-0.11, 'Top50', BLUE), (0.11, 'Top100', ORANGE)]:
        g = two[two.signature == size].sort_values('excluded_pair')
        x = np.linspace(-0.085, 0.085, len(g)) + shift
        ax.scatter(x, g.rho, c=col, s=10, alpha=0.7, label=size)
    ax.set(xticks=[-0.11, 0.11], xticklabels=['Top50', 'Top100'], ylabel='Spearman ρ', ylim=(-1, -0.4))
    fig.text(0.5, 0.07, 'All 10 single and 45 double deletions per size. Rings mark single-deletion BH7 q < 0.05.', ha='center', fontsize=8)
    save(fig, 'FigS13')
