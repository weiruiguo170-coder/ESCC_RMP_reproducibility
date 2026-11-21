"""Draw discovery inputs, ranked yields and source-weighting sensitivity."""
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import argparse
from pathlib import Path
from figure_layout import ALIAS_UNIT, BLUE, COHORTS, GRID, HERE, INK, ORANGE, PROGRAMS, SLATE, WIDTH_MM, plt, read_tsv, save, style, write_tsv
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator

def load_data(folder):
    cells = read_tsv(folder / 'cohort_cell_counts.tsv')
    fits = {r['cohort']: r for r in read_tsv(folder / 'formal_cnmf.tsv')}
    summary = read_tsv(folder / 'weighting_summary.tsv')
    rows = sorted(summary, key=lambda r: r['rmp_id'])
    if tuple((r['rmp_id'].removeprefix('RMP_') for r in rows)) != PROGRAMS:
        raise ValueError('Expected one authoritative weighting row per program C01–C07')
    return (cells, fits, rows)

def values_table(summary, warning):
    return [dict(program=r['rmp_id'].removeprefix('RMP_'), GSE188900_program=r['program_B_GSE188900'], n_analytical_aliases=int(r['GSE188900_P_aliases']), RAW_fraction=r['GSE188900_RAW_max'], RAW_max_percent=f"{100 * float(r['GSE188900_RAW_max']):.12f}", RAW_top_alias=r['GSE188900_RAW_top_alias'], RAW_at_or_above_80=float(r['GSE188900_RAW_max']) >= warning, EQUAL_fraction=r['GSE188900_EQUAL_max'], EQUAL_max_percent=f"{100 * float(r['GSE188900_EQUAL_max']):.12f}", EQUAL_top_alias=r['GSE188900_EQUAL_top_alias'], EQUAL_at_or_above_80=float(r['GSE188900_EQUAL_max']) >= warning, EQUAL_minus_RAW_percentage_points=f"{100 * (float(r['GSE188900_EQUAL_max']) - float(r['GSE188900_RAW_max'])):.12f}") for r in summary]

def heading(ax, letter, title, y):
    ax.text(0, y, letter, transform=ax.transAxes, fontsize=12, fontweight='bold', va='center')
    ax.text(0.064 if letter != 'a' else 0.029, y, title, transform=ax.transAxes, fontsize=9.4, fontweight='bold', va='center')

def discovery_table(ax, cells, fits):
    heading(ax, 'a', 'Discovery inputs and cNMF fits', 0.963)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.add_patch(Rectangle((0, 0.584), 1, 0.202, facecolor='#EAF2F8', edgecolor='none'))
    ax.add_patch(Rectangle((0, 0.023), 1, 0.188, facecolor='#F0F3F5', edgecolor='none'))
    for level in (0.584, 0.398, 0.21, 0.023):
        ax.axhline(level, color=GRID, linewidth=0.5)
    columns = (0.016, 0.159, 0.263, 0.537, 0.615, 0.705, 0.822, 0.942)
    headers = ('Cohort', 'Role', 'Input units', 'Cells', 'Rank', 'Silhouette', 'Formal runs\ncompleted', 'Components\nretained')
    for (x, title) in zip(columns, headers):
        ax.text(x, 0.686, title, ha='left' if x < 0.3 else 'center', va='center', weight='bold', fontsize=8, linespacing=1.05)
    unit_labels = ('patients', ALIAS_UNIT, 'source patient labels')
    colors = (BLUE, ORANGE, SLATE)
    for (i, (cohort, y)) in enumerate(zip(COHORTS, (0.491, 0.305, 0.116))):
        group = [r for r in cells if r['cohort'] == cohort]
        m = fits[cohort]
        values = (cohort, 'Auxiliary' if i == 2 else 'Primary', f'{len(group)} {unit_labels[i]}', f"{sum((int(r['n_cells']) for r in group)):,}", f"K = {m['k']}", f"{float(m['silhouette']):.3f}", f"{m['replicates_completed']}/{m['replicates_requested']}", f"{int(m['components_retained']):,}/{int(m['components_total']):,}")
        for (j, (x, value)) in enumerate(zip(columns, values)):
            ax.text(x, y, value, ha='left' if j < 3 else 'center', va='center', color=colors[i] if j == 0 else INK, weight='bold' if j == 0 else 'normal')

def ranked_yields(ax, cells):
    heading(ax, 'b', 'Cell yield by discovery unit', 1.23)
    for (cohort, color, marker) in zip(COHORTS, (BLUE, ORANGE), ('o', 's')):
        group = sorted((int(r['n_cells']) for r in cells if r['cohort'] == cohort), reverse=True)
        shares = [n * 100 / sum(group) for n in group]
        ax.plot(range(1, len(group) + 1), shares, color=color, marker=marker, markersize=2.9 if marker == 'o' else 3.9, linewidth=1, label=cohort)
        if cohort == 'GSE188900':
            ax.annotate(f'{shares[0]:.2f}%', (1, shares[0]), xytext=(6, 1), textcoords='offset points', color=ORANGE, fontsize=8, va='center')
    ax.set(xscale='log', xlim=(0.9, 67), ylim=(0, 75), xlabel='Discovery unit rank (log scale)', ylabel='Share of cohort cells (%)', yticks=[0, 20, 40, 60])
    ticks = [1, 2, 3, 6, 10, 20, 40, 60]
    ax.set_xticks(ticks, labels=[str(t) for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis='y', color=GRID, linewidth=0.5)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.067), ncols=2, frameon=False, handlelength=1.65, handletextpad=0.45, borderaxespad=0, columnspacing=1.2)

def weighting(ax, values, threshold):
    heading(ax, 'c', 'GSE188900 weighting sensitivity', 1.23)
    for (y, row) in enumerate(values):
        raw = float(row['RAW_fraction']) * 100
        equal = float(row['EQUAL_fraction']) * 100
        ax.plot([raw, equal], [y - 0.14, y + 0.14], color='#AAB6C0', linewidth=1.05)
        ax.plot(raw, y - 0.14, color=ORANGE, marker='o', markersize=4.2, zorder=3)
        ax.plot(equal, y + 0.14, color=BLUE, marker='D', markersize=4.05, zorder=3)
    ax.axvline(threshold, color=SLATE, linestyle=(0, (3, 2.6)), linewidth=0.8, zorder=0)
    ax.text(threshold, 1.015, f'{threshold:g}%', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7.3, color=SLATE)
    ax.set(xlim=(0, 101), xticks=range(0, 101, 20), ylim=(6.58, -0.55), xlabel='Maximum contribution (%)', ylabel='Recurrent malignant program')
    ax.set_yticks(range(len(values)), labels=PROGRAMS)
    handles = (Line2D([], [], color=ORANGE, marker='o', linestyle='none', markersize=4.2), Line2D([], [], color=BLUE, marker='D', linestyle='none', markersize=4.05))
    ax.legend(handles, ('RAW (cell-weighted)', 'EQUAL (equal-unit)'), loc='lower left', bbox_to_anchor=(0, 1.067), ncols=2, frameon=False, handlelength=0.75, handletextpad=0.4, borderaxespad=0, columnspacing=1.0, fontsize=8)

def draw(cells, fits, values, warning):
    if tuple((sum((r['cohort'] == c for r in cells)) for c in COHORTS)) != (60, 6, 3):
        raise ValueError('Unexpected discovery or auxiliary unit count')
    if any((r['n_analytical_aliases'] != 6 for r in values)):
        raise ValueError('Panel c requires the historical six-alias grouping')
    style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 105 / 25.4), facecolor='white')
    a = fig.add_axes([0.045, 0.683, 0.93, 0.284])
    b = fig.add_axes([0.095, 0.115, 0.385, 0.42])
    c = fig.add_axes([0.596, 0.115, 0.374, 0.42])
    discovery_table(a, cells, fits)
    ranked_yields(b, cells)
    weighting(c, values, warning * 100)
    for ax in (b, c):
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(length=3, pad=3)
    fig.canvas.draw()
    for artist in fig.axes:
        bbox = artist.get_tightbbox(fig.canvas.get_renderer())
        if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > fig.bbox.x1 or (bbox.y1 > fig.bbox.y1):
            raise ValueError('Figure 2 contains a clipped panel or label')
    return fig
