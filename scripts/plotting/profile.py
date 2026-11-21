"""Draw the final seven-program evidence profile from frozen source tables."""
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import csv
import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedFormatter, FixedLocator
import numpy as np
import pandas as pd
from PIL import Image
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MAIN = OUTPUT_ROOT / 'figures'
DPI = 600
WIDTH_MM = 174
INK = '#172B36'
MUTED = '#425762'
BLUE = '#005C91'
AMBER = '#8D4A19'
TEAL = '#006E62'
PURPLE = '#59447F'
PDO = '#8B3E68'
FRAME = '#9CAEB6'
SOFT = '#DCE4E8'

def configure_style() -> None:
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 8, 'text.color': INK, 'axes.labelcolor': INK, 'axes.unicode_minus': True, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'fig-escc-adjudicated-20261119', 'xtick.color': INK, 'ytick.color': INK, 'savefig.facecolor': 'white'})

def plate(height_mm: float, width_mm: float=WIDTH_MM):
    configure_style()
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), facecolor='white')
    canvas = fig.add_axes([0, 0, 1, 1], xlim=(0, width_mm), ylim=(height_mm, 0))
    canvas.set_axis_off()
    return (fig, canvas)

def export_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f'{stem}.pdf', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(out_dir / f'{stem}.svg', metadata={'Date': None})
    buf = Path(out_dir / f'{stem}.png')
    fig.savefig(buf, format='png', dpi=DPI, facecolor='white')
    with Image.open(buf) as im:
        rgb = im.convert('RGB')
        rgb.save(buf, dpi=(DPI, DPI))
        rgb.save(out_dir / f'{stem}.tiff', dpi=(DPI, DPI), compression='tiff_lzw')
    plt.close(fig)

@dataclass(frozen=True)
class Node:
    x: float
    y: float
    w: float
    h: float
    title: str
    lines: str
    color: str
    role: str = 'data'

@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    payload: str
    basis: str
    points: tuple
    net: str = ''

def prepare_fig9_inputs():
    return data_path('Fig9_input_data.tsv')

def _qfmt(v: str) -> str:
    q = float(v)
    if q >= 0.001:
        return f'{q:.3f}'
    power = math.floor(math.log10(q))
    return f'{q / 10 ** power:.1f}e{power}'.replace('-', '−')

def draw_fig9() -> None:
    data = prepare_fig9_inputs()
    with data.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    assert [r['RMP'] for r in rows] == [f'RMP_C0{i}' for i in range(1, 8)]
    H = 148.0
    (PLOT_TOP, PLOT_BOTTOM) = (19.8, 102.5)
    ROWS = [26.6 + 11.25 * i for i in range(7)]
    (DOT_DY, NOTE_DY) = (-1.1, 3.25)
    OK = 'SUPPORTED_WITHIN_SCOPE'
    STATES = {OK: '+', 'NOT_ESTABLISHED': '−', 'NOT_EVALUABLE': '×'}
    (fig, layer) = plate(H)
    text_checks = []

    def txt(x, y, s, *, size=7.8, color=INK, bold=False, ha='left', region=None):
        a = layer.text(x, y, s, fontsize=size, color=color, fontweight='bold' if bold else 'normal', ha=ha, va='center', linespacing=1.12, zorder=5)
        text_checks.append((a, region if region else (0, WIDTH_MM, 0, H)))
        return a

    def heading(edge, right, letter, title, color, line1=None, line2=None):
        txt(edge, 5.1, letter, size=9.6, color=color, bold=True, region=(edge - 0.2, right, 2, 8.5))
        txt(edge + 4.1, 5.1, title, size=8.4, bold=True, region=(edge + 3.8, right + 0.4, 1.5, 8.5))
        if line1:
            txt(edge, 11.0, line1, size=7.4, color=MUTED, region=(edge - 0.2, right + 0.8, 8.2, 13.6))
        if line2:
            txt(edge, 15.65, line2, size=7.4, color=MUTED, region=(edge - 0.2, right + 0.8, 13.3, 18.4))
    ranges = ((28.8, 51.4), (55.4, 80.8), (84.6, 114.5), (118.9, 145.8), (150.1, 172.0))
    (a, b, c, d, e) = ranges
    txt(2.3, 5.1, 'Top50 program', size=8.5, bold=True, region=(2, 26.5, 1.5, 8.5))
    heading(*a, 'a', 'Recurrence', BLUE)
    heading(*b, 'b', 'Source use', AMBER, '6 patients')
    heading(*c, 'c', 'External RNA', TEAL, 'GSE199654 · 10/14', 'Specs · ≥8/14 test')
    heading(*d, 'd', 'Tissue protein', PURPLE, '124 patient pairs', 'q · measured /50')
    heading(*e, 'e', 'Cisplatin', PDO, '10 PDOs', 'Exploratory AUC')

    def symbol_key(x, y, shape, color, fill, text, right):
        layer.plot(x, y, marker=shape, ms=4.5, markerfacecolor=fill, markeredgecolor=color, mew=0.9, clip_on=False)
        txt(x + 2.2, y, text, size=7.4, region=(x + 2.0, right, 8, 18.5), color=MUTED)
    symbol_key(a[0] + 1, 11.0, 'o', BLUE, BLUE, 'Composite', a[1])
    symbol_key(a[0] + 1, 15.65, 'D', BLUE, 'white', 'Min. margin', a[1])
    symbol_key(b[0] + 1, 15.65, 'o', AMBER, AMBER, 'RAW', b[0] + 12.2)
    symbol_key(b[0] + 13.4, 15.65, 's', AMBER, 'white', 'EQUAL', b[1] + 0.4)
    layer.plot([2.1, 172], [18.8, 18.8], lw=0.5, c=FRAME)
    for y in ROWS[:-1]:
        layer.plot([2.1, 172], [y + 5.5, y + 5.5], lw=0.35, c=SOFT)
    themes = {'Stress / inflammation': 'Stress /\ninflammation', 'Epithelial inflammation': 'Epithelial\ninflammation'}
    for (y, r) in zip(ROWS, rows, strict=True):
        txt(2.3, y - 1.9, r['RMP'][4:], size=8.6, bold=True, region=(2, 26.5, y - 5.2, y))
        txt(2.3, y + 2.25, themes.get(r['display_theme'], r['display_theme']), size=7.5, color=MUTED, region=(2, 26.5, y - 1, y + 5.4))

    def axis(region, limits, ticks, ticklabels, label, guide=None):
        (l, r) = region
        ax = fig.add_axes([l / WIDTH_MM, 1 - PLOT_BOTTOM / H, (r - l) / WIDTH_MM, (PLOT_BOTTOM - PLOT_TOP) / H], xlim=limits, ylim=(PLOT_BOTTOM, PLOT_TOP), facecolor='none')
        ax.set_yticks([])
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(ticklabels))
        ax.set_xlabel(label, fontsize=7.8, labelpad=4)
        for side in ('left', 'top', 'right'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.62)
        ax.spines['bottom'].set_color('#4C606A')
        ax.tick_params(axis='x', labelsize=7.6, length=2.4, width=0.6, pad=2)
        if guide is not None:
            ax.axvline(guide, c=FRAME, lw=0.7, ls=(0, (2.5, 2.2)), zorder=1)
        return ax
    axa = axis(a, (-0.034, 0.74), (0, 0.35, 0.7), ['0', '0.35', '0.70'], 'Similarity units')
    axb = axis(b, (-0.035, 1.035), (0, 0.5, 1), ['0', '50', '100'], 'Max. share (%)', 0.8)
    axc = axis(c, (-0.062, 0.248), (-0.05, 0, 0.1, 0.2), ['−.05', '0', '.10', '.20'], 'Excess correlation', 0)
    axd = axis(d, (-2.86, 2.3), (-2, 0, 2), ['−2', '0', '2'], 'Paired effect, $\\mathregular{d_z}$', 0)
    axe = axis(e, (-1.06, 1.06), (-1, 0, 1), ['−1', '0', '+1'], 'Spearman ρ', 0)

    def point(ax, x, y, shape, color, filled=True, size=4.6):
        ax.plot(x, y, shape, ms=size, markerfacecolor=color if filled else 'white', markeredgecolor=color, markeredgewidth=0.9, zorder=4)

    def estimate(ax, v, lo, hi, y, color, filled):
        assert lo <= v <= hi
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt='o', ms=4.4, lw=0.95, color=color, ecolor=color, markerfacecolor=color if filled else 'white', markeredgecolor=color, markeredgewidth=0.9, capsize=2.0, capthick=0.7, zorder=4)

    def note(ax, region, y, text, *, bold=False, color=MUTED):
        item = ax.text(0.5, y + NOTE_DY, text, transform=ax.get_yaxis_transform(), va='center', ha='center', fontsize=7.6, color=color, fontweight='bold' if bold else 'normal', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.42}, zorder=5)
        text_checks.append((item, (*region, y + 1.0, y + 5.6)))
    for (y, r) in zip(ROWS, rows, strict=True):
        plot_y = y + DOT_DY
        point(axa, float(r['D_match_similarity']), plot_y, 'o', BLUE)
        point(axa, float(r['D_margin_min']), plot_y, 'D', BLUE, False)
        raw = float(r['GSE188900_alias_RAW'])
        eq = float(r['GSE188900_alias_EQUAL'])
        (raw_y, eq_y) = (plot_y - 0.86, plot_y + 0.86)
        axb.plot([raw, eq], [raw_y, eq_y], color=AMBER, lw=0.91, alpha=0.74, zorder=2)
        point(axb, raw, raw_y, 'o', AMBER)
        point(axb, eq, eq_y, 's', AMBER, False)
        q = float(r['Q_effect'])
        (qlo, qhi) = (float(r['Q_CI_low']), float(r['Q_CI_high']))
        q_dep = float(r['Q_depth_effect'])
        estimate(axc, q, qlo, qhi, plot_y - 0.7, TEAL, r['RNA_ref_mean_state'] == OK)
        point(axc, q_dep, plot_y + 0.95, 'D', TEAL, False, size=3.7)
        count = r['supported_specs_effect_state'] + '/' + r['assessed_specs_effect_state']
        status = STATES[r['Q_reference_PC8_state']]
        note(axc, c, y, f'{count}   ·   {status}')
        (p, lo, hi, pq) = [float(r[k]) for k in ('P_dz', 'P_dz_CI_low', 'P_dz_CI_high', 'P_matched_q')]
        support = (lo > 0 or hi < 0) and pq < 0.05
        assert support == (r['protein_measured_subset_support'] == 'True')
        estimate(axd, p, lo, hi, plot_y, PURPLE, support)
        note(axd, d, y, f"{_qfmt(pq)} · {r['P_measured']}/50")
        (rho, eqv) = (float(r['PDO_spearman']), float(r['PDO_q']))
        point(axe, rho, plot_y, 'o', PDO, eqv < 0.05)
        note(axe, e, y, 'q = ' + _qfmt(eqv), bold=eqv < 0.05, color=PDO if eqv < 0.05 else MUTED)
    layer.plot([2.1, 172], [114.2, 114.2], lw=0.5, c=FRAME)
    txt(2.3, 118.2, 'Keys: filled / open points = gate met / not met (panel e: 7-program BH q < 0.05); horizontal bars = 95% bootstrap CIs.', size=6.8, color=MUTED)
    txt(2.3, 123.6, 'Panel b: 80% guide. Panel c: circle = reference Q (frozen Table S4 CI); open diamond = linear-depth Q; specs/9 · PC8 (+ / −).', size=6.8, color=MUTED)
    txt(2.3, 129.0, 'Panel d: C07 aggregate q = 0.725 reflects 24+ vs 16− measured members (DCI = 0.012); Zhao 57-pair 21-gene d_z ρ = 0.796.', size=6.8, color=MUTED)
    txt(2.3, 134.4, 'GSE203067 (2/4 evaluable): mean and majority claims not evaluable. No global rank is formed (EDSC bridge in Table S5).', size=6.8, color=MUTED)
    fig.canvas.draw()
    inv = layer.transData.inverted()
    rendered = []
    failures = []
    for (item, (left, right, top, bottom)) in text_checks:
        bb = item.get_window_extent().transformed(inv)
        if not (left - 0.05 <= bb.xmin <= bb.xmax <= right + 0.05 and top <= bb.ymin <= bb.ymax <= bottom):
            failures.append((item.get_text(), bb.bounds, (left, right, top, bottom)))
        for (other, old) in rendered:
            assert bb.xmin >= old.xmax or bb.xmax <= old.xmin or bb.ymin >= old.ymax or (bb.ymax <= old.ymin), (item.get_text(), other)
        rendered.append((item.get_text(), bb))
    assert not failures, failures
    export_figure(fig, MAIN, 'Fig9')
    print('Rendered Fig9 cleanly with 0 label collisions or overflows.')
