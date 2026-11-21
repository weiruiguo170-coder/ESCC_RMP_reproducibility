"""Apply final scientific labels and print layouts to frozen figure inputs."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table
from pathlib import Path
import os, sys, json, hashlib, io, importlib, importlib.util
HERE = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.text import Text
from PIL import Image
sys.path.insert(0, str(ROOT / 'scripts/plotting'))
from common import use_style
FIGURES = ['Fig1', 'Fig2', 'FigS4', 'FigS8', 'FigS10', 'FigS12']
AUDIT = {}

def style_supp():
    use_style()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'escc-supplement-rebuild', 'axes.spines.right': False, 'axes.spines.top': False})

def numerical_artists(fig):
    """Snapshot every data-coordinate line, collection and bar before display edits."""
    items = []

    def take(kind, value):
        a = np.asarray(value)
        if a.size:
            items.append([kind, a.shape, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()])
    for ax in fig.axes:
        for line in ax.lines:
            take('line', line.get_xydata())
        for col in ax.collections:
            if hasattr(col, 'get_offsets'):
                take('offsets', col.get_offsets())
            if hasattr(col, 'get_segments'):
                for segment in col.get_segments():
                    take('segment', segment)
            if hasattr(col, 'get_array') and col.get_array() is not None:
                take('array', col.get_array())
        for patch in ax.patches:
            if isinstance(patch, matplotlib.patches.Rectangle):
                take('rectangle', [patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()])
    return items

def replace_text(fig, old, new, expected=1, fontsize=None):
    matches = [t for t in fig.findobj(Text) if t.get_text() == old]
    if len(matches) != expected:
        raise ValueError(f'Expected {expected} copies of {old!r}; found {len(matches)}')
    for t in matches:
        t.set_text(new)
        if fontsize is not None:
            t.set_fontsize(fontsize)

def add_canvas_margin(fig, *, right_mm=0, top_mm=0):
    """Add blank print area while preserving physical axes and observed point positions."""
    (width, height) = fig.get_size_inches()
    neww = width + right_mm / 25.4
    newh = height + top_mm / 25.4
    positions = [(ax, ax.get_position().bounds) for ax in fig.axes]
    texts = [(t, t.get_position()) for t in fig.texts]
    fig.set_size_inches(neww, newh, forward=False)
    for (ax, (x, y, w, h)) in positions:
        ax.set_position([x * width / neww, y * height / newh, w * width / neww, h * height / newh])
    for (t, (x, y)) in texts:
        t.set_position((x * width / neww, y * height / newh))

def capture(module, function):
    saved = []
    old = getattr(module, 'save', None)
    module.save = lambda fig, name: saved.append(fig)
    try:
        getattr(module, function)()
    finally:
        if old is None:
            delattr(module, 'save')
        else:
            module.save = old
    assert len(saved) == 1
    return saved[0]

def create(name):
    if name == 'Fig1':
        m = importlib.import_module('workflow')
        fig = m.draw(*m.load_data(ROOT / 'data/processed/figures'))
        old = numerical_artists(fig)
        replace_text(fig, '6,801 malignant squamous cells\n6 analytical specimen aliases', '6,801 malignant squamous cells\n6 patients (P1–P6)')
        replace_text(fig, 'Cell-weighted RAW and equal-unit EQUAL\nSource-level concentration within each cohort\nGSE160269 patient IDs and GSE188900 aliases', 'Cell-weighted RAW and equal-unit EQUAL\nPatient-level concentration within each cohort\nGSE160269 patients and GSE188900 patients')
        replace_text(fig, '80 pairs across 16,143 shared genes\nComposite similarity, margins and adjusted q', '80 candidate component pairs\n16,143 shared genes · 7 reciprocal best matches\nComposite similarity, margins and adjusted q', fontsize=7.25)
        replace_text(fig, 'Tumor-associated epithelium\nHistorical: 489 cells, K = 5\nCNV candidate sets: 992 / 625\nMalignancy unconfirmed\nExcluded from primary\ndiscovery and Q', 'Tumor-associated epithelium\nHistorical: 489 cells, K = 5\nRepaired: 992; constrained: 625\nMalignancy unconfirmed\nExcluded from primary\ndiscovery and Q')
        replace_text(fig, '7 recurrent programs (C01–C07)   ·   Frozen Top50 and nested Top100 signatures', '7 reciprocal best matches (C01–C07) · Frozen Top50 and nested Top100 signatures')
    elif name == 'Fig2':
        use_style()
        m = importlib.import_module('discovery')
        fig = m.figure_3(ROOT / 'results/tables')
        old = numerical_artists(fig)
        ax = next((ax for ax in fig.axes if ax.get_gid() == 'c'))
        mapping = {'C03': 'Stress / inflammation', 'C04': 'Cycling / repair (provisional)', 'C05': 'Epithelial inflammation', 'C07': 'IFN / MHC-II'}
        for (code, label) in mapping.items():
            y = next((t.get_position()[1] for t in ax.texts if t.get_text() == code))
            candidates = [t for t in ax.texts if np.isclose(t.get_position()[0], 0.338) and np.isclose(t.get_position()[1], y)]
            assert len(candidates) == 1
            candidates[0].set_text(label)
    elif name == 'FigS4':
        style_supp()
        m = importlib.import_module('evidence')
        fig = m.figure_s4(ROOT / 'results/tables')
        old = numerical_artists(fig)
        fig.canvas.draw()
        original_positions = [ax.transData.transform(c.get_offsets()) for ax in fig.axes for c in ax.collections if hasattr(c, 'get_offsets')]
        add_canvas_margin(fig, top_mm=6)
        legend = fig.legends[0]
        legend.set_bbox_to_anchor((0.125, 0.994), transform=fig.transFigure)
        fig.canvas.draw()
        new_positions = [ax.transData.transform(c.get_offsets()) for ax in fig.axes for c in ax.collections if hasattr(c, 'get_offsets')]
        assert all((np.allclose(a, b, atol=1e-07) for (a, b) in zip(original_positions, new_positions)))
        AUDIT['FigS4_physical_point_positions_unchanged'] = True
    elif name == 'FigS8':
        style_supp()
        m = importlib.import_module('framework_plot')
        fig = capture(m, 'figs8')
        old = numerical_artists(fig)
        replace_text(fig, 'EDSC condition changes: 17 / 280 threshold records; 6 / 126 QC records.', 'EDSC changes vs baseline: 17/280 threshold records; 6/126 QC records.', fontsize=8)
        replace_text(fig, 'a-b quantify evaluability and interpretation, not accuracy. Cross-claim status changes after masking: 0.', 'a–b show evidence availability, not accuracy. Cross-claim status changes after masking: 0.', fontsize=7.2)
        replace_text(fig, 'c uses fixed usages and original sample aliases. d uses 20 technical repeats per condition, not new patients.', 'c: fixed usages, six patient groups (P1–P6). d: 20 technical repeats per condition, not new patients.', fontsize=7.2)
        fig.text(0.105, 0.06, 'Records: 7 programs × 2 sizes × 20 threshold or 9 QC settings; changes include indeterminate outputs.', fontsize=7.2, color=m.GRAY)
        next((t for t in fig.texts if t.get_text().startswith('a–b show evidence availability'))).set_y(0.042)
        next((t for t in fig.texts if t.get_text().startswith('c: fixed usages'))).set_y(0.023)
    elif name == 'FigS10':
        style_supp()
        m = importlib.import_module('schematics')
        fig = capture(m, 'framework')
        old = numerical_artists(fig)
        replace_text(fig, 'Patient IDs or specimen aliases · concentration', 'Patients (P1–P6) · concentration')
        replace_text(fig, '80 pairs · BH80 · alternative margins', '80 candidate pairs · BH80 · alternative margins')
        replace_text(fig, 'b  Decision for each claim', 'b  Generic claim decisions')
        replace_text(fig, 'The bridge is conditional on the frozen signatures, reference RNA setting and six source aliases.', 'The bridge is conditional on the frozen signatures, reference RNA setting and six patient groups.')
        fig.axes[0].text(7, 116, 'Fig. S8 uses four reporting states, retaining Limited support separately.', fontsize=7.3, color='#425A66')
    elif name == 'FigS12':
        style_supp()
        m = importlib.import_module('schematics')
        fig = capture(m, 'contributions')
        old = numerical_artists(fig)
        replace_text(fig, 'b  Equal-source share', 'b  Equal-patient share')
        replace_text(fig, 'Ranges cover the original grouping and 15 single-pair merges of six specimen aliases.', 'Six-patient reference (P1–P6) plus 15 artificial pair-aggregation stress tests.')
        fig.canvas.draw()
        positions = [ax.transData.transform(c.get_offsets()) for ax in fig.axes for c in ax.collections if hasattr(c, 'get_offsets')]
        add_canvas_margin(fig, right_mm=8)
        fig.canvas.draw()
        new_positions = [ax.transData.transform(c.get_offsets()) for ax in fig.axes for c in ax.collections if hasattr(c, 'get_offsets')]
        assert all((np.allclose(a, b, atol=1e-07) for (a, b) in zip(positions, new_positions)))
        AUDIT['FigS12_physical_point_positions_unchanged'] = True
    else:
        raise ValueError(name)
    new = numerical_artists(fig)
    assert old == new, f'{name}: scientific artist coordinates changed'
    AUDIT[name] = {'scientific_artists_unchanged': True, 'snapshot_sha256': hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest(), 'artist_arrays': len(new)}
    return fig
