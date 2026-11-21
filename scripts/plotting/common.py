"""Physical-size typography and shared scientific display conventions."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import matplotlib as mpl
from matplotlib import transforms
import numpy as np
import pandas as pd


def mm(value):
    return value / 25.4


WIDTH = mm(174)
INK = "#20272d"
GRAY = "#505c64"
GRID = "#dce3e6"
PAPER = "#f2f5f6"
BLUE = "#00669c"
ORANGE = "#ad4b21"
TEAL = "#147862"
PROGRAMS = tuple(f"RMP_C{i:02d}" for i in range(1, 8))
SHORT = tuple(f"C{i:02d}" for i in range(1, 8))
COLORS = ("#00669c", "#ad4b21", "#147862", "#795191", "#586957", "#5d6473", "#a23e64")


def use_style():
    """Set type sizes in output points, independently of raster DPI."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 8.5,
        "axes.labelcolor": INK,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.titlepad": 5,
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.60,
        "ytick.major.width": 0.60,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.pad": 3.0,
        "ytick.major.pad": 3.0,
        "text.color": INK,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.use14corefonts": False,
        "ps.useafm": False,
        "image.composite_image": False,
    })


def data(directory, name):
    return pd.read_csv(resolve_data(directory / f"{name}.tsv"), sep="\t", low_memory=False)


def panel(ax, letter, title=None, *, x=0, y=1.025):
    letter = letter.lower()
    ax.set_gid(letter)
    ax.text(x, y, letter, transform=ax.transAxes, va="bottom", ha="left",
            fontweight="bold", fontsize=11, clip_on=False)
    if title:
        # Fixed print-unit gap after the letter, even in markedly different panels.
        transform = transforms.offset_copy(ax.transAxes, fig=ax.figure, x=13, units="points")
        ax.text(x, y, title, transform=transform, va="bottom", ha="left",
                fontsize=8.5, clip_on=False)


def alignment(fig, *, axes=None, rows=None, columns=None, exemptions=()):
    fig._nature_alignment = {
        "axes": axes,
        "row_groups": rows,
        "column_groups": columns,
        "exemptions": tuple(exemptions),
    }


def q_text(value):
    if pd.isna(value):
        return "NA"
    return f"{value:.2g}" if value < 0.01 else f"{value:.3f}"


def forest(ax, values, low, high, labels, colors=None, reference=0):
    values, low, high = map(np.asarray, (values, low, high))
    colors = colors if colors is not None else [BLUE] * len(values)
    rows = np.arange(len(values))
    ax.axvline(reference, color=GRAY, linewidth=0.75, linestyle=(0, (3, 3)), zorder=0)
    ax.hlines(rows, low, high, color=colors, linewidth=1.3, zorder=2)
    ax.scatter(values, rows, s=18, c=colors, linewidth=0.4, edgecolors="white", zorder=3)
    ax.set_yticks(rows, labels)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
