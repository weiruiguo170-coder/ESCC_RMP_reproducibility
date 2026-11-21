"""Shared publication settings and deterministic exports for Figures 1 and 2."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

import csv
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
WIDTH_MM = 174
INK = "#172C3F"
BLUE = "#145F90"
ORANGE = "#A1481C"
TEAL = "#126C68"
PURPLE = "#594087"
MAGENTA = "#8C3269"
SLATE = "#526171"
GRID = "#D7E0E7"
COHORTS = ("GSE160269", "GSE188900", "GSE196756")
PROGRAMS = tuple(f"C{i:02}" for i in range(1, 8))
ALIAS_UNIT = "analytical specimen aliases"


def read_tsv(path):
    with resolve_data(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_tsv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": SLATE,
        "axes.linewidth": 0.65,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "svg.fonttype": "none",
        "svg.hashsalt": "FIG-final-20261119",
        "pdf.fonttype": 42,
    })


def save(fig, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.svg", metadata={"Date": None})
    fig.savefig(output / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    for extension, dpi in (("png", 600), ("tiff", 1200)):
        path = output / f"{stem}.{extension}"
        options = {"compression": "tiff_lzw"} if extension == "tiff" else {}
        with BytesIO() as raster:
            fig.savefig(raster, format=extension, dpi=dpi, facecolor="white",
                        transparent=False, pil_kwargs=options)
            raster.seek(0)
            with Image.open(raster) as rgba:
                rgb = rgba.convert("RGB")
        with rgb:
            rgb.save(path, dpi=(dpi, dpi), **options)
    plt.close(fig)
