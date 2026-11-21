"""Draw the analytical workflow from the frozen study records."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

import argparse
from math import cos, pi, sin
from pathlib import Path

from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from figure_layout import (
    ALIAS_UNIT, BLUE, COHORTS, GRID, HERE, INK, MAGENTA, ORANGE, PURPLE, SLATE,
    TEAL, WIDTH_MM, plt, read_tsv, save, style,
)

HEIGHT = 182
NAVY = "#244661"


def load_data(folder):
    cells = read_tsv(folder / "cohort_cell_counts.tsv")
    fits = {r["cohort"]: r for r in read_tsv(folder / "formal_cnmf.tsv")}
    cohorts = {}
    for cohort in COHORTS:
        counts = [int(r["n_cells"]) for r in cells if r["cohort"] == cohort]
        cohorts[cohort] = (len(counts), sum(counts), fits[cohort]["k"])
    return cohorts, {r["field"]: r["value"] for r in read_tsv(folder / "workflow.tsv")}


def icon(ax, x, y, kind, color):
    def line(points, width=.8):
        xx, yy = zip(*[(x + dx, y + dy) for dx, dy in points])
        ax.plot(xx, yy, color=color, lw=width, solid_capstyle="round", zorder=4)

    def dot(dx, dy, radius, filled=False, alpha=1):
        ax.add_patch(Circle((x + dx, y + dy), radius, ec=color,
                            fc=color if filled else "white", lw=.7, alpha=alpha, zorder=4))

    if kind == "cells":
        for dx, dy in ((-1.2, -.85), (1.05, -.7), (0, 1.1)):
            dot(dx, dy, 1.15, True, .28)
            dot(dx, dy, .37, True)
    elif kind == "matrix":
        for row in range(3):
            for col in range(3):
                ax.add_patch(Rectangle((x-1.75+col*1.2, y-1.75+row*1.2), .94, .94,
                                       color=color, lw=0, alpha=(.25, .49, .83)[(row*2+col)%3], zorder=4))
    elif kind == "match":
        dot(-.8, 0, 1.45, True, .30)
        dot(.8, 0, 1.45, True, .59)
    elif kind == "people":
        for dx, dy, radius in ((-1.52, .03, .44), (1.52, .03, .44), (0, -.65, .64)):
            dot(dx, dy-1.05, radius, True)
            line(((dx-.65, dy+1), (dx-.49, dy+.08), (dx, dy-.18),
                  (dx+.49, dy+.08), (dx+.65, dy+1)))
    elif kind == "rna":
        xx = [i / 8-2 for i in range(33)]
        for sign in (-1, 1):
            line([(v, sign*sin(v*pi/2)) for v in xx], .8)
        for dx in (-1.6, -.9, .9, 1.6):
            dy = sin(dx*pi/2) * .72
            line(((dx, -dy), (dx, dy)), .55)
    elif kind == "scope":
        line(((-1.8, 1.75), (1.85, 1.75)))
        line(((-.15, 1.65), (1.25, .62), (1.3, -.38), (.68, -1.15)))
        line(((-1.55, .62), (.33, .62)))
        line(((.68, -1.75), (.14, -2.13), (-1.2, -.45), (-.63, .02), (.68, -1.75)))
    elif kind == "protein":
        points = ((-2, .85), (-1.65, -.9), (-.75, -1.65), (.3, -.85),
                  (-.55, .75), (.2, 1.65), (1.45, 1.35), (2, -.35), (1.33, -1.63))
        line(points)
        for index in (0, 2, 5, 8):
            dot(*points[index], .34, True)
    elif kind == "pdo":
        for offset in (0, .42):
            line([(2.13*cos(i*pi/24), offset+1.19*sin(i*pi/24)) for i in range(49)], .72)
        for dx, dy in ((-.9, -.12), (.28, .35), (.94, -.23)):
            dot(dx, dy, .33, True, .65)
    elif kind == "evidence":
        line(((-1.65, -2.05), (1.43, -2.05), (1.43, 2.05), (-1.65, 2.05), (-1.65, -2.05)))
        for dy in (-1.1, -.25):
            line(((-.98, dy), (.73, dy)), .7)
        line(((-.84, .81), (-.39, 1.25), (.62, .45)), .95)


def draw(cohorts, v):
    style()
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT/25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH_MM), ylim=(HEIGHT, 0))
    ax.set_axis_off()
    labels = []

    def label(x, y, value, size=8, color=INK, weight="normal", align="left", region=None):
        artist = ax.text(x, y, value, color=color, fontsize=size, fontweight=weight,
                         ha=align, va="center", zorder=5, linespacing=1.23)
        labels.append((artist, region or (1, WIDTH_MM-1, 1, HEIGHT-1)))
        return artist

    def box(x, y, w, h, fill, stroke=None, radius=1.75):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                    fc=fill, ec=stroke or fill, lw=.64, zorder=3))

    def route(points, color=NAVY, arrow=False, dashed=False, zorder=2):
        ls = (0, (2.7, 2)) if dashed else "-"
        if not arrow or len(points) > 2:
            xx, yy = zip(*(points[:-1] if arrow else points))
            ax.plot(xx, yy, lw=.83, color=color, linestyle=ls, zorder=zorder)
        if arrow:
            ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", shrinkA=0, shrinkB=0,
                                         mutation_scale=6.8, lw=.83, linestyle=ls, color=color, zorder=zorder))

    def phase(letter, y, title, width=164):
        box(5, y, width, 7.1, "#EAF2F8", radius=1.25)
        box(5, y, 7.2, 7.1, NAVY, radius=1.25)
        label(8.6, y+3.55, letter, 10.6, "white", "bold", "center")
        label(15, y+3.55, title, 10.4, INK, "bold")

    def card(x, y, w, h, title, lines, color, fill, kind=None, size=8.1, body_size=8):
        box(x, y, w, h, fill, color)
        inset = 9 if kind else 3.2
        if kind:
            icon(ax, x+5.3, y+4.8, kind, color)
        region = (x+2, x+w-2, y+.7, y+h-.7)
        label(x+inset, y+4.8, title, size, color, "bold", region=region)
        label(x+3.2, y+9.05, "\n".join(lines), body_size, INK, region=region).set_va("top")

    def strip(x, y, w, title, color, fill):
        box(x, y, w, 7, fill, color)
        label(x+w/2, y+3.5, title, 8.25, color, "bold", "center", (x+2, x+w-2, y+.6, y+6.4))

    phase("a", 3, "Independent discovery and source contribution")
    phase("b", 92, "Parallel assays for multidimensional evidence")
    phase("c", 159, "Evidence synthesis", width=141)

    # Two independent fit inputs. The historical auxiliary cohort is a contextual card.
    blue_fill, orange_fill, gray_fill = "#F0F6FB", "#FCF3EC", "#F2F4F6"
    n, cells, k_a = cohorts[COHORTS[0]]
    card(7, 13.5, 47, 17.9, COHORTS[0], [f"{cells:,} tumor epithelial cells", f"{n} patients"],
         BLUE, blue_fill, "cells", 9)
    n, cells, k_b = cohorts[COHORTS[1]]
    card(59, 13.5, 51, 17.9, COHORTS[1], [f"{cells:,} malignant squamous cells", f"{n} {ALIAS_UNIT}"],
         ORANGE, orange_fill, "cells", 9)
    _, cells, k_aux = cohorts[COHORTS[2]]
    card(115, 13.5, 52, 30, f"{COHORTS[2]} (auxiliary)",
         ["Tumor-associated epithelium", f"Historical: {cells:,} cells, K = {k_aux}",
          f'CNV candidate sets: {v["cnv_whole_genome"]} / {v["cnv_constrained"]}',
          "Malignancy unconfirmed", "Excluded from primary", "discovery and Q"],
         SLATE, gray_fill, "cells", 8.7)
    strip(7, 35, 47, f"Independent cNMF (K = {k_a})", BLUE, blue_fill)
    strip(59, 35, 51, f"Independent cNMF (K = {k_b})", BLUE, blue_fill)
    for x in (30.5, 84.5):
        route([(x, 31.4), (x, 35)], arrow=True)
        route([(x, 42), (x, 46.5)], arrow=True)

    card(7, 46.5, 103, 14, "Cohort-specific fit outputs",
         ["Program spectra and cell usages"], BLUE, blue_fill, "matrix", 8.8)
    route([(34, 60.5), (34, 66.3)], arrow=True)
    route([(93, 60.5), (93, 63.3), (121, 63.3), (121, 66.3)], ORANGE, arrow=True)
    label(38, 63.4, "Program spectra", 7.7, BLUE)
    label(69, 63.4, "Cell usages", 7.7, ORANGE)

    card(7, 66.3, 77, 18.9, "Reciprocal program matching",
         [f'{v["match_pairs"]} pairs across {int(v["shared_genes"]):,} shared genes',
          "Composite similarity, margins and adjusted q"],
         BLUE, blue_fill, "match", 8.8)
    card(89, 66.3, 78, 18.9, "Source contribution assessment",
         ["Cell-weighted RAW and equal-unit EQUAL", "Source-level concentration within each cohort",
          "GSE160269 patient IDs and GSE188900 aliases"],
         ORANGE, orange_fill, "people", 8.8, body_size=7.4)
    route([(106, 85.2), (106, 89.6)], ORANGE, arrow=True, dashed=True)
    label(110, 88.5, f'{v["groupings"]} grouping-sensitivity configurations', 8, ORANGE, "bold")

    # The matching and concentration assessments each inform synthesis.
    route([(7, 74), (2.7, 74), (2.7, 172.9), (7, 172.9)], BLUE, arrow=True)
    route([(167, 76.2), (171.3, 76.2), (171.3, 172.9), (167, 172.9)], ORANGE, arrow=True)
    route([(37, 85.2), (37, 88.5), (4.45, 88.5), (4.45, 105.9), (7, 105.9)], arrow=True)
    strip(7, 102.4, 160,
          f'{v["program_count"]} recurrent programs ({v["program_range"]})   ·   Frozen Top50 and nested Top100 signatures',
          BLUE, blue_fill)

    xs = (7, 39.3, 71.6, 103.9, 136.2)
    center = [x+15.4 for x in xs]
    route([(87, 109.4), (87, 113.3)])
    route([(center[0], 113.3), (center[-1], 113.3)])
    for x in center:
        route([(x, 113.3), (x, 117.2)], arrow=True)

    cards = (
        ("External RNA", ["GSE199654", f'{v["GSE199654_evaluable"]} evaluable',
                         "Within-patient Q, PC8", "GSE203067", f'{v["GSE203067_evaluable"]} evaluable',
                         "Descriptive only"], TEAL, "#EFF8F6", "rna"),
        ("Bulk tissue RNA", ["GSE53625", f'{v["bulk_pairs"]} patient pairs',
                            "Tumor-only context", "Paired RNA shifts"], BLUE, blue_fill, "matrix"),
        ("Pathology / clinical", ["GSE199654 reused", f'{v["pathology_observations"]} patient–state',
                                  "observations", "GSE53625 clinical", "associations and", "adjusted OS models"], BLUE, blue_fill, "scope"),
        ("Tissue proteomics", [f'Liu: {v["liu_pairs"]} pairs', "Tumor–normal",
                                "Paired effects, DCI", f'Zhao: {v["zhao_pairs"]} pairs',
                                "Gene-level", "concordance"], PURPLE, "#F5F1FA", "protein"),
        ("PDO cisplatin", ["GSE269447", f'{v["pdo_auc"]} PDOs with AUC',
                           f'{v["pdo_day15"]} Day-15 models', "Exact 7-program tests",
                           "Exploratory", "association"], MAGENTA, "#FAF0F6", "pdo"),
    )
    for x, (title, lines, color, fill, kind) in zip(xs, cards):
        box(x, 117.2, 30.8, 30.4, fill, color)
        region = (x+1.3, x+29.5, 118, 147)
        icon(ax, x+15.4, 120.65, kind, color)
        label(x+15.4, 125.1, title, 8, color, "bold", "center", region)
        label(x+2.2, 127.95, "\n".join(lines), 7.55, region=region).set_va("top")
    for x in center:
        route([(x, 147.6), (x, 149.1)])
    route([(center[0], 149.1), (center[-1], 149.1)])
    route([(center[4], 149.1), (center[4], 165.5)], zorder=3.3)
    route([(center[4], 165.5), (center[4], 168.6)], arrow=True, zorder=3.3)

    # Dashed arrows identify sensitivity and calibration, away from the evidence bus.
    route([(10.6, 147.6), (10.6, 151.2)], TEAL, arrow=True, dashed=True)
    box(7, 151.2, 139, 6.25, "#EFF8F6", "#A8CDCA")
    label(9.6, 154.325, "RNA measurement & calibration checks", 7.8, TEAL, "bold")
    label(65, 154.325, f'{v["fixed_cell_specs"]} fixed-cell specs · QC · signature size · '
          f'{v["simulated_cohorts"]} simulated cohorts', 7.8, region=(7, 144.1, 151.3, 157.3))

    box(7, 168.6, 160, 11.6, "#F2EDF9", PURPLE)
    icon(ax, 13, 174.4, "evidence", PURPLE)
    label(19, 172.25, f'Claim-specific evidence profiles for {v["program_range"]}', 9.8, PURPLE, "bold")
    label(19, 176.65, "No single reliability ranking · EDSC three-category audit: Supplementary Table S5", 8)

    # Validate all labels against their allocated regions and one another.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for artist, (left, right, top, bottom) in labels:
        bbox = artist.get_window_extent(renderer)
        corners = ax.transData.inverted().transform(bbox.get_points())
        if not (left <= corners[0][0] and corners[1][0] <= right
                and top <= corners[1][1] and corners[0][1] <= bottom):
            raise ValueError(f"Label outside its region: {artist.get_text()!r}, {corners}")
        for other, existing in bounds:
            if bbox.overlaps(existing):
                raise ValueError(f"Overlapping workflow labels: {artist.get_text()!r}, {other!r}")
        bounds.append((artist.get_text(), bbox))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/'data/processed/figures')
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    save(draw(*load_data(args.data_dir)), args.output_dir, "FIG_Figure1_FINAL")


if __name__ == "__main__":
    main()
