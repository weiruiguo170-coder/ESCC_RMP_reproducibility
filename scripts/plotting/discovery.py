"""Discovery design, alias weighting, cross-cohort matching and cNMF diagnostics."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from common import (
    BLUE, GRAY, GRID, INK, ORANGE, PAPER, PROGRAMS, SHORT, TEAL, WIDTH,
    alignment, data, mm, panel,
)


COHORTS = ("GSE160269", "GSE188900", "GSE196756")
COHORT_COLORS = (BLUE, ORANGE, GRAY)
THEMES = {
    "CELL_CYCLE": "Cell cycle / DNA repair",
    "EPITHELIAL_DIFFERENTIATION": "Differentiation / keratinization",
    "ECM_MOTILITY": "ECM / motility",
    "INTERFERON_IMMUNE_RESPONSE": "IFN / inflammation",
}


def figure_1(src):
    units = data(src, "S1_cnmf_input_cell_counts").groupby("cohort").size()
    match_count = len(data(src, "S3_matching_margins"))
    external = data(src, "S4_Q_summary").groupby("cohort").n_patients_original.first()
    bulk_count = len(data(src, "S6_bulk_patient_scores"))
    protein_count = int(data(src, "S7_proteomic_contrasts").N_pairs.iloc[0])
    pdo_count = len(data(src, "S8_PDO_scores_endpoints"))
    assert (int(units.iloc[0]), int(units.iloc[1]), match_count) == (60, 6, 7)
    assert (bulk_count, protein_count, pdo_count) == (179, 124, 10)

    fig = plt.figure(figsize=(WIDTH, mm(83)))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 174), ylim=(0, 83))
    ax.set_axis_off()

    def label(x, y, content, *, size=8.5, bold=False, color=INK, ha="left"):
        ax.text(x, y, content, size=size, color=color, ha=ha, va="center",
                weight="bold" if bold else "normal")

    def arrow(start, stop, style="-"):
        ax.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>",
                                    color=GRAY, linewidth=.8, linestyle=style,
                                    mutation_scale=8, shrinkA=0, shrinkB=0))

    for x, width, name in ((3, 44, "Discovery"), (61, 47, "Matching"), (125, 46, "Evidence domains")):
        label(x, 76, name, size=11, bold=True)
        ax.hlines(72, x, x + width, color=GRID, linewidth=.65)

    ax.add_patch(Rectangle((3, 30), 44, 33, facecolor=PAPER, edgecolor=GRID, linewidth=.65))
    label(7, 57, "Independent cNMF", size=9.5, bold=True)
    label(7, 49.5, "GSE160269", bold=True, color=BLUE)
    label(7, 44.4, f"{units.loc['GSE160269']} source patients")
    label(7, 38.8, "GSE188900", bold=True, color=ORANGE)
    label(7, 33.7, f"{units.loc['GSE188900']} analysis aliases")

    ax.add_patch(Rectangle((61, 30), 47, 33, facecolor="#E8F2F5", edgecolor=BLUE, linewidth=.75))
    label(65, 57, "Cross-cohort matching", size=9.5, bold=True)
    label(65, 48, f"{match_count} reciprocal best matches")
    label(65, 42, "Frozen RNA signatures")
    label(65, 36, "Top50 and Top100")
    arrow((47, 46.5), (61, 46.5))

    ax.add_patch(Rectangle((61, 3.5), 47, 18, facecolor="white", edgecolor=GRID, linewidth=.65))
    label(65, 17, "Auxiliary projection", size=9, bold=True)
    label(65, 11.5, "GSE196756 historical subset")
    label(65, 6.5, "Tumor-associated epithelium")
    arrow((84.5, 30), (84.5, 21.5), "--")

    endpoints = (
        (63, "External epithelial RNA", f"GSE199654: {external.loc['GSE199654']} patients",
         f"GSE203067: {external.loc['GSE203067']} patients"),
        (46.5, "Tissue RNA and pathology", f"Bulk RNA: {bulk_count} patient pairs",
         "Pathology: GSE199654 reuse"),
        (30, "Paired tissue protein", f"TMT: {protein_count} patient pairs",
         "Tumor minus adjacent tissue"),
        (13.5, "Organoid drug response", f"Cisplatin AUC: {pdo_count} PDO lines",
         "Exploratory association"),
    )
    ax.plot([108, 117, 117], [46.5, 46.5, 63], color=GRAY, linewidth=.8)
    ax.plot([117, 117], [13.5, 46.5], color=GRAY, linewidth=.8)
    for y, name, first, second in endpoints:
        arrow((117, y), (123, y))
        ax.vlines(124, y - 6, y + 6, color=TEAL, linewidth=1.35)
        label(127, y + 4, name, size=9.1, bold=True)
        label(127, y - 1.1, first, size=8.1)
        label(127, y - 5.4, second, size=8.1)
    ax.scatter([117, 117], [30, 46.5], s=6.5, c=GRAY, linewidths=0, zorder=4)
    return fig


def figure_2(src):
    metrics = data(src, "S2_formal_cnmf").set_index("cohort")
    cells = data(src, "S1_cnmf_input_cell_counts")
    weight = data(src, "S3_weighting_summary").set_index("rmp_id").loc[list(PROGRAMS)]
    assert cells.groupby("cohort").size().reindex(COHORTS).tolist() == [60, 6, 3]
    fig = plt.figure(figsize=(WIDTH, mm(107)))
    grid = fig.add_gridspec(
        2, 2, left=.091, right=.972, top=.929, bottom=.125,
        height_ratios=(1.05, 1.74), hspace=.46, wspace=.25,
    )
    a, b, c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    panel(a, "A", "Discovery inputs and cNMF fits", x=0, y=1.035)
    a.set(xlim=(0, 1), ylim=(0, 1))
    a.axis("off")
    columns = (.018, .205, .370, .605, .719, .830, .946)
    a.add_patch(Rectangle((0, .052), 1, .20, facecolor=PAPER, edgecolor="none"))
    for j, (x, header) in enumerate(zip(columns, ("Cohort", "Role", "Discovery units", "Cells", "Rank", "Silhouette", "Formal fits"))):
        a.text(x, .775, header, ha="left" if j < 3 else "center", va="center", weight="bold")
    for line in (.68, .052):
        a.axhline(line, linewidth=.55, color=GRID)
    for i, (key, y) in enumerate(zip(COHORTS, (.57, .36, .15))):
        cohort = cells[cells.cohort == key]
        m = metrics.loc[key]
        label = ("patients", "aliases", "patient labels")[i]
        values = (key, "Auxiliary" if i == 2 else "Primary", f"{len(cohort)} {label}",
                  f"{cohort.n_cells.sum():,}", f"K = {m.k}", f"{m.silhouette:.3f}",
                  f"{m.replicates_completed}/{m.replicates_requested}")
        for j, (x, value) in enumerate(zip(columns, values)):
            a.text(x, y, value, ha="left" if j < 3 else "center", va="center",
                   color=COHORT_COLORS[i] if j == 0 else INK,
                   fontweight="bold" if j == 0 else "normal")

    panel(b, "B", "Cell yield by discovery unit", x=0, y=1.12)
    for key, color, marker in zip(COHORTS[:2], COHORT_COLORS[:2], ("o", "s")):
        group = cells[cells.cohort == key].sort_values("n_cells", ascending=False)
        ranks = np.arange(1, len(group) + 1)
        b.plot(ranks, group.cell_fraction * 100, color=color, marker=marker,
               markersize=2.6 if key == COHORTS[0] else 3.4, linewidth=.95, label=key)
    b.set(xscale="log", xlim=(.90, 67), ylim=(0, 72), xlabel="Discovery unit rank",
          ylabel="Share of cohort cells (%)", yticks=[0, 20, 40, 60])
    b.set_xticks([1, 2, 3, 6, 10, 30, 60], [1, 2, 3, 6, 10, 30, 60])
    b.grid(axis="y", color=GRID, linewidth=.45)
    b.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncols=2, frameon=False,
             handlelength=1.3, handletextpad=.35, borderaxespad=0, columnspacing=1)

    panel(c, "C", "GSE188900 weighting sensitivity", x=0, y=1.12)
    y = np.arange(len(weight))
    raw, equal = weight.GSE188900_RAW_max * 100, weight.GSE188900_EQUAL_max * 100
    for i, (raw_value, equal_value) in enumerate(zip(raw, equal)):
        c.plot([raw_value, equal_value], [i - .16, i + .16], color="#A0ABB2", linewidth=1.05)
    c.scatter(raw, y - .16, color=ORANGE, marker="o", s=18, label="Cell-weighted", linewidths=0, zorder=3)
    c.scatter(equal, y + .16, color=BLUE, marker="D", s=17, label="Equal-unit", linewidths=0, zorder=3)
    c.axvline(80, color=GRAY, linestyle=(0, (3, 3)), linewidth=.8, zorder=0)
    c.set(xlim=(0, 102), xticks=(0, 20, 40, 60, 80, 100), ylim=(6.55, -.55),
          xlabel="Maximum contribution (%)")
    c.set_yticks(y, SHORT)
    c.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncols=2, frameon=False,
             handlelength=1.0, handletextpad=.35, borderaxespad=0, columnspacing=1)
    alignment(fig, axes=[a, b, c], rows=[["b", "c"]])
    return fig


def figure_3(src):
    pairs = data(src, "S3_all_80_pairs")
    match = data(src, "S3_matching_margins").set_index("rmp_id").loc[list(PROGRAMS)]
    annotations = data(src, "S2_program_annotation").set_index("rmp_id").loc[list(PROGRAMS)]
    matrix = pairs.pivot(index="program_A", columns="program_B", values="composite_similarity")
    assert matrix.shape == (10, 8) and matrix.notna().all().all()
    assert all(np.isclose(matrix.loc[row.program_A, row.program_B], row.composite_similarity)
               for row in match.itertuples())

    fig = plt.figure(figsize=(WIDTH, mm(140)))
    gs = fig.add_gridspec(2, 2, left=.087, right=.971, top=.92, bottom=.043,
                          height_ratios=(1.65, 1.03), hspace=.35, wspace=.30)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    panel(a, "A", "All 80 pair similarities", x=0, y=1.08)
    palette = LinearSegmentedColormap.from_list("similarity", ("#F5F9FA", "#9EC6D7", "#15567A"))
    norm = Normalize(vmin=0, vmax=.7)
    mesh = a.pcolormesh(np.arange(9), np.arange(11), matrix, cmap=palette, norm=norm,
                        edgecolors="white", linewidth=.44, shading="flat", rasterized=False)
    a.set(xlim=(0, 8), ylim=(10, 0), xlabel="GSE188900 programs", ylabel="GSE160269 programs")
    a.set_xticks(np.arange(8) + .5, matrix.columns.str[-3:])
    a.set_yticks(np.arange(10) + .5, matrix.index.str[-3:])
    a.tick_params(length=0, pad=3)
    for side in a.spines.values():
        side.set_visible(False)
    for rmp, row in match.iterrows():
        x = matrix.columns.get_loc(row.program_B)
        y = matrix.index.get_loc(row.program_A)
        a.add_patch(Rectangle((x + .025, y + .025), .95, .95, fill=False,
                              edgecolor=ORANGE, linewidth=1.03))
        a.text(x + .5, y + .5, rmp[-3:], ha="center", va="center",
               color="white" if row.composite_similarity > .47 else INK,
               size=8, weight="bold")
    bounds = a.get_position().bounds
    cax = fig.add_axes([bounds[0] + bounds[2] + .012, bounds[1] + bounds[3] * .2, .014, bounds[3] * .6])
    bar = fig.colorbar(mesh, cax=cax, orientation="vertical", ticks=(0, .35, .7))
    bar.solids.set_rasterized(False)
    cax.tick_params(length=0, pad=2)
    bar.outline.set_visible(False)

    panel(b, "B", "Selected pairs and alternatives", x=0, y=1.08)
    y = np.arange(len(match))
    best, next_a, next_b = match.composite_similarity, match.second_composite_A, match.second_composite_B
    b.hlines(y, np.minimum(next_a, next_b), best, color="#B5C2C9", linewidth=.85, zorder=1)
    b.scatter(best, y, c=BLUE, s=19, marker="o", linewidths=0, zorder=3)
    b.scatter(next_a, y + .15, c=ORANGE, s=23, marker="^", linewidths=0, zorder=3)
    b.scatter(next_b, y - .15, c=TEAL, s=17, marker="s", linewidths=0, zorder=3)
    b.set(xlim=(0, .73), ylim=(6.62, -.62), xticks=(0, .2, .4, .6), xlabel="Composite similarity")
    b.set_yticks(y, SHORT)
    b.grid(axis="x", color=GRID, linewidth=.45)
    handles = [Line2D([], [], marker=symbol, color=color, ls="none", markersize=4)
               for symbol, color in (("o", BLUE), ("^", ORANGE), ("s", TEAL))]
    b.legend(handles, ("Selected", "Fix GSE160269", "Fix GSE188900"),
             loc="upper left", bbox_to_anchor=(0, -.18), ncols=2, borderaxespad=0,
             handlelength=.8, handletextpad=.28, columnspacing=.9, labelspacing=.35)

    panel(c, "C", "Program annotation and component gene overlap", x=0, y=1.035)
    c.set(xlim=(0, 1), ylim=(0, 1))
    c.axis("off")
    locations = (.02, .147, .338, .667, .855, .947)
    for x, title in zip(locations, ("RMP", "Matched pair", "Functional annotation", "Confidence", "Top50", "Top100")):
        c.text(x, .86, title, ha="left" if x < .8 else "center", va="center", weight="bold")
    c.axhline(.785, color=GRAY, linewidth=.6)
    rows = np.linspace(.727, .087, len(match))
    for y_loc, (rmp, row) in zip(rows, match.iterrows()):
        note = annotations.loc[rmp]
        theme = THEMES[note.primary_theme]
        if rmp.endswith("03"):
            theme += " / stress"
        if rmp.endswith("04"):
            theme += " (provisional)"
        content = (rmp[-3:], f"{row.program_A[-3:]} / {row.program_B[-3:]}",
                   theme, note.annotation_confidence.title(), f"{row.top50_overlap}/50", f"{row.top100_overlap}/100")
        for j, (x, label) in enumerate(zip(locations, content)):
            c.text(x, y_loc, label, ha="center" if j >= 4 else "left", va="center",
                   fontweight="bold" if j == 0 else "normal", color=BLUE if j == 0 else INK)
        c.axhline(y_loc - .056, color=GRID, linewidth=.38)
    alignment(fig, axes=[a, b, c], rows=[["a", "b"]])
    return fig


def figure_s1(src):
    ranks = data(src, "S2_candidate_rank_diagnostics")
    assert len(ranks) == 18 and set(ranks.Cohort) == set(COHORTS)
    fig = plt.figure(figsize=(WIDTH, mm(96)))
    grid = fig.add_gridspec(2, 3, left=.105, right=.978, top=.875, bottom=.19,
                           hspace=.46, wspace=.27)
    axes = grid.subplots(sharex="col")
    for i, (cohort, color) in enumerate(zip(COHORTS, COHORT_COLORS)):
        sub = ranks[ranks.Cohort == cohort].sort_values("K")
        selected = sub[sub.selected == "YES"]
        assert len(sub) == 6 and len(selected) == 1
        chosen = selected.K.iloc[0]
        for j, field in enumerate(("consensus_silhouette", "reconstruction_metric")):
            ax = axes[j, i]
            panel(ax, "ABCDEF"[i + j * 3], x=0, y=1.07)
            if j == 0:
                name = cohort if i < 2 else f"{cohort} (auxiliary)"
                ax.text(.14, 1.07, name, transform=ax.transAxes, ha="left", va="baseline", weight="bold")
            scale = 1e6 if j == 1 else 1
            ax.plot(sub.K, sub[field] / scale, linewidth=1, color=color, marker="o",
                    markersize=3, markerfacecolor="white", markeredgewidth=.75, zorder=2)
            ax.axvline(chosen, color=GRAY, linewidth=.75, linestyle=(0, (3, 3)), zorder=0)
            ax.scatter([chosen], selected[field] / scale, color=color, s=18, linewidths=0, zorder=3)
            ax.set_xticks(sub.K)
            ax.set_xlim(sub.K.min() - .45, sub.K.max() + .45)
            if j == 0:
                ax.set(ylim=(.65, 1.04), yticks=(.7, .8, .9, 1.0))
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.grid(axis="y", color=GRID, linewidth=.42)
    axes[0, 0].set_ylabel("Consensus silhouette")
    axes[1, 0].set_ylabel("Frobenius SSE (×10⁶)")
    fig.text(.545, .091, "Candidate rank K", ha="center", va="center")
    fig.text(.545, .033, "Dashed line and filled marker: selected K", ha="center", va="center", color=GRAY)
    alignment(fig, axes=list(axes.ravel()), rows=[["a", "b", "c"], ["d", "e", "f"]],
              columns=[["a", "d"], ["b", "e"], ["c", "f"]])
    return fig
