"""Figures for organoid response, evidence profiles and tissue-protein sensitivity."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


from itertools import product
from string import ascii_uppercase

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np

from common import (
    BLUE, COLORS, GRAY, GRID, INK, ORANGE, PROGRAMS, SHORT, TEAL, WIDTH,
    alignment, data, mm, panel,
)


def _signed(number, places=3, positive=False):
    sign = "+" if positive else ""
    return f"{number:{sign}.{places}f}".replace("-", "−")


def _probability(number):
    if number < .001:
        return f"{number:.6f}".rstrip("0")
    return f"{number:.4g}"


def _pdo_inputs(src, size, *, wild=False):
    scores = data(src, "S8_PDO_scores_endpoints")
    manifest = data(src, "S8_PDO_exact_results")
    subset = "NRF2_Wild8_author_AUC" if wild else "All10_author_AUC"
    exact = manifest.loc[
        manifest["subset"].eq(subset)
        & manifest["signature"].eq(size)
        & manifest["program_or_control"].isin(PROGRAMS)
    ].set_index("program_or_control").loc[list(PROGRAMS)]
    assert exact.index.is_unique and exact["n"].eq(8 if wild else 10).all()
    assert scores["PDO_ID"].is_unique and len(scores) == 10
    assert scores["NRF2_mutation"].value_counts().to_dict() == {"Wild": 8, "Mutant": 2}
    assert scores["author_rank"].eq(scores["linear_rank"]).all()
    if wild:
        scores = scores.loc[scores["NRF2_mutation"].eq("Wild")]
    assert len(scores) == exact["n"].iloc[0]
    assert scores["author_AUC"].notna().all()
    return scores, exact


def _scatter_coordinates(ax, scores, field, *, wild):
    assert scores[field].notna().all()
    x = scores[field].to_numpy()
    y = scores["author_AUC"].to_numpy()
    margin = (x.max() - x.min()) * .17
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim((-380, 6400) if wild else (-550, 13950))
    ax.set_yticks([0, 2000, 4000, 6000] if wild else [0, 4000, 8000, 12000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}".rstrip("0").rstrip(".")))
    status = scores["NRF2_mutation"].to_numpy()
    for category, marker, color in (("Wild", "o", BLUE), ("Mutant", "D", ORANGE)):
        selected = status == category
        if np.any(selected):
            ax.scatter(x[selected], y[selected], s=24, marker=marker, color=color,
                       edgecolors="white", linewidths=.55, zorder=3)
    # Plot positions are the observed values. Only the text labels may move.
    return np.column_stack((x, y))


def _place_pdo_labels(ax, scores, coordinates):
    """Place the actual ID suffixes in display points around the fixed observations."""
    fig = ax.figure
    assert fig.canvas.get_renderer() is not None
    box = ax.get_window_extent()
    factor = 72 / fig.dpi
    points = (ax.transData.transform(coordinates) - [box.x0, box.y0]) * factor
    bounds = np.array([box.width, box.height]) * factor
    ids = scores["PDO_ID"].str.removeprefix("ESCCO").to_numpy()
    statuses = scores["NRF2_mutation"].to_numpy()
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbor_distances = distances.min(axis=1)
    displacements = (0, 7, -7, 12, -12, 17, -17, 22, -22, 28, -28, 35, -35, 43, -43, 52, -52)
    offsets = sorted(product(displacements, displacements),
                     key=lambda delta: (np.hypot(*delta), abs(delta[0])))
    half_height = 4.1
    candidates = {}
    for index, point in enumerate(points):
        half_width = 2.6 * len(ids[index])
        candidates[index] = []
        for offset in offsets:
            anchor = point + offset
            x, y = anchor
            if not (half_width + 1.4 < x < bounds[0] - half_width - 1.4
                    and half_height + 1.4 < y < bounds[1] - half_height - 1.4):
                continue
            if np.any((abs(points[:, 0] - x) < half_width + 3.5)
                      & (abs(points[:, 1] - y) < half_height + 3.5)):
                continue
            distance = np.hypot(*offset)
            needs_leader = distance > 9.5 or neighbor_distances[index] < 11
            new_box = Bbox.from_bounds(x - half_width, y - half_height,
                                       2 * half_width, 2 * half_height)
            leader = None
            if needs_leader:
                direction = np.array(offset, dtype=float) / distance
                label_edge = min((edge / abs(movement) for edge, movement
                                  in zip((half_width, half_height), direction) if movement))
                if distance > label_edge + 3.9:
                    start = point + direction * 2.5
                    end = anchor - direction * (label_edge + 1.2)
                    leader = Path([start, end])
            candidates[index].append((anchor, half_width, new_box, leader))

    def compatible(first, second):
        first_anchor, first_width, first_box, first_line = first
        second_anchor, second_width, second_box, second_line = second
        dx, dy = np.abs(first_anchor - second_anchor)
        if dx < first_width + second_width + 6 and dy < 2 * half_height + 7.5:
            return False
        if first_line is not None and first_line.intersects_bbox(second_box, filled=False):
            return False
        return second_line is None or not second_line.intersects_bbox(first_box, filled=False)

    def choose(options):
        if not options:
            return {}
        index = min(options, key=lambda key: len(options[key]))
        for candidate in options[index]:
            remaining = {
                key: [other for other in valid if compatible(candidate, other)]
                for key, valid in options.items() if key != index
            }
            if not all(remaining.values()):
                continue
            selected = choose(remaining)
            if selected is not None:
                return {index: candidate, **selected}
        return None

    chosen = choose(candidates)
    if chosen is None:
        raise RuntimeError(f"Cannot place all PDO labels inside panel {ax.get_gid()} without overlap")
    for index, (anchor, _, _, leader) in chosen.items():
        color = BLUE if statuses[index] == "Wild" else ORANGE
        if leader is not None:
            line = leader.vertices / bounds
            ax.plot(line[:, 0], line[:, 1], transform=ax.transAxes,
                    color=color, alpha=.48, linewidth=.55, zorder=1)
        ax.text(*(anchor / bounds), ids[index], transform=ax.transAxes,
                color=color, ha="center", va="center", fontsize=8, zorder=4)


def _rho_q_text(stats):
    return f"ρ = {_signed(stats.rho)}, q = {_probability(stats.FDR_seven_exact)}"


def _sensitivity_table(ax, full, wild):
    ax.set_axis_off()
    panel(ax, "H", "C07 sensitivity analyses")
    # Three aligned columns keep sample membership, the effect and its test family explicit.
    headers = [(0.01, "PDO sample"), (.59, "ρ"), (.75, "Exact P"), (.95, "BH q")]
    for at, title in headers:
        ax.text(at, .82, title, ha="left" if at == .01 else "right", va="center",
                fontsize=8, weight="semibold", transform=ax.transAxes)
    for y, name, row in ((.58, "All organoids (n = 10)", full),
                         (.35, "NRF2 wild-type (n = 8)", wild)):
        values = (name, _signed(row.rho, 4), _probability(row.P_exact_two_sided),
                  _probability(row.FDR_seven_exact))
        for (at, _), value in zip(headers, values):
            ax.text(at, y, value, ha="left" if at == .01 else "right", va="center",
                    fontsize=8.5, transform=ax.transAxes)
    for y in (.72, .22):
        ax.plot([.01, .96], [y, y], transform=ax.transAxes, color=GRID, linewidth=.6)
    ax.text(.01, .06, "Author and linear AUC yield identical ranks.", fontsize=8,
            transform=ax.transAxes, va="center", color=GRAY)


def _pdo_grid(src, size):
    scores, exact = _pdo_inputs(src, size)
    _, wild = _pdo_inputs(src, size, wild=True)
    fig = plt.figure(figsize=(WIDTH, mm(142)))
    grid = fig.add_gridspec(3, 3, left=.125, right=.975, bottom=.087, top=.893,
                           wspace=.24, hspace=.64)
    axes = []
    coordinates = []
    for i, code in enumerate(SHORT):
        row, column = divmod(i, 3)
        ax = fig.add_subplot(grid[row, column])
        coordinates.append(_scatter_coordinates(ax, scores, f"{code}_{size}", wild=False))
        axes.append(ax)
        panel(ax, ascii_uppercase[i], code)
        ax.annotate(_rho_q_text(exact.loc[f"RMP_{code}"]), xy=(0, 1),
                    xycoords="axes fraction", xytext=(0, 21), textcoords="offset points",
                    fontsize=7.5, ha="left", va="bottom", color=GRAY)
        ax.tick_params(axis="y", labelleft=column == 0)
    h = fig.add_subplot(grid[2, 1:])
    _sensitivity_table(h, exact.loc["RMP_C07"], wild.loc["RMP_C07"])
    fig.supylabel("Author cisplatin AUC (×10³)", x=.023)
    fig.supxlabel(f"{size} program score", y=.024)
    handles = [
        Line2D([], [], marker="o", color=BLUE, linestyle="none", markersize=5,
               label="NRF2 wild-type"),
        Line2D([], [], marker="D", color=ORANGE, linestyle="none", markersize=5,
               label="Recorded NRF2 mutation"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.125, .99), ncol=2,
               frameon=False, fontsize=8.5, handletextpad=.35, columnspacing=1.5,
               borderaxespad=0)
    fig.canvas.draw()
    for ax, points in zip(axes, coordinates):
        _place_pdo_labels(ax, scores, points)
    alignment(fig, axes=[*axes, h], rows=[["a", "b", "c"], ["d", "e", "f"], ["g", "h"]],
              columns=[["a", "d", "g"], ["b", "e"], ["c", "f"]])
    return fig


def figure_8(src):
    return _pdo_grid(src, "Top50")


def figure_s4(src):
    return _pdo_grid(src, "Top100")


def _rule_marker(ax, x, y, passed):
    ax.scatter(x, y, s=22, color=TEAL if passed else "white", marker="o",
               edgecolors=TEAL if passed else GRAY, linewidths=.85, zorder=4)


def figure_9(src):
    profiles = data(src, "S5_profiles")
    profiles = profiles.loc[profiles["signature_size"].eq(50)].set_index("RMP").loc[list(PROGRAMS)]
    assert profiles.index.is_unique and len(profiles) == 7
    assert profiles["D_pass"].notna().all() and profiles["Q_pass"].notna().all()
    assert profiles["P_pass"].notna().all() and profiles["R_caution"].notna().all()
    assert profiles[["RAW_max_contribution", "EQUAL_max_contribution"]].max(axis=1).ge(.8).eq(
        profiles["R_caution"]
    ).all()
    assert profiles["PDO_n"].eq(10).all() and profiles["protein_pairs"].eq(124).all()
    fig, ax = plt.subplots(figsize=(WIDTH, mm(83)))
    fig.subplots_adjust(left=.025, right=.985, bottom=.19, top=.96)
    ax.set(xlim=(0, 1), ylim=(8.05, -.55))
    ax.set_axis_off()

    columns = (
        (.035, "RMP", "Top50"),
        (.164, "Discovery (D)", "Composite S"),
        (.350, "Contribution (R)", "Raw / equal share"),
        (.538, "RNA coord. (Q)", "Mean excess"),
        (.715, "Protein (P)", "Paired dz"),
        (.902, "PDO response", "Exploratory ρ / q"),
    )
    for x, heading, metric in columns:
        ax.text(x, 0, heading, ha="center", va="center", weight="semibold", fontsize=9)
        ax.text(x, .60, metric, ha="center", va="center", color=GRAY, fontsize=8)
    ax.hlines(.96, .005, .99, color=GRAY, linewidth=.75)
    ax.hlines(np.arange(1.92, 8, 1), .005, .99, color=GRID, linewidth=.6)
    assert not profiles[
        ["matching_composite", "GSE199654_coherence_mean_patient_effect", "protein_dz",
         "PDO_rho", "PDO_exact_q7"]
    ].isna().any().any()
    for i, (rmp, row) in enumerate(profiles.iterrows()):
        y = 1.43 + i
        coord = row.GSE199654_coherence_mean_patient_effect
        ax.text(.035, y, rmp[-3:], ha="center", va="center", fontsize=9, weight="semibold")
        for x, value, state in (
            (.171, _signed(row.matching_composite), row.D_pass),
            (.554, _signed(coord, positive=True), row.Q_pass),
            (.730, _signed(row.protein_dz, 2, positive=True), row.P_pass),
        ):
            _rule_marker(ax, x - .060, y, state)
            ax.text(x, y, value, ha="center", va="center", fontsize=9)
        if row.R_caution:
            ax.scatter(.263, y, color=ORANGE, marker="^", s=29, linewidths=.55, zorder=4)
        ax.text(.360, y, f"{row.RAW_max_contribution:.1%} / {row.EQUAL_max_contribution:.1%}",
                ha="center", va="center", fontsize=8.5)
        ax.text(.902, y, f"{_signed(row.PDO_rho)} / {_probability(row.PDO_exact_q7)}",
                ha="center", va="center", fontsize=8.5)
    handles = [
        Line2D([], [], marker="o", color=TEAL, linestyle="none", markersize=5,
               label="D/Q/P criterion met"),
        Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=GRAY,
               linestyle="none", markersize=5, label="D/Q/P criterion not met"),
        Line2D([], [], marker="^", color=ORANGE, linestyle="none", markersize=5,
               label="R concentration warning"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(.5, .093),
               frameon=False, fontsize=8, handletextpad=.30, columnspacing=1.1, borderaxespad=0)
    fig.text(.5, .035,
             "Raw and equal maxima span the discovery cohorts. Q: GSE199654; protein: 124 pairs; PDO: 10 lines.",
             fontsize=8, color=GRAY, ha="center")
    return fig


def figure_s2(src):
    contrasts = data(src, "S7_proteomic_contrasts")
    effects = contrasts.loc[
        contrasts["signature"].eq("Top100") & contrasts["framework"].eq("raw_paired_difference")
    ].set_index("RMP").loc[list(PROGRAMS)]
    assert effects.index.is_unique and effects["N_pairs"].eq(124).all()
    assert effects["mapped_genes"].between(0, 100).all()
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, mm(79)),
                               gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=.105, right=.975, bottom=.17, top=.87, wspace=.17)
    panel(a, "A", "Top100 paired tissue-protein contrast")
    panel(b, "B")
    positions = np.arange(7)
    a.set(ylim=(6.53, -.53), xlim=(-3, 2.22), yticks=positions, yticklabels=SHORT,
          xticks=[-3, -2, -1, 0, 1, 2], xlabel="Paired Cohen’s dz (95% CI)")
    a.axvline(0, color=GRAY, linestyle="--", linewidth=.7, zorder=0)
    for i, (_, row) in enumerate(effects.iterrows()):
        a.hlines(i, row["dz_CI_low"], row["dz_CI_high"], color=COLORS[i], linewidth=1.15)
        a.scatter(row["Cohens_dz"], i, s=27, marker="o", zorder=4, color=COLORS[i],
                  edgecolors="white", linewidths=.5)
    a.tick_params(axis="y", length=0, pad=6)
    b.set(xlim=(0, 1), ylim=a.get_ylim())
    b.set_axis_off()
    for column, title in ((.28, "Mapped genes"), (.76, "Matched-null q")):
        b.text(column, 1.025, title, ha="center", va="bottom", weight="semibold",
               fontsize=8, transform=b.transAxes)
    b.hlines(-.48, .02, .97, color=GRAY, linewidth=.6)
    for i, (_, row) in enumerate(effects.iterrows()):
        b.text(.28, i, f"{int(row.mapped_genes)}/100", ha="center", va="center", fontsize=9)
        b.text(.76, i, _probability(row.matched_null_FDR), ha="center", va="center", fontsize=9)
    alignment(fig, axes=[a, b], rows=[["a", "b"]])
    return fig


def figure_s3(src):
    genes = data(src, "S7_Liu_gene_effects")
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, mm(100)), sharex=True)
    fig.subplots_adjust(left=.115, right=.97, bottom=.15, top=.80, hspace=.68)
    for axis, letter, code in zip(axes, "ABC", ("C01", "C02", "C07")):
        selected = genes.loc[
            genes["signature"].eq("Top50") & genes["RMP"].eq(f"RMP_{code}") & genes["mapped"].eq("YES")
        ]
        estimates = selected["mean_delta_raw_T_minus_N"].to_numpy()
        assert selected["gene"].is_unique and np.isfinite(estimates).all()
        qualified = selected["within_sig_BH_FDR"].lt(.05).to_numpy()
        positive = estimates > 0
        negative = estimates < 0
        counts = (int(positive.sum()), int(negative.sum()),
                  int((qualified & positive).sum()), int((qualified & negative).sum()))
        if code == "C07":
            assert counts == (24, 16, 19, 15)
        panel(axis, letter, code)
        axis.annotate(
            f"All:  {counts[0]} positive / {counts[1]} negative      "
            f"q < 0.05:  {counts[2]} positive / {counts[3]} negative",
            xy=(1, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
            ha="right", va="bottom", fontsize=8, color=GRAY,
        )
        axis.vlines(estimates, .08, np.where(qualified, .92, .48),
                    color=np.select([positive, negative], [TEAL, ORANGE], default=GRAY),
                    linewidth=1.05)
        axis.axvline(0, color=GRAY, linestyle="--", linewidth=.7, zorder=0)
        axis.set(xlim=(-2.82, 1.42), ylim=(0, 1), yticks=[], xticks=[-2.5, -2, -1.5, -1, -.5, 0, .5, 1])
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="x", labelbottom=letter == "C", bottom=letter == "C")
        if letter != "C":
            axis.spines["bottom"].set_visible(False)
    fig.legend(
        handles=[
            Line2D([], [], color=TEAL, marker="|", linestyle="none", markersize=8, label="Positive"),
            Line2D([], [], color=ORANGE, marker="|", linestyle="none", markersize=8, label="Negative"),
        ],
        frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(.115, .98),
        borderaxespad=0, fontsize=8, handletextpad=.20, columnspacing=.85,
    )
    fig.text(.97, .945, "Long marks: within-signature q < 0.05",
             fontsize=8, color=GRAY, ha="right", va="center")
    axes[-1].set_xlabel("Mean paired tumor-minus-adjacent log2 abundance")
    alignment(fig, axes=list(axes), columns=[["a", "b", "c"]])
    return fig


def figure_s5(src):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, mm(78)), sharey=True, sharex=True)
    fig.subplots_adjust(left=.113, right=.971, bottom=.18, top=.88, wspace=.16)
    labeled_points = []
    for ax, letter, size in zip(axes, "AB", ("Top50", "Top100")):
        scores, exact = _pdo_inputs(src, size, wild=True)
        result = exact.loc["RMP_C07"]
        xy = _scatter_coordinates(ax, scores, f"C07_{size}", wild=True)
        labeled_points.append((ax, scores, xy))
        panel(ax, letter)
        ax.annotate(
            f"ρ = {_signed(result.rho)}    Exact P = {_probability(result.P_exact_two_sided)}"
            f"    q = {_probability(result.FDR_seven_exact)}",
            xy=(1, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
            fontsize=8, ha="right", va="bottom", color=GRAY,
        )
        ax.set_xlabel(f"C07 {size} score")
    global_x = np.concatenate([points[:, 0] for _, _, points in labeled_points])
    margin = .12 * np.ptp(global_x)
    axes[0].set_xlim(global_x.min() - margin, global_x.max() + margin)
    axes[0].set_ylabel("Author cisplatin AUC (×10³)")
    fig.canvas.draw()
    for ax, scores, points in labeled_points:
        _place_pdo_labels(ax, scores, points)
    alignment(fig, axes=list(axes), rows=[["a", "b"]])
    return fig
