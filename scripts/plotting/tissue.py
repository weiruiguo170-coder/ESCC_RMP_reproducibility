"""External RNA and tissue evidence for the seven frozen ESCC programs."""

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, StrMethodFormatter
import numpy as np

from common import (
    BLUE, COLORS, GRAY, GRID, INK, ORANGE, PAPER, PROGRAMS, SHORT, WIDTH,
    alignment, data, forest, mm, panel, q_text,
)


def cohort_rows(frame, key, signature_field, signature):
    return frame.loc[frame[signature_field].eq(signature)].set_index(key).loc[list(PROGRAMS)]


def rows(ax, count, labels=None):
    positions = np.arange(count)
    ax.set(yticks=positions, yticklabels=labels if labels is not None else [""] * count,
           ylim=(count - .48, -.52))
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.45)
    return positions


def zero(ax, horizontal=False):
    kwargs = {"color": GRAY, "linestyle": (0, (2.1, 2.1)), "linewidth": .55, "zorder": 1}
    (ax.axhline if horizontal else ax.axvline)(0, **kwargs)


def figure_4(src):
    """Expression detectability and coordination are separate patient-level estimands."""
    expr = data(src, "S4_original_expression_summary")
    coord = data(src, "S4_Q_summary")
    fig = plt.figure(figsize=(WIDTH, mm(116)))
    ax_a = fig.add_axes([.115, .58, .852, .337])
    ax_b = fig.add_axes([.115, .115, .383, .335])
    ax_c = fig.add_axes([.584, .115, .383, .335])

    panel(ax_a, "A", "Patient expression scores")
    zero(ax_a, horizontal=True)
    x = np.arange(7)
    for cohort, offset, color, marker, n in (
        ("GSE199654", -.14, BLUE, "o", 14),
        ("GSE203067", .14, ORANGE, "D", 4),
    ):
        estimates = cohort_rows(expr[expr.cohort == cohort], "rmp_id", "signature_type", "TOP50")
        assert estimates.patients_evaluable.eq(n).all()
        ax_a.vlines(x + offset, estimates.median_patient_z_ci_low,
                    estimates.median_patient_z_ci_high, color=color, linewidth=.85, zorder=2)
        ax_a.scatter(x + offset, estimates.median_patient_z, color=color, marker=marker,
                     s=18, edgecolors=PAPER, linewidth=.35, label=f"{cohort} (n={n})", zorder=3)
    ax_a.set(xticks=x, xticklabels=SHORT, xlim=(-.45, 6.45), ylabel="Median matched-null z")
    ax_a.yaxis.set_major_locator(FixedLocator([-4, 0, 4, 8]))
    ax_a.grid(axis="y", color=GRID, linewidth=.45)
    ax_a.set_axisbelow(True)
    ax_a.legend(loc="lower right", bbox_to_anchor=(1., 1.013), ncol=2, frameon=False,
                handletextpad=.35, columnspacing=1.25, fontsize=8, borderaxespad=0)

    for ax, cohort, letter, evaluable, original, color in (
        (ax_b, "GSE199654", "B", 10, 14, BLUE),
        (ax_c, "GSE203067", "C", 2, 4, ORANGE),
    ):
        estimates = cohort_rows(coord[coord.cohort == cohort], "rmp_id", "signature", "TOP50")
        assert estimates.n_patients_evaluable.eq(evaluable).all()
        assert estimates.n_patients_original.eq(original).all()
        descriptor = f"{cohort} ({evaluable}/{original} evaluable)"
        panel(ax, letter, descriptor)
        forest(ax, estimates.mean_patient_effect, estimates.CI_low, estimates.CI_high, SHORT,
               [color] * len(PROGRAMS))
        ax.set(xlim=(-.065, .305), xticks=[-.05, 0, .10, .20, .30],
               xlabel="Mean RNA coordination excess")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
        if letter == "C":
            ax.text(1., 1.026, "Descriptive", transform=ax.transAxes, va="bottom",
                    ha="right", color=GRAY, fontsize=8)
    alignment(fig, axes=[ax_a, ax_b, ax_c], rows=[["b", "c"]])
    return fig


CONTEXT_LABEL = {
    "IMMUNE_TNK": "T/NK",
    "INTERFERON_CONTEXT": "IFN",
    "EPITHELIAL": "Generic epithelial",
    "DIFFERENTIATION_KERATINIZATION": "Diff. / keratinization",
    "FIBROBLAST_ECM": "Fibroblast / ECM",
}


def figure_5(src):
    """Pooled and subseries associations plus all paired C01 observations."""
    context = data(src, "S6_bulk_context")
    context = context.loc[context.signature_size.eq("TOP50")].sort_values(
        ["RMP", "context"], kind="stable",
    ).reset_index(drop=True)
    scores = data(src, "S6_bulk_patient_scores")
    assert len(context) == 10 and context.patients.eq(179).all()
    assert len(scores) == 179 and scores.patient_id.nunique() == 179
    assert scores.subseries.value_counts().to_dict() == {"GSE53624": 119, "GSE53622": 60}
    fig = plt.figure(figsize=(WIDTH, mm(140)))
    ax_a = fig.add_axes([.254, .466, .281, .430])
    ax_b = fig.add_axes([.670, .466, .281, .430])
    ax_c = fig.add_axes([.254, .108, .697, .230])

    panel(ax_a, "A", "Tumor-only estimates (n=179)")
    panel(ax_b, "B", "Subseries estimates")
    labels = [f"{row.RMP[-3:]}  {CONTEXT_LABEL[row.context]}" for row in context.itertuples()]
    foreground = [COLORS[SHORT.index(row.RMP[-3:])] for row in context.itertuples()]
    y = rows(ax_a, len(context), labels)
    rows(ax_b, len(context))
    for ax in (ax_a, ax_b):
        zero(ax)
        ax.set(xlim=(-.36, .91), xticks=[-.3, 0, .3, .6, .9], xlabel="Spearman ρ")
    for idx, (row, color) in enumerate(zip(context.itertuples(), foreground)):
        focal = row.RMP == "RMP_C02" and row.context == "DIFFERENTIATION_KERATINIZATION"
        if focal:
            for ax in (ax_a, ax_b):
                ax.axhspan(idx - .43, idx + .43, color=GRID, lw=0, zorder=0)
        ax_a.hlines(idx, row.ci_low, row.ci_high, color=color, linewidth=1.10, zorder=2)
        ax_a.scatter(row.spearman_rho, idx, s=23, color=color, edgecolors=PAPER, linewidth=.35, zorder=3)
    ax_a.text(1.235, 1.022, "BH q", transform=ax_a.transAxes, ha="center", va="bottom",
              fontsize=8, color=INK)
    for idx, value in enumerate(context.BH_FDR):
        ax_a.text(1.235, idx, q_text(value), transform=ax_a.get_yaxis_transform(), ha="center",
                  va="center", fontsize=8, color=GRAY)
    for cohort, offset, color, marker, n in (
        ("GSE53622", -.15, BLUE, "o", 60), ("GSE53624", .15, ORANGE, "D", 119),
    ):
        ax_b.scatter(context[f"{cohort}_rho"], y + offset, marker=marker, s=17, color=color,
                     edgecolors=PAPER, linewidth=.25, label=f"{cohort} (n={n})", zorder=3)
    ax_b.legend(loc="lower left", bbox_to_anchor=(0, 1.075), frameon=False,
                fontsize=8, handletextpad=.25, labelspacing=.28, borderaxespad=0)

    panel(ax_c, "C", "Paired C01 expression (179 patients)")
    endpoints = scores[["RMP_C01_NORMAL", "RMP_C01_TUMOR"]].to_numpy()
    ax_c.plot([0, 1], endpoints.T, color=BLUE, alpha=.16, linewidth=.5, zorder=2)
    ax_c.scatter([0, 1], np.median(endpoints, axis=0), s=35, marker="D", facecolor=COLORS[0],
                 edgecolors=PAPER, linewidth=.55, zorder=3)
    ax_c.set(xlim=(-.08, 1.08), xticks=[0, 1], xticklabels=["Adjacent", "Tumor"],
             ylabel="Top50 expression score")
    ax_c.grid(axis="y", color=GRID, linewidth=.45)
    ax_c.set_axisbelow(True)
    ax_c.text(1., 1.027, f"Median paired Δ = {scores.RMP_C01_DELTA.median():.3f}",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=8, color=GRAY)
    alignment(fig, axes=[ax_a, ax_b, ax_c], rows=[["a", "b"]])
    return fig


def plot_states(ax, state, rmp, letter, x_offsets, y_ticks, show_y):
    selected = state.loc[state.rmp_id.eq(rmp)].copy()
    assert selected.groupby("ordered_state").size().to_list() == [9, 11, 14]
    assert selected.patient_id.nunique() == 14
    selected["plot_x"] = selected.ordered_state + selected.patient_id.map(x_offsets)
    for _, observations in selected.groupby("patient_id"):
        if len(observations) > 1:
            observations = observations.sort_values("ordered_state")
            ax.plot(observations.plot_x, observations.patient_state_score,
                    color=GRAY, alpha=.43, linewidth=.47, zorder=1)
    color = COLORS[SHORT.index(rmp[-3:])]
    ax.scatter(selected.plot_x, selected.patient_state_score, s=18, color=color,
               edgecolors=PAPER, linewidth=.32, zorder=2)
    panel(ax, letter, rmp[-3:])
    ax.set(xlim=(-.25, 2.25), xticks=[0, 1, 2],
           xticklabels=["Non-tumor\n(n=9)", "Dysplasia\n(n=11)", "Tumor\n(n=14)"],
           ylim=(y_ticks[0] - .025, y_ticks[-1] + .025), yticks=y_ticks)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    ax.grid(axis="y", color=GRID, linewidth=.45)
    ax.set_axisbelow(True)
    if show_y:
        ax.set_ylabel("Patient–state Top50 score")
    else:
        ax.tick_params(axis="y", labelleft=False)


def figure_6(src):
    """Show both multiple-testing families and keep C04 estimands separate."""
    state = data(src, "S6_patient_state_scores").query("signature_type == 'TOP50'")
    models = cohort_rows(data(src, "S6_ordered_state_models"), "rmp_id", "signature", "TOP50")
    contrasts = data(src, "S6_paired_state_contrasts")
    contrasts = contrasts.loc[
        contrasts.rmp_id.eq("RMP_C04") & contrasts.signature_type.eq("TOP50")
    ].set_index("transition").loc[
        ["DYSPLASIA_MINUS_NON_TUMOR", "TUMOR_MINUS_DYSPLASIA", "TUMOR_MINUS_NON_TUMOR"]
    ]
    assert contrasts.paired_patient_count.to_list() == [7, 11, 9]
    assert models.n_patients.eq(14).all() and models.n_patient_states.eq(34).all()
    fig = plt.figure(figsize=(WIDTH, mm(170)))
    ax_a = fig.add_axes([.138, .707, .578, .223])
    trajectories = [
        fig.add_axes([left, .392, .254, .213]) for left in (.138, .421, .704)
    ]
    ax_e = fig.add_axes([.395, .098, .562, .185])

    panel(ax_a, "A", "Ordered-state slopes (14 patients)")
    forest(ax_a, models.slope, models.CI_low, models.CI_high, SHORT, COLORS)
    ax_a.set(xlim=(-.060, .072), xticks=[-.06, -.03, 0, .03, .06],
             xlabel="Score change per category step (95% model CI)")
    ax_a.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    for label, location, column in (
        ("Historical q", 1.14, "BH_FDR"), ("All-7 q", 1.34, "BH_FDR_all_seven_sensitivity"),
    ):
        ax_a.text(location, 1.025, label, transform=ax_a.transAxes, color=INK, fontsize=8,
                  ha="center", va="bottom")
        for idx, value in enumerate(models[column]):
            ax_a.text(location, idx, q_text(value), transform=ax_a.get_yaxis_transform(),
                      color=GRAY, fontsize=8, ha="center", va="center")
    rmp_ids = ("RMP_C02", "RMP_C04", "RMP_C06")
    y_range = state.loc[state.rmp_id.isin(rmp_ids), "patient_state_score"]
    lower = np.floor(y_range.min() / .1) * .1
    upper = np.ceil(y_range.max() / .1) * .1
    y_ticks = np.round(np.arange(lower, upper + .001, .1), 2)
    patient_ids = sorted(state.patient_id.unique())
    x_offsets = dict(zip(patient_ids, np.linspace(-.13, .13, len(patient_ids))))
    for idx, (ax, rmp, letter) in enumerate(zip(trajectories, rmp_ids, "BCD")):
        plot_states(ax, state, rmp, letter, x_offsets, y_ticks, idx == 0)
    row = models.loc["RMP_C04"]
    assert np.isclose(row.slope, .0245695278, rtol=0, atol=1e-9)
    assert np.allclose(contrasts.median_paired_delta, [.053940177, -.000325322, .062131524])
    panel(ax_e, "E", "C04 slope and paired state contrasts")
    labels = ["Ordered slope (n=14)",
              "Dysplasia − non-tumor (n=7)",
              "Tumor − dysplasia (n=11)",
              "Tumor − non-tumor (n=9)"]
    low = [row.CI_low, *contrasts.bootstrap_CI_low]
    high = [row.CI_high, *contrasts.bootstrap_CI_high]
    estimates = [row.slope, *contrasts.median_paired_delta]
    plot_rows = rows(ax_e, len(labels), labels)
    zero(ax_e)
    ax_e.axhline(.5, color=GRID, linewidth=.70)
    for idx, (estimate, lo, hi) in enumerate(zip(estimates, low, high)):
        color = BLUE if idx == 0 else ORANGE
        ax_e.hlines(idx, lo, hi, color=color, linewidth=1.15, zorder=2)
        ax_e.scatter(estimate, plot_rows[idx], c=color, marker="s" if idx == 0 else "o",
                     s=20, edgecolors=PAPER, linewidth=.35, zorder=3)
    ax_e.set(xlim=(-.075, .139), xticks=[-.06, -.03, 0, .03, .06, .09, .12],
             xlabel="Slope (square) or paired median difference (circle)")
    ax_e.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
    alignment(fig, axes=[ax_a, *trajectories, ax_e], rows=[["b", "c", "d"]])
    return fig


def strip_offsets(values, resolution=.040):
    """Place equal-bin points symmetrically, with no adjustment to their x values."""
    values = np.asarray(values)
    offsets = np.empty_like(values)
    bins = np.floor(values / resolution).astype(int)
    for bin_id in np.unique(bins):
        positions = np.flatnonzero(bins == bin_id)
        span = min(.255, .055 * (len(positions) - 1))
        offsets[positions] = np.linspace(-span, span, len(positions))
    return offsets


def figure_7(src):
    """Signed differences, mapping coverage and distinct matched-null q values."""
    contrast = data(src, "S7_proteomic_contrasts")
    protein = cohort_rows(contrast[contrast.framework.eq("raw_paired_difference")],
                          "RMP", "signature", "Top50")
    paired = data(src, "S7_paired_proteomic_scores")
    paired = paired.loc[paired.signature.eq("Top50")]
    assert protein.N_pairs.eq(124).all()
    fig = plt.figure(figsize=(WIDTH, mm(109)))
    ax_a = fig.add_axes([.130, .508, .414, .400])
    ax_b = fig.add_axes([.606, .508, .251, .400])
    ax_c = fig.add_axes([.130, .113, .821, .246])

    panel(ax_a, "A", "Standardized paired effects")
    forest(ax_a, protein.Cohens_dz, protein.dz_CI_low, protein.dz_CI_high, SHORT, COLORS)
    ax_a.set(xlim=(-2.85, 2.18), xticks=[-2, -1, 0, 1, 2], xlabel="Paired Cohen’s dz (bootstrap 95% CI)")
    panel(ax_b, "B", "Coverage and matched null")
    positions = rows(ax_b, 7)
    ax_b.barh(positions, protein.mapped_genes, height=.35, color=COLORS, alpha=.72, zorder=2)
    ax_b.set(xlim=(0, 53), xticks=[0, 10, 20, 30, 40, 50], xlabel="Measured Top50 genes")
    ax_b.text(1.26, 1.024, "Null q", transform=ax_b.transAxes, ha="center", va="bottom",
              fontsize=8, color=INK)
    for idx, q in enumerate(protein.matched_null_FDR):
        ax_b.text(1.26, idx, q_text(q), transform=ax_b.get_yaxis_transform(), ha="center",
                  va="center", fontsize=8, color=GRAY)

    selected = ("RMP_C01", "RMP_C02", "RMP_C07")
    panel(ax_c, "C", "Observed paired differences (124 patients)")
    rows(ax_c, 3, [name[-3:] for name in selected])
    zero(ax_c)
    median_rule = Line2D([], [], marker="D", markerfacecolor=INK, markeredgecolor=PAPER,
                        markeredgewidth=.4, markersize=5, linestyle="none", label="Median")
    for idx, rmp in enumerate(selected):
        observations = paired.loc[paired.RMP.eq(rmp)].sort_values("patient_id")
        values = observations.raw_paired_delta.to_numpy()
        assert len(values) == 124 and observations.patient_id.nunique() == 124
        color = COLORS[SHORT.index(rmp[-3:])]
        ax_c.scatter(values, idx + strip_offsets(values), color=color, alpha=.38, s=9,
                     linewidth=0, zorder=2)
        ax_c.scatter(np.median(values), idx, marker="D", color=INK, s=29,
                     edgecolor=PAPER, linewidth=.55, zorder=3)
    ax_c.set(xlabel="Patient tumor − adjacent abundance difference (log2 scale)")
    ax_c.legend(handles=[median_rule], loc="lower right", bbox_to_anchor=(1, 1.025),
                frameon=False, fontsize=8, handletextpad=.25, borderaxespad=0)
    alignment(fig, axes=[ax_a, ax_b, ax_c], rows=[["a", "b"]])
    return fig
