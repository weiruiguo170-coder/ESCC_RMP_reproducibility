from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import hashlib
import json
import math
import os
import platform
import re
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, wilcoxon


PROJECT_ROOT = INPUT_ROOT
OUT = PROJECT_ROOT / "results" / "proteomics"
DATA_XLSX = PROJECT_ROOT / "data" / "raw" / "proteomics" / "41467_2021_25202_MOESM3_ESM.xlsx"
DATA_TSV = PROJECT_ROOT / "data" / "interim" / "processed_protein_matrix_author_supdata1.tsv"
SIG_DIR = PROJECT_ROOT / "signatures"
EVIDENCE_FILE = PROJECT_ROOT / "supplementary" / "results" / "integrated_evidence_matrix.tsv"
PHASE_F_EFFECTS = PROJECT_ROOT / "supplementary" / "results" / "GSE53625_paired_tumor_normal_RMP_effects.tsv"

RMPS = [f"RMP_C{i:02d}" for i in range(1, 8)]
DISPLAY_ORDER = ["RMP_C07", "RMP_C02", "RMP_C01", "RMP_C05", "RMP_C04", "RMP_C06", "RMP_C03"]
HIERARCHY = {
    "RMP_C01": "PRIMARY", "RMP_C02": "PRIMARY", "RMP_C07": "PRIMARY",
    "RMP_C05": "SECONDARY", "RMP_C04": "SECONDARY",
    "RMP_C03": "EXPLORATORY", "RMP_C06": "EXPLORATORY",
}
SEED = 20260828
BOOTSTRAP_REPLICATES = 5000
RANDOM_SET_REPLICATES = 10000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def read_signatures():
    signatures = {50: {}, 100: {}}
    rows = []
    for size in (50, 100):
        for rmp in RMPS:
            path = SIG_DIR / f"{rmp}_signature_top{size}.txt"
            genes = [x.strip().upper() for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
            if len(genes) != size or len(set(genes)) != size:
                raise ValueError(f"Frozen signature integrity failure: {path} has {len(genes)} rows and {len(set(genes))} unique genes")
            signatures[size][rmp] = genes
            rows.append({
                "RMP": rmp,
                "hierarchy": HIERARCHY[rmp],
                "signature": f"Top{size}",
                "frozen_gene_count": len(genes),
                "unique_gene_count": len(set(genes)),
        "source_file": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(path),
                "status": "PASS",
            })
    return signatures, pd.DataFrame(rows)


def read_proteome():
    # This TSV is a lossless tabular extraction of the author-provided Excel
    # sheet, created because the project analysis environment and plotting
    # environment provide complementary dependencies.
    raw = pd.read_csv(resolve_data(DATA_TSV), sep="\t")
    sample_cols = [c for c in raw.columns if re.fullmatch(r"\d+[NT]", str(c))]
    if len(sample_cols) != 248:
        raise ValueError(f"Expected 248 protein sample columns; found {len(sample_cols)}")
    patient_ids = sorted({int(str(c)[:-1]) for c in sample_cols})
    if patient_ids != list(range(1, 125)):
        raise ValueError("Patient IDs are not the complete 1-124 set")
    for patient in patient_ids:
        if f"{patient}N" not in sample_cols or f"{patient}T" not in sample_cols:
            raise ValueError(f"Incomplete pair for patient {patient}")

    raw["Gene symbol"] = raw["Gene symbol"].astype("string").str.strip().str.upper()
    values = raw[sample_cols].apply(pd.to_numeric, errors="coerce")
    source = pd.concat([raw[["UniProt ID", "Gene symbol"]], values], axis=1)
    valid = source["Gene symbol"].notna() & (source["Gene symbol"] != "")
    source = source.loc[valid].copy()

    # Frozen mapping rule: when multiple protein entries map to one gene, take the
    # per-sample median. No entry is selected by effect size or outcome.
    gene_matrix = source.groupby("Gene symbol", sort=True)[sample_cols].median()
    gene_matrix = gene_matrix.apply(pd.to_numeric, errors="coerce")
    means = gene_matrix.mean(axis=1, skipna=True)
    sds = gene_matrix.std(axis=1, skipna=True, ddof=1)
    valid_sd = sds.notna() & (sds > 0)
    gene_matrix = gene_matrix.loc[valid_sd]
    z = gene_matrix.sub(means.loc[valid_sd], axis=0).div(sds.loc[valid_sd], axis=0)

    sample_rows = []
    for c in sample_cols:
        sample_rows.append({
            "sample_id": c,
            "patient_id": int(c[:-1]),
            "tissue": "Adjacent normal" if c.endswith("N") else "Tumor",
            "source_missing_proteins": int(values[c].isna().sum()),
            "source_detected_proteins": int(values[c].notna().sum()),
        })
    sample_audit = pd.DataFrame(sample_rows)

    duplicate_audit = (source.groupby("Gene symbol").agg(
        protein_entries=("UniProt ID", "size"),
        uniprot_ids=("UniProt ID", lambda x: ";".join(sorted(map(str, x))))
    ).reset_index())
    duplicate_audit = duplicate_audit[duplicate_audit["protein_entries"] > 1].copy()
    return raw, source, gene_matrix, z, sample_cols, patient_ids, sample_audit, duplicate_audit


def mapping_tables(signatures, source, gene_matrix):
    summary = []
    details = []
    source_groups = source.groupby("Gene symbol")["UniProt ID"].agg(lambda x: ";".join(sorted(map(str, x))))
    source_counts = source.groupby("Gene symbol")["UniProt ID"].size()
    universe = set(gene_matrix.index)
    for size in (50, 100):
        for rmp in RMPS:
            genes = signatures[size][rmp]
            mapped = [g for g in genes if g in universe]
            for rank, gene in enumerate(genes, 1):
                details.append({
                    "RMP": rmp,
                    "hierarchy": HIERARCHY[rmp],
                    "signature": f"Top{size}",
                    "frozen_rank": rank,
                    "gene": gene,
                    "mapped": "YES" if gene in universe else "NO",
                    "protein_entries": int(source_counts.get(gene, 0)),
                    "uniprot_ids": source_groups.get(gene, ""),
                    "duplicate_rule": "per-sample median" if source_counts.get(gene, 0) > 1 else "not applicable",
                })
            coverage = len(mapped) / size
            category = "GOOD" if coverage >= 0.60 else ("MODERATE" if coverage >= 0.40 else "LOW / EXPLORATORY")
            summary.append({
                "RMP": rmp,
                "hierarchy": HIERARCHY[rmp],
                "signature": f"Top{size}",
                "frozen_genes": size,
                "mapped_proteins": len(mapped),
                "coverage": coverage,
                "coverage_category": category,
                "unmapped_genes": ";".join(g for g in genes if g not in universe),
            })
    return pd.DataFrame(summary), pd.DataFrame(details)


def score_signatures(signatures_by_rmp, z, sample_cols):
    score = pd.DataFrame(index=sample_cols)
    detected = pd.DataFrame(index=sample_cols)
    mapped_by_rmp = {}
    for rmp in RMPS:
        mapped = [g for g in signatures_by_rmp[rmp] if g in z.index]
        mapped_by_rmp[rmp] = mapped
        block = z.loc[mapped, sample_cols]
        score[rmp] = block.mean(axis=0, skipna=True)
        detected[rmp] = block.notna().sum(axis=0)
    return score, detected, mapped_by_rmp


def paired_arrays(score, patient_ids, rmp):
    normal = np.asarray([score.loc[f"{p}N", rmp] for p in patient_ids], dtype=float)
    tumor = np.asarray([score.loc[f"{p}T", rmp] for p in patient_ids], dtype=float)
    keep = np.isfinite(normal) & np.isfinite(tumor)
    return normal[keep], tumor[keep], np.asarray(patient_ids)[keep]


def bootstrap_effect(diff, rng, n_boot=BOOTSTRAP_REPLICATES):
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    draws = diff[idx].mean(axis=1)
    observed = diff.mean()
    if observed > 0:
        sign_stability = float(np.mean(draws > 0))
    elif observed < 0:
        sign_stability = float(np.mean(draws < 0))
    else:
        sign_stability = float(np.mean(draws == 0))
    return draws, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), sign_stability


def paired_stats(score, mapped_by_rmp, patient_ids, rng):
    rows = []
    draws_out = []
    for rmp in RMPS:
        normal, tumor, ids = paired_arrays(score, patient_ids, rmp)
        diff = tumor - normal
        stat, p = wilcoxon(tumor, normal, alternative="two-sided", zero_method="wilcox")
        boot, lo, hi, stability = bootstrap_effect(diff, rng)
        sd_diff = np.std(diff, ddof=1)
        dz = float(np.mean(diff) / sd_diff) if sd_diff > 0 else np.nan
        rows.append({
            "RMP": rmp,
            "hierarchy": HIERARCHY[rmp],
            "N_pairs": len(diff),
            "mapped_proteins": len(mapped_by_rmp[rmp]),
            "mean_normal_score": float(np.mean(normal)),
            "mean_tumor_score": float(np.mean(tumor)),
            "median_normal_score": float(np.median(normal)),
            "median_tumor_score": float(np.median(tumor)),
            "mean_paired_difference_T_minus_N": float(np.mean(diff)),
            "median_paired_difference_T_minus_N": float(np.median(diff)),
            "paired_Cohens_dz": dz,
            "bootstrap_CI_low": lo,
            "bootstrap_CI_high": hi,
            "bootstrap_sign_stability": stability,
            "wilcoxon_statistic": float(stat),
            "raw_P": float(p),
        })
        draws_out.append(pd.DataFrame({"RMP": rmp, "bootstrap_replicate": np.arange(1, len(boot) + 1), "mean_paired_difference": boot}))
    stats = pd.DataFrame(rows)
    stats["FDR"] = bh_fdr(stats["raw_P"])
    return stats, pd.concat(draws_out, ignore_index=True)


def create_matching_strata(gene_matrix):
    props = pd.DataFrame(index=gene_matrix.index)
    props["missingness"] = gene_matrix.isna().mean(axis=1)
    props["mean_abundance"] = gene_matrix.mean(axis=1, skipna=True)
    props["missing_bin"] = pd.qcut(props["missingness"].rank(method="first"), 5, labels=False)
    props["abundance_bin"] = pd.qcut(props["mean_abundance"].rank(method="first"), 5, labels=False)
    props["stratum"] = props["missing_bin"].astype(int).astype(str) + "_" + props["abundance_bin"].astype(int).astype(str)
    return props


def random_set_calibration(stats, mapped_by_rmp, z, gene_matrix, sample_cols, patient_ids, rng):
    props = create_matching_strata(gene_matrix.loc[z.index])
    gene_to_idx = {g: i for i, g in enumerate(z.index)}
    zarr = z.loc[:, sample_cols].to_numpy(dtype=float)
    n_idx = np.asarray([sample_cols.index(f"{p}N") for p in patient_ids])
    t_idx = np.asarray([sample_cols.index(f"{p}T") for p in patient_ids])
    by_stratum = {s: [gene_to_idx[g] for g in grp.index] for s, grp in props.groupby("stratum")}
    rows = []
    all_null = []
    for rmp in RMPS:
        target_genes = mapped_by_rmp[rmp]
        target_idx = {gene_to_idx[g] for g in target_genes}
        counts = props.loc[target_genes, "stratum"].value_counts().to_dict()
        null = np.empty(RANDOM_SET_REPLICATES, dtype=float)
        fallback_count = 0
        for b in range(RANDOM_SET_REPLICATES):
            selected = []
            used = set()
            for stratum, count in counts.items():
                candidates = [i for i in by_stratum[stratum] if i not in target_idx and i not in used]
                if len(candidates) < count:
                    fallback_count += 1
                    candidates = [i for i in range(len(z.index)) if i not in target_idx and i not in used]
                pick = rng.choice(candidates, size=count, replace=False).tolist()
                selected.extend(pick)
                used.update(pick)
            block = zarr[np.asarray(selected, dtype=int), :]
            with np.errstate(invalid="ignore"):
                random_score = np.nanmean(block, axis=0)
            null[b] = float(np.nanmean(random_score[t_idx] - random_score[n_idx]))
        obs = float(stats.loc[stats["RMP"] == rmp, "mean_paired_difference_T_minus_N"].iloc[0])
        center = float(np.median(null))
        empirical_p = float((1 + np.sum(np.abs(null - center) >= abs(obs - center))) / (RANDOM_SET_REPLICATES + 1))
        percentile = float(100 * (1 + np.sum(null <= obs)) / (RANDOM_SET_REPLICATES + 1))
        rows.append({
            "RMP": rmp,
            "hierarchy": HIERARCHY[rmp],
            "mapped_proteins": len(target_genes),
            "random_sets": RANDOM_SET_REPLICATES,
            "matching_variables": "protein missingness quintile + mean-abundance quintile",
            "null_mean": float(np.mean(null)),
            "null_median": center,
            "null_SD": float(np.std(null, ddof=1)),
            "observed_effect": obs,
            "null_percentile": percentile,
            "empirical_P_two_sided_centered_on_null_median": empirical_p,
            "fallback_draw_events": fallback_count,
        })
        all_null.append(pd.DataFrame({"RMP": rmp, "random_set_replicate": np.arange(1, RANDOM_SET_REPLICATES + 1), "mean_paired_difference": null}))
    result = pd.DataFrame(rows)
    result["empirical_FDR"] = bh_fdr(result["empirical_P_two_sided_centered_on_null_median"])
    return result, pd.concat(all_null, ignore_index=True), props.reset_index(names="gene")


def add_cross_layer_and_support(primary, top100, random_cal):
    prior = pd.read_csv(resolve_data(PHASE_F_EFFECTS), sep="\t")[["RMP", "median_paired_delta"]].copy()
    prior["median_paired_delta"] = pd.to_numeric(prior["median_paired_delta"])
    out = primary.merge(prior, on="RMP", how="left", validate="one_to_one")
    out = out.merge(top100[["RMP", "mean_paired_difference_T_minus_N", "FDR", "bootstrap_sign_stability"]].rename(columns={
        "mean_paired_difference_T_minus_N": "Top100_effect",
        "FDR": "Top100_FDR",
        "bootstrap_sign_stability": "Top100_bootstrap_sign_stability",
    }), on="RMP", validate="one_to_one")
    out = out.merge(random_cal[["RMP", "empirical_P_two_sided_centered_on_null_median", "empirical_FDR", "null_percentile"]], on="RMP", validate="one_to_one")
    out["cross_layer_direction_vs_frozen_GSE53625"] = np.where(
        np.sign(out["mean_paired_difference_T_minus_N"]) == np.sign(out["median_paired_delta"]), "CONSISTENT", "DISCORDANT"
    )
    out["Top50_Top100_direction"] = np.where(
        np.sign(out["mean_paired_difference_T_minus_N"]) == np.sign(out["Top100_effect"]), "CONSISTENT", "DISCORDANT"
    )
    ci_excludes = (out["bootstrap_CI_low"] > 0) | (out["bootstrap_CI_high"] < 0)
    classes = []
    for idx, row in out.iterrows():
        coverage = row["mapped_proteins"] / 50
        direction_ok = row["cross_layer_direction_vs_frozen_GSE53625"] == "CONSISTENT"
        size_ok = row["Top50_Top100_direction"] == "CONSISTENT"
        strong = coverage >= .60 and direction_ok and size_ok and row["FDR"] < .05 and ci_excludes.iloc[idx] and row["empirical_FDR"] < .05
        evidence_count = int(row["FDR"] < .05) + int(ci_excludes.iloc[idx]) + int(row["empirical_FDR"] < .05)
        moderate = coverage >= .40 and direction_ok and size_ok and row["bootstrap_sign_stability"] >= .90 and evidence_count >= 2
        opposite_stable = (not direction_ok) and ci_excludes.iloc[idx] and row["bootstrap_sign_stability"] >= .95
        null_across = row["FDR"] >= .20 and (not ci_excludes.iloc[idx]) and row["empirical_FDR"] >= .20 and row["bootstrap_sign_stability"] <= .80
        if strong:
            label = "STRONG"
        elif moderate:
            label = "MODERATE"
        elif opposite_stable or null_across:
            label = "NOT SUPPORTED"
        else:
            label = "WEAK / EXPLORATORY"
        classes.append(label)
    out["orthogonal_protein_support"] = classes
    out["tier_preservation_note"] = [
        ("Proteomic support does not alter the frozen " + HIERARCHY[rmp].lower() + " hierarchy")
        for rmp in out["RMP"]
    ]
    return out


def write_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df.to_csv(path, index=False, sep=sep, encoding="utf-8-sig")


def add_panel_label(ax, label):
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def make_figures(mapping, combined, score50, bootstrap_draws, random_null, random_cal, sensitivity, corr):
    fig_dir = OUT / "07_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"PRIMARY": "#245B8A", "SECONDARY": "#D08C2C", "EXPLORATORY": "#7B7B8A"}
    tumor_color, normal_color = "#B5403B", "#4C78A8"

    # Main Figure H1
    fig = plt.figure(figsize=(13.2, 15.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.3, 1.15])
    axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :]); axD = fig.add_subplot(gs[2, 0]); axE = fig.add_subplot(gs[2, 1])

    axA.axis("off")
    boxes = [
        (.23, .72, "Frozen scRNA-seq\nRMP signatures"),
        (.76, .72, "Gene-to-protein\nmapping"),
        (.23, .28, "124 paired ESCC\nTMT proteomes"),
        (.76, .28, "Unweighted scores +\npaired robustness tests"),
    ]
    for x, yb, lab in boxes:
        axA.text(x, yb, lab, ha="center", va="center", fontsize=9.0,
                 bbox=dict(boxstyle="round,pad=.55", fc="#F4F6F8", ec="#7A8793", lw=.9))
    for start, end in [((.40,.72),(.59,.72)), ((.76,.58),(.38,.31)), ((.40,.28),(.59,.28))]:
        axA.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.0, color="#59636E"))
    axA.text(.5, .04, "Frozen before proteomic outcomes; no reweighting or gene selection", ha="center", fontsize=8, color="#444444")
    add_panel_label(axA, "A")

    cov = mapping[mapping["signature"] == "Top50"].set_index("RMP").loc[DISPLAY_ORDER]
    y = np.arange(len(cov))
    axB.barh(y, cov["coverage"] * 100, color=[colors[h] for h in cov["hierarchy"]], height=.65)
    axB.set_yticks(y, [x.replace("RMP_", "") for x in cov.index])
    axB.invert_yaxis(); axB.set_xlim(0, 100); axB.set_xlabel("Mapped frozen Top50 proteins (%)", fontsize=9)
    axB.axvline(60, color="#444444", ls="--", lw=.8); axB.axvline(40, color="#999999", ls=":", lw=.8)
    for yi, (_, row) in enumerate(cov.iterrows()):
        axB.text(row["coverage"]*100+1.2, yi, f'{int(row["mapped_proteins"])}/50', va="center", fontsize=8)
    style_axis(axB); add_panel_label(axB, "B")

    prim = ["RMP_C01", "RMP_C02", "RMP_C07"]
    positions = {r: i for i, r in enumerate(prim)}
    rng_j = np.random.default_rng(SEED + 91)
    for rmp in prim:
        x0 = positions[rmp] * 3.0
        nvals = np.array([score50.loc[f"{p}N", rmp] for p in range(1,125)])
        tvals = np.array([score50.loc[f"{p}T", rmp] for p in range(1,125)])
        jitter = rng_j.normal(0, .025, len(nvals))
        for j in range(len(nvals)):
            axC.plot([x0+jitter[j], x0+1+jitter[j]], [nvals[j], tvals[j]], color="#B6BBC1", alpha=.22, lw=.45, zorder=1)
        axC.scatter(np.full(len(nvals), x0)+jitter, nvals, s=9, color=normal_color, alpha=.58, edgecolors="none", zorder=2)
        axC.scatter(np.full(len(tvals), x0+1)+jitter, tvals, s=9, color=tumor_color, alpha=.58, edgecolors="none", zorder=2)
        axC.plot([x0, x0+1], [np.mean(nvals), np.mean(tvals)], color="black", lw=2.2, marker="o", ms=4, zorder=3)
        p = combined.loc[combined["RMP"] == rmp, "FDR"].iloc[0]
        axC.text(x0+.5, max(np.nanmax(nvals),np.nanmax(tvals))+.12, f"FDR = {p:.3g}", ha="center", fontsize=8)
    axC.set_xticks([0,1,3,4,6,7], ["N","T","N","T","N","T"])
    for rmp, x0 in positions.items(): axC.text(x0+.5, axC.get_ylim()[0]-.12, rmp.replace("RMP_", ""), ha="center", va="top", fontsize=9, fontweight="bold")
    axC.set_ylabel("Protein-RMP score", fontsize=9); axC.set_title("Primary RMPs: paired patient-level tumor–normal scores", fontsize=10, pad=12)
    style_axis(axC); add_panel_label(axC, "C")

    forest = combined.set_index("RMP").loc[DISPLAY_ORDER].reset_index()
    yy = np.arange(len(forest))
    for i, row in forest.iterrows():
        axD.plot([row["bootstrap_CI_low"], row["bootstrap_CI_high"]], [i,i], color=colors[row["hierarchy"]], lw=1.8)
        marker = "o" if row["FDR"] < .05 else "o"
        face = colors[row["hierarchy"]] if row["FDR"] < .05 else "white"
        axD.scatter(row["mean_paired_difference_T_minus_N"], i, s=42, marker=marker, facecolor=face, edgecolor=colors[row["hierarchy"]], lw=1.2, zorder=3)
    axD.axvline(0, color="#555555", lw=.8)
    axD.set_yticks(yy, [x.replace("RMP_", "") for x in forest["RMP"]]); axD.invert_yaxis()
    axD.set_xlabel("Mean paired difference (Tumor − Normal)\nwith patient-pair bootstrap 95% CI", fontsize=9)
    axD.text(.02, -.17, "Filled: BH-FDR < 0.05; open: FDR not supported", transform=axD.transAxes, fontsize=7.5)
    style_axis(axD); add_panel_label(axD, "D")

    null_primary = random_null[random_null["RMP"].isin(prim)]
    null_colors = {"RMP_C01":"#4C78A8","RMP_C02":"#F28E2B","RMP_C07":"#7B5AA6"}
    for rmp in prim:
        vals = null_primary.loc[null_primary["RMP"] == rmp, "mean_paired_difference"].to_numpy()
        axE.hist(vals, bins=45, density=True, histtype="step", lw=1.4, color=null_colors[rmp], label=rmp.replace("RMP_", ""))
        obs = combined.loc[combined["RMP"] == rmp, "mean_paired_difference_T_minus_N"].iloc[0]
        axE.axvline(obs, color=null_colors[rmp], lw=2)
    axE.set_xlabel("Random-set mean paired difference", fontsize=9); axE.set_ylabel("Density", fontsize=9)
    axE.legend(frameon=False, fontsize=8, title="Curves/null; lines/observed", title_fontsize=7)
    axE.set_title("Matched random protein-set calibration", fontsize=10)
    style_axis(axE); add_panel_label(axE, "E")
    fig.savefig(fig_dir / "Figure_H1_frozen_RMP_orthogonal_proteomic_validation.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_dir / "Figure_H1_frozen_RMP_orthogonal_proteomic_validation.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Supplementary H-S1
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    sens = sensitivity.set_index("RMP").loc[DISPLAY_ORDER].reset_index()
    y = np.arange(len(sens)); off=.16
    ax.errorbar(sens["Top50_effect"], y-off, xerr=[sens["Top50_effect"]-sens["Top50_CI_low"],sens["Top50_CI_high"]-sens["Top50_effect"]], fmt="o", color="#245B8A", capsize=2, label="Frozen Top50")
    ax.errorbar(sens["Top100_effect"], y+off, xerr=[sens["Top100_effect"]-sens["Top100_CI_low"],sens["Top100_CI_high"]-sens["Top100_effect"]], fmt="s", color="#D08C2C", capsize=2, label="Frozen Top100")
    ax.axvline(0,color="#555",lw=.8); ax.set_yticks(y,[x.replace("RMP_","") for x in sens["RMP"]]); ax.invert_yaxis()
    ax.set_xlabel("Mean paired difference (Tumor − Normal), bootstrap 95% CI"); ax.legend(frameon=False)
    ax.set_title("Frozen Top50 primary analysis and Top100 sensitivity")
    style_axis(ax)
    fig.savefig(fig_dir / "Supplementary_Figure_H_S1_Top50_Top100_sensitivity.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_dir / "Supplementary_Figure_H_S1_Top50_Top100_sensitivity.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Supplementary H-S2
    fig, axes = plt.subplots(4,2,figsize=(10,9),constrained_layout=True)
    for ax, rmp in zip(axes.flat, DISPLAY_ORDER):
        vals=bootstrap_draws.loc[bootstrap_draws["RMP"]==rmp,"mean_paired_difference"].to_numpy()
        ax.hist(vals,bins=40,color=colors[HIERARCHY[rmp]],alpha=.78)
        ax.axvline(0,color="#555",lw=.8); ax.axvline(np.mean(vals),color="black",lw=1.3)
        st=combined.loc[combined["RMP"]==rmp,"bootstrap_sign_stability"].iloc[0]
        ax.set_title(f'{rmp.replace("RMP_","")} | sign stability = {st:.3f}',fontsize=9)
        style_axis(ax)
    axes.flat[-1].axis("off")
    fig.suptitle("Patient-pair bootstrap distributions for frozen Top50 Protein-RMP effects",fontsize=12)
    fig.savefig(fig_dir / "Supplementary_Figure_H_S2_bootstrap_stability.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_dir / "Supplementary_Figure_H_S2_bootstrap_stability.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Supplementary H-S3
    fig, ax = plt.subplots(figsize=(7.2,6.2), constrained_layout=True)
    labels=[x.replace("RMP_","") for x in DISPLAY_ORDER]
    mat=corr.loc[DISPLAY_ORDER,DISPLAY_ORDER].to_numpy()
    im=ax.imshow(mat,vmin=-1,vmax=1,cmap="RdBu_r")
    ax.set_xticks(range(7),labels,rotation=45,ha="right"); ax.set_yticks(range(7),labels)
    for i in range(7):
        for j in range(7): ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(mat[i,j])>.55 else "black")
    cb=fig.colorbar(im,ax=ax,shrink=.78); cb.set_label("Spearman rho")
    ax.set_title("Tumor-sample Protein-RMP score correlations (n = 124 patients)")
    fig.savefig(fig_dir / "Supplementary_Figure_H_S3_protein_RMP_correlations.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(fig_dir / "Supplementary_Figure_H_S3_protein_RMP_correlations.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rng = np.random.default_rng(SEED)
    signatures, freeze_audit = read_signatures()
    raw, source, gene_matrix, z, sample_cols, patient_ids, sample_audit, duplicate_audit = read_proteome()
    mapping, mapping_details = mapping_tables(signatures, source, gene_matrix)

    score50, detected50, mapped50 = score_signatures(signatures[50], z, sample_cols)
    score100, detected100, mapped100 = score_signatures(signatures[100], z, sample_cols)
    stats50, boot50 = paired_stats(score50, mapped50, patient_ids, rng)
    stats100, boot100 = paired_stats(score100, mapped100, patient_ids, rng)
    random_cal, random_null, match_props = random_set_calibration(stats50, mapped50, z, gene_matrix, sample_cols, patient_ids, rng)
    combined = add_cross_layer_and_support(stats50, stats100, random_cal)

    sensitivity = combined[["RMP","hierarchy","mean_paired_difference_T_minus_N","bootstrap_CI_low","bootstrap_CI_high","FDR","bootstrap_sign_stability","Top100_effect","Top100_FDR","Top100_bootstrap_sign_stability","Top50_Top100_direction"]].rename(columns={
        "mean_paired_difference_T_minus_N":"Top50_effect","bootstrap_CI_low":"Top50_CI_low","bootstrap_CI_high":"Top50_CI_high","FDR":"Top50_FDR","bootstrap_sign_stability":"Top50_bootstrap_sign_stability"
    }).merge(stats100[["RMP","bootstrap_CI_low","bootstrap_CI_high","mapped_proteins"]].rename(columns={"bootstrap_CI_low":"Top100_CI_low","bootstrap_CI_high":"Top100_CI_high","mapped_proteins":"Top100_mapped_proteins"}),on="RMP")
    sensitivity = sensitivity.merge(mapping[(mapping["signature"]=="Top50")][["RMP","mapped_proteins","coverage"]].rename(columns={"mapped_proteins":"Top50_mapped_proteins","coverage":"Top50_coverage"}),on="RMP")
    sensitivity = sensitivity.merge(mapping[(mapping["signature"]=="Top100")][["RMP","coverage"]].rename(columns={"coverage":"Top100_coverage"}),on="RMP")

    score_meta = pd.DataFrame({"sample_id": sample_cols, "patient_id": [int(c[:-1]) for c in sample_cols], "tissue": ["Adjacent normal" if c.endswith("N") else "Tumor" for c in sample_cols]})
    score50_out = pd.concat([score_meta.reset_index(drop=True), score50.loc[sample_cols].reset_index(drop=True)], axis=1)
    score100_out = pd.concat([score_meta.reset_index(drop=True), score100.loc[sample_cols].reset_index(drop=True)], axis=1)
    detected50_out = pd.concat([score_meta.reset_index(drop=True), detected50.loc[sample_cols].reset_index(drop=True)], axis=1)
    tumor_scores = score50_out[score50_out["tissue"] == "Tumor"].set_index("patient_id")[RMPS]
    corr = tumor_scores.corr(method="spearman")

    # Audits and core tables
    write_csv(freeze_audit, OUT / "01_freeze_audit" / "signature_freeze_audit.tsv")
    write_csv(sample_audit, OUT / "00_data_audit" / "sample_pair_and_missingness_audit.tsv")
    write_csv(duplicate_audit, OUT / "02_mapping" / "duplicate_gene_mapping_audit.tsv")
    write_csv(mapping, OUT / "08_tables" / "Table_H1_RMP_protein_mapping_coverage.csv")
    write_csv(mapping_details, OUT / "02_mapping" / "gene_to_protein_mapping_detail.tsv")
    write_csv(combined, OUT / "08_tables" / "Table_H2_paired_tumor_normal_statistics.csv")
    write_csv(combined[["RMP","hierarchy","N_pairs","mapped_proteins","mean_paired_difference_T_minus_N","bootstrap_CI_low","bootstrap_CI_high","bootstrap_sign_stability"]], OUT / "08_tables" / "Table_H3_bootstrap_stability.csv")
    write_csv(random_cal, OUT / "08_tables" / "Table_H4_random_protein_set_calibration.csv")
    write_csv(sensitivity, OUT / "08_tables" / "Table_H5_Top50_vs_Top100_sensitivity.csv")
    write_csv(score50_out, OUT / "03_primary_analysis" / "Top50_sample_Protein_RMP_scores.tsv")
    write_csv(score100_out, OUT / "06_top100_sensitivity" / "Top100_sample_Protein_RMP_scores.tsv")
    write_csv(detected50_out, OUT / "03_primary_analysis" / "Top50_per_sample_detected_proteins.tsv")
    write_csv(boot50, OUT / "04_bootstrap" / "Top50_patient_pair_bootstrap_draws.tsv")
    write_csv(boot100, OUT / "06_top100_sensitivity" / "Top100_patient_pair_bootstrap_draws.tsv")
    write_csv(random_null, OUT / "05_random_set_calibration" / "matched_random_set_null_draws.tsv")
    write_csv(match_props, OUT / "05_random_set_calibration" / "protein_universe_matching_strata.tsv")
    corr.to_csv(OUT / "06_top100_sensitivity" / "Top50_tumor_patient_Spearman_correlations.tsv", sep="\t", encoding="utf-8-sig")

    provenance = {
        "publication": "Liu W, et al. Large-scale and high-resolution mass spectrometry-based proteomics profiling defines molecular subtypes of esophageal cancer for therapeutic targeting.",
        "journal": "Nature Communications",
        "year": 2021,
        "volume_article": "12:4961",
        "doi": "10.1038/s41467-021-25202-5",
        "proteomexchange_accession": "PXD021701",
        "iprox_accession": "IPX0002501000",
        "processed_data_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8368010/supplementaryFiles",
        "processed_file": DATA_XLSX.name,
        "processed_file_sha256": sha256(DATA_XLSX),
        "download_date": str(date.today()),
        "processed_sheet": "b. 9,300 proteins",
        "source_description": "Expression ratios of 9,300 proteins quantified with high confidence in at least half of samples; column-mean normalized and log2-transformed by the authors.",
        "patients": 124,
        "samples": 248,
        "paired_design": True,
    }
    (OUT / "00_data_audit" / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "random_seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_set_replicates": RANDOM_SET_REPLICATES,
    }
    (OUT / "10_logs" / "environment_versions.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (OUT / "10_logs" / "analysis_complete.json").write_text(json.dumps({"status":"PASS","rows_protein":len(raw),"unique_mapped_gene_symbols":len(gene_matrix),"patients":len(patient_ids)},indent=2),encoding="utf-8")

    make_figures(mapping, combined, score50, boot50, random_null, random_cal, sensitivity, corr)
    print(combined[["RMP","hierarchy","mapped_proteins","mean_paired_difference_T_minus_N","bootstrap_CI_low","bootstrap_CI_high","raw_P","FDR","empirical_FDR","Top50_Top100_direction","cross_layer_direction_vs_frozen_GSE53625","orthogonal_protein_support"]].to_string(index=False))


if __name__ == "__main__":
    main()
