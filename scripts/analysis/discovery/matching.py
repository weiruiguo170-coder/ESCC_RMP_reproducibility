#!/usr/bin/env python
"""Phase C: locked, technical-only cross-cohort program matching."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import anndata as ad
from scipy.stats import hypergeom, rankdata


PROJECT_ROOT = INPUT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results"
B1 = RESULTS_DIR / "cnmf_formal"
OUT = RESULTS_DIR / "cross_cohort_matching"
CONFIG = json.loads((PROJECT_ROOT / "config" / "cross_cohort_matching.json").read_text(encoding="utf-8"))
SEED = int(CONFIG["random_seed"])
N_PERM = int(CONFIG["gene_label_permutations"])
COHORTS = ("GSE160269", "GSE188900", "GSE196756")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def verify_b1() -> pd.DataFrame:
    manifest = pd.read_csv(resolve_data(B1 / "output_manifest.tsv"), sep="\t")
    rows = []
    for row in manifest.itertuples(index=False):
        path = Path(row.file)
        observed = sha256(path)
        if observed.lower() != str(row.sha256).lower():
            raise RuntimeError(f"PHASE_B1_HASH_MISMATCH: {path}")
        rows.append({"cohort": row.cohort, "phase_b1_file": str(path.resolve()),
                     "bytes": path.stat().st_size, "sha256": observed, "hash_status": "PASS"})
    frozen = pd.DataFrame(rows)
    frozen.to_csv(OUT / "phase_b1_frozen_file_manifest.tsv", sep="\t", index=False)
    return frozen


def canonicalize(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict[str, int]]:
    raw_n = len(frame)
    temp = frame.copy()
    temp.index = temp.index.astype(str).str.strip().str.upper()
    empty_mask = np.asarray(temp.index == "")
    empty = int(empty_mask.sum())
    temp = temp.loc[~empty_mask]
    duplicate_rows = int(temp.index.duplicated(keep=False).sum())
    if mode == "score":
        collapsed = temp.groupby(level=0, sort=True).mean()
    elif mode == "spectra":
        collapsed = temp.groupby(level=0, sort=True).sum()
    else:
        raise ValueError(mode)
    return collapsed, {"raw_gene_rows": raw_n, "empty_removed": empty,
                       "duplicate_rows_involved": duplicate_rows,
                       "canonical_unique_genes": len(collapsed),
                       "duplicates_collapsed": raw_n - empty - len(collapsed)}


def load_cohort(cohort: str) -> dict[str, object]:
    scores_raw = pd.read_csv(resolve_data(B1 / cohort / "gene_program_scores.tsv"), sep="\t", index_col=0)
    spectra_raw = pd.read_csv(resolve_data(B1 / cohort / "gene_program_spectra.tsv"), sep="\t", index_col=0)
    b1_config = json.loads((B1 / cohort / "exact_configuration.json").read_text(encoding="utf-8"))
    input_path = Path(b1_config["input"])
    expected_input_hash = str(pd.read_csv(resolve_data(B1 / cohort / "formal_cnmf_metrics.tsv"), sep="\t").iloc[0]["input_sha256"])
    observed_input_hash = sha256(input_path)
    if observed_input_hash.lower() != expected_input_hash.lower():
        raise RuntimeError(f"{cohort}: frozen malignant-ready input hash mismatch")
    adata = ad.read_h5ad(input_path, backed="r")
    if "gene_symbol" in adata.var.columns:
        symbols = adata.var["gene_symbol"].astype(str)
        symbol_map = pd.Series(symbols.to_numpy(), index=adata.var_names.astype(str))
    else:
        symbol_map = pd.Series(adata.var_names.astype(str), index=adata.var_names.astype(str))
    adata.file.close()
    def map_symbols(frame: pd.DataFrame) -> pd.DataFrame:
        original = frame.index.astype(str)
        mapped = symbol_map.reindex(original)
        fallback = pd.Series(original, index=original)
        mapped.index = original
        use = mapped.where(mapped.notna() & mapped.astype(str).str.strip().ne(""), fallback)
        result = frame.copy()
        result.index = use.astype(str).to_numpy()
        return result
    scores_raw = map_symbols(scores_raw)
    spectra_raw = map_symbols(spectra_raw)
    scores, score_audit = canonicalize(scores_raw, "score")
    spectra, spectra_audit = canonicalize(spectra_raw, "spectra")
    if list(scores.columns) != list(spectra.columns):
        raise RuntimeError(f"{cohort}: score/spectra program IDs disagree")
    flags = pd.read_csv(resolve_data(B1 / cohort / "program_qc_flags.tsv"), sep="\t").set_index("program_id")
    expected = set(scores.columns)
    if set(flags.index) != expected:
        raise RuntimeError(f"{cohort}: QC flags/program IDs disagree")
    score_audit["identifier_source"] = "frozen_input_var_gene_symbol"
    spectra_audit["identifier_source"] = "frozen_input_var_gene_symbol"
    return {"scores": scores, "spectra": spectra, "flags": flags,
            "score_audit": score_audit, "spectra_audit": spectra_audit}


def prep_arrays(scores_a: pd.DataFrame, spectra_a: pd.DataFrame,
                scores_b: pd.DataFrame, spectra_b: pd.DataFrame) -> dict[str, object]:
    genes = scores_a.index.intersection(scores_b.index).intersection(spectra_a.index).intersection(spectra_b.index)
    genes = pd.Index(sorted(genes))
    if len(genes) < 100:
        raise RuntimeError("Shared gene universe is too small")
    sa = scores_a.loc[genes].to_numpy(float)
    sb = scores_b.loc[genes].to_numpy(float)
    pa = np.maximum(spectra_a.loc[genes].to_numpy(float), 0)
    pb = np.maximum(spectra_b.loc[genes].to_numpy(float), 0)

    ra = rankdata(sa, axis=0, method="average")
    rb = rankdata(sb, axis=0, method="average")
    za = (ra - ra.mean(axis=0)) / ra.std(axis=0)
    zb = (rb - rb.mean(axis=0)) / rb.std(axis=0)
    pa = pa / np.where(pa.sum(axis=0) == 0, 1, pa.sum(axis=0))
    pb = pb / np.where(pb.sum(axis=0) == 0, 1, pb.sum(axis=0))
    pa = pa / np.where(np.sqrt((pa * pa).sum(axis=0)) == 0, 1, np.sqrt((pa * pa).sum(axis=0)))
    pb = pb / np.where(np.sqrt((pb * pb).sum(axis=0)) == 0, 1, np.sqrt((pb * pb).sum(axis=0)))

    def top_matrix(scores: np.ndarray, n: int) -> np.ndarray:
        result = np.zeros(scores.shape, dtype=np.int16)
        for j in range(scores.shape[1]):
            take = np.argpartition(scores[:, j], -n)[-n:]
            result[take, j] = 1
        return result

    return {"genes": genes, "za": za, "zb": zb, "pa": pa, "pb": pb,
            "top50a": top_matrix(sa, 50), "top50b": top_matrix(sb, 50),
            "top100a": top_matrix(sa, 100), "top100b": top_matrix(sb, 100)}


def pairwise_analysis(name: str, scores_a: pd.DataFrame, spectra_a: pd.DataFrame,
                      scores_b: pd.DataFrame, spectra_b: pd.DataFrame,
                      flags_a: pd.DataFrame, flags_b: pd.DataFrame, seed_offset: int = 0) -> pd.DataFrame:
    arrays = prep_arrays(scores_a, spectra_a, scores_b, spectra_b)
    za, zb, pa, pb = arrays["za"], arrays["zb"], arrays["pa"], arrays["pb"]
    n_genes = len(arrays["genes"])
    rho = za.T @ zb / n_genes
    cosine = pa.T @ pb
    overlap50 = arrays["top50a"].T @ arrays["top50b"]
    overlap100 = arrays["top100a"].T @ arrays["top100b"]
    jaccard50 = overlap50 / (100 - overlap50)
    jaccard100 = overlap100 / (200 - overlap100)
    hyper50 = hypergeom.sf(overlap50 - 1, n_genes, 50, 50)
    hyper100 = hypergeom.sf(overlap100 - 1, n_genes, 100, 100)
    composite = (np.maximum(rho, 0) + cosine + overlap50 / 50.0) / 3.0

    exceed_rho = np.zeros_like(rho, dtype=np.int64)
    exceed_cos = np.zeros_like(cosine, dtype=np.int64)
    exceed_comp = np.zeros_like(composite, dtype=np.int64)
    sum_rho = np.zeros_like(rho); sum_cos = np.zeros_like(cosine); sum_comp = np.zeros_like(composite)
    ss_rho = np.zeros_like(rho); ss_cos = np.zeros_like(cosine); ss_comp = np.zeros_like(composite)
    rng = np.random.default_rng(SEED + seed_offset)
    top50a_t = arrays["top50a"].T
    for _ in range(N_PERM):
        perm = rng.permutation(n_genes)
        nr = za.T @ zb[perm, :] / n_genes
        nc = pa.T @ pb[perm, :]
        no = top50a_t @ arrays["top50b"][perm, :]
        ncomp = (np.maximum(nr, 0) + nc + no / 50.0) / 3.0
        exceed_rho += nr >= rho - 1e-15
        exceed_cos += nc >= cosine - 1e-15
        exceed_comp += ncomp >= composite - 1e-15
        sum_rho += nr; sum_cos += nc; sum_comp += ncomp
        ss_rho += nr * nr; ss_cos += nc * nc; ss_comp += ncomp * ncomp
    p_rho = (exceed_rho + 1) / (N_PERM + 1)
    p_cos = (exceed_cos + 1) / (N_PERM + 1)
    p_comp = (exceed_comp + 1) / (N_PERM + 1)
    mean_rho, mean_cos, mean_comp = sum_rho / N_PERM, sum_cos / N_PERM, sum_comp / N_PERM
    sd_rho = np.sqrt(np.maximum(ss_rho / N_PERM - mean_rho**2, 0))
    sd_cos = np.sqrt(np.maximum(ss_cos / N_PERM - mean_cos**2, 0))
    sd_comp = np.sqrt(np.maximum(ss_comp / N_PERM - mean_comp**2, 0))

    programs_a, programs_b = list(scores_a.columns), list(scores_b.columns)
    rows = []
    for i, program_a in enumerate(programs_a):
        for j, program_b in enumerate(programs_b):
            rows.append({
                "program_A": program_a, "program_B": program_b, "shared_gene_universe": n_genes,
                "spearman_rho": rho[i, j], "cosine_similarity": cosine[i, j],
                "top50_overlap": int(overlap50[i, j]), "top50_jaccard": jaccard50[i, j],
                "top50_hypergeom_p": hyper50[i, j], "top100_overlap": int(overlap100[i, j]),
                "top100_jaccard": jaccard100[i, j], "top100_hypergeom_p": hyper100[i, j],
                "composite_similarity": composite[i, j],
                "spearman_permutation_p": p_rho[i, j], "cosine_permutation_p": p_cos[i, j],
                "permutation_p": p_comp[i, j],
                "spearman_null_mean": mean_rho[i, j], "spearman_null_sd": sd_rho[i, j],
                "cosine_null_mean": mean_cos[i, j], "cosine_null_sd": sd_cos[i, j],
                "composite_null_mean": mean_comp[i, j], "composite_null_sd": sd_comp[i, j],
                "patient_dominance_flag_A": flags_a.loc[program_a, "patient_dominance_flag"],
                "patient_dominance_flag_B": flags_b.loc[program_b, "patient_dominance_flag"],
                "technical_flag_A": flags_a.loc[program_a, "technical_flags"],
                "technical_flag_B": flags_b.loc[program_b, "technical_flags"],
                "review_flag_A": flags_a.loc[program_a, "review_flags"],
                "review_flag_B": flags_b.loc[program_b, "review_flags"],
            })
    table = pd.DataFrame(rows)
    table["top50_hypergeom_FDR"] = bh(table["top50_hypergeom_p"].to_numpy())
    table["top100_hypergeom_FDR"] = bh(table["top100_hypergeom_p"].to_numpy())
    table["spearman_permutation_FDR"] = bh(table["spearman_permutation_p"].to_numpy())
    table["cosine_permutation_FDR"] = bh(table["cosine_permutation_p"].to_numpy())
    table["permutation_FDR"] = bh(table["permutation_p"].to_numpy())
    table["spearman_supported"] = (table.spearman_rho > 0) & (table.spearman_permutation_FDR < 0.05)
    table["cosine_supported"] = table.cosine_permutation_FDR < 0.05
    table["top50_supported"] = table.top50_hypergeom_FDR < 0.05
    table["independent_metric_support_count"] = table[["spearman_supported", "cosine_supported", "top50_supported"]].sum(axis=1)
    table["no_technical_artifact"] = table.technical_flag_A.eq("NONE") & table.technical_flag_B.eq("NONE")
    table["statistically_supported_edge"] = ((table.independent_metric_support_count >= 2) &
                                               (table.permutation_FDR < 0.05) & table.no_technical_artifact)
    table.to_csv(OUT / f"pairwise_similarity_{name}.tsv", sep="\t", index=False)
    return table


def add_rbh_and_split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_b = table.loc[table.groupby("program_A")["composite_similarity"].idxmax()].set_index("program_A")["program_B"]
    best_a = table.loc[table.groupby("program_B")["composite_similarity"].idxmax()].set_index("program_B")["program_A"]
    table = table.copy()
    table["best_B_for_A"] = table.program_A.map(best_b)
    table["best_A_for_B"] = table.program_B.map(best_a)
    table["is_best_for_A"] = table.program_B.eq(table.best_B_for_A)
    table["is_best_for_B"] = table.program_A.eq(table.best_A_for_B)
    table["RBH"] = table.is_best_for_A & table.is_best_for_B
    table["robust_primary_match"] = table.RBH & table.statistically_supported_edge
    best_score_a = table.groupby("program_A").composite_similarity.transform("max")
    best_score_b = table.groupby("program_B").composite_similarity.transform("max")
    relative = ((table.composite_similarity >= 0.85 * best_score_a) &
                (table.composite_similarity >= 0.85 * best_score_b))
    table["POTENTIAL_SPLIT_MERGE"] = table.statistically_supported_edge & ~table.RBH & relative
    rbh_cols = ["program_A", "program_B", "composite_similarity", "RBH", "robust_primary_match",
                "independent_metric_support_count", "permutation_FDR", "patient_dominance_flag_A",
                "patient_dominance_flag_B", "POTENTIAL_SPLIT_MERGE"]
    rbh = table.loc[table.RBH, rbh_cols].sort_values("composite_similarity", ascending=False)
    split = table.loc[table.POTENTIAL_SPLIT_MERGE, rbh_cols].sort_values("composite_similarity", ascending=False)
    return table, pd.concat([rbh.assign(relation="RBH"), split.assign(relation="POTENTIAL_SPLIT_MERGE")], ignore_index=True)


def make_candidate_representations(primary: pd.DataFrame, cohorts: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    robust = primary.loc[primary.robust_primary_match].sort_values("composite_similarity", ascending=False).copy()
    robust["rmp_id"] = [f"RMP_C{i:02d}" for i in range(1, len(robust) + 1)]
    genes = (cohorts["GSE160269"]["scores"].index.intersection(cohorts["GSE188900"]["scores"].index)
             .intersection(cohorts["GSE196756"]["scores"].index))
    genes = pd.Index(sorted(genes))
    candidate_scores, candidate_spectra = {}, {}
    for row in robust.itertuples():
        a = cohorts["GSE160269"]["scores"].loc[genes, row.program_A].to_numpy(float)
        b = cohorts["GSE188900"]["scores"].loc[genes, row.program_B].to_numpy(float)
        candidate_scores[row.rmp_id] = (rankdata(a) / len(a) + rankdata(b) / len(b)) / 2
        pa = np.maximum(cohorts["GSE160269"]["spectra"].loc[genes, row.program_A].to_numpy(float), 0)
        pb = np.maximum(cohorts["GSE188900"]["spectra"].loc[genes, row.program_B].to_numpy(float), 0)
        pa = pa / pa.sum(); pb = pb / pb.sum()
        mean = (pa + pb) / 2
        candidate_spectra[row.rmp_id] = mean / mean.sum()
    return robust, pd.DataFrame(candidate_scores, index=genes), pd.DataFrame(candidate_spectra, index=genes)


def plot_heatmap(table: pd.DataFrame, value: str, filename: str, title: str, fmt: str = ".2f") -> None:
    matrix = table.pivot(index="program_A", columns="program_B", values=value)
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    sns.heatmap(matrix, cmap="viridis", annot=True, fmt=fmt, ax=ax, cbar_kws={"label": value})
    ax.set_title(title); ax.set_xlabel("GSE188900 neutral program ID"); ax.set_ylabel("GSE160269 neutral program ID")
    fig.savefig(OUT / f"{filename}.png", dpi=250); fig.savefig(OUT / f"{filename}.pdf"); plt.close(fig)


def plot_network(primary: pd.DataFrame, evidence: pd.DataFrame) -> None:
    graph = nx.Graph()
    colors = {"GSE160269": "#4C78A8", "GSE188900": "#F58518", "GSE196756": "#54A24B"}
    for cohort, n in (("GSE160269", 10), ("GSE188900", 8), ("GSE196756", 5)):
        for i in range(1, n + 1):
            node = f"{cohort}_P{i:02d}"
            graph.add_node(node, cohort=cohort)
    for row in primary.loc[primary.statistically_supported_edge].itertuples():
        graph.add_edge(row.program_A, row.program_B, weight=max(float(row.composite_similarity), 0.01),
                       kind="primary", rbh=bool(row.RBH))
    for row in evidence.loc[evidence.THREE_COHORT_SUPPORT.eq("YES")].itertuples():
        graph.add_edge(row.program_A, row.supportive_program, weight=max(float(row.supportive_composite_similarity), 0.01),
                       kind="supportive", rbh=True)
        graph.add_edge(row.program_B, row.supportive_program, weight=max(float(row.supportive_composite_similarity), 0.01),
                       kind="supportive", rbh=True)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    fig, ax = plt.subplots(figsize=(16, 11), constrained_layout=True)
    if graph.number_of_nodes():
        pos = {}
        for x, cohort in enumerate(("GSE160269", "GSE188900", "GSE196756")):
            nodes = sorted([n for n, d in graph.nodes(data=True) if d["cohort"] == cohort])
            ys = np.linspace(1, -1, len(nodes)) if len(nodes) > 1 else np.array([0.0])
            pos.update({node: (x, float(y)) for node, y in zip(nodes, ys)})
        node_colors = [colors[graph.nodes[n]["cohort"]] for n in graph.nodes]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1200, edgecolors="white", ax=ax)
        primary_rbh = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "primary" and d["rbh"]]
        primary_other = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "primary" and not d["rbh"]]
        support_edges = [(u, v) for u, v, d in graph.edges(data=True) if d["kind"] == "supportive"]
        nx.draw_networkx_edges(graph, pos, edgelist=primary_other, width=0.8, alpha=0.18, edge_color="#777777", ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=primary_rbh, width=3.0, alpha=0.9, edge_color="#333333", ax=ax)
        nx.draw_networkx_edges(graph, pos, edgelist=support_edges, width=2.3, alpha=0.9, edge_color="#54A24B", style="dashed", ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=7.5, ax=ax)
        legend = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[c], markersize=10, label=c)
            for c in colors
        ] + [
            Line2D([0], [0], color="#333333", lw=3, label="Primary RBH"),
            Line2D([0], [0], color="#777777", lw=1, alpha=0.35, label="Other statistically supported primary edge"),
            Line2D([0], [0], color="#54A24B", lw=2.3, ls="--", label="Three-cohort supportive edge"),
        ]
        ax.legend(handles=legend, loc="upper center", ncol=3, frameon=False)
    ax.set_title("Phase C statistically supported cross-cohort program matching network")
    ax.axis("off")
    fig.savefig(OUT / "cross_cohort_program_matching_network.png", dpi=250)
    fig.savefig(OUT / "cross_cohort_program_matching_network.pdf")
    plt.close(fig)


def refresh_output_manifest() -> None:
    deliverables = [p for p in OUT.iterdir() if p.is_file() and p.name != "output_manifest.tsv"]
    pd.DataFrame([{"file": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha256(p)}
                  for p in sorted(deliverables)]).to_csv(OUT / "output_manifest.tsv", sep="\t", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = verify_b1()
    cohorts = {cohort: load_cohort(cohort) for cohort in COHORTS}
    audit_rows = []
    for cohort in COHORTS:
        for representation in ("score_audit", "spectra_audit"):
            audit_rows.append({"cohort": cohort, "representation": representation.replace("_audit", ""),
                               **cohorts[cohort][representation]})
    pair_universes = {
        "GSE160269_vs_GSE188900": len(cohorts["GSE160269"]["scores"].index.intersection(cohorts["GSE188900"]["scores"].index)),
        "GSE160269_vs_GSE196756": len(cohorts["GSE160269"]["scores"].index.intersection(cohorts["GSE196756"]["scores"].index)),
        "GSE188900_vs_GSE196756": len(cohorts["GSE188900"]["scores"].index.intersection(cohorts["GSE196756"]["scores"].index)),
    }
    for pair, count in pair_universes.items():
        audit_rows.append({"cohort": pair, "representation": "pairwise_shared_gene_universe",
                           "raw_gene_rows": np.nan, "empty_removed": np.nan, "duplicate_rows_involved": np.nan,
                           "canonical_unique_genes": count, "duplicates_collapsed": np.nan})
    pd.DataFrame(audit_rows).to_csv(OUT / "shared_gene_universe_audit.tsv", sep="\t", index=False)

    primary = pairwise_analysis("GSE160269_vs_GSE188900", cohorts["GSE160269"]["scores"],
                                cohorts["GSE160269"]["spectra"], cohorts["GSE188900"]["scores"],
                                cohorts["GSE188900"]["spectra"], cohorts["GSE160269"]["flags"],
                                cohorts["GSE188900"]["flags"], 0)
    primary, rbh_split = add_rbh_and_split(primary)
    primary.to_csv(OUT / "pairwise_similarity_GSE160269_vs_GSE188900.tsv", sep="\t", index=False)
    rbh_split.to_csv(OUT / "reciprocal_best_matches.tsv", sep="\t", index=False)
    primary.loc[primary.POTENTIAL_SPLIT_MERGE].to_csv(OUT / "potential_split_merge_flags.tsv", sep="\t", index=False)

    robust, candidate_scores, candidate_spectra = make_candidate_representations(primary, cohorts)
    if len(robust):
        candidate_flags = pd.DataFrame({
            "patient_dominance_flag": ["NO_DOMINANCE"] * len(robust), "technical_flags": ["NONE"] * len(robust),
            "review_flags": ["NONE"] * len(robust)}, index=robust.rmp_id)
        supportive = pairwise_analysis("RMP_candidates_vs_GSE196756", candidate_scores, candidate_spectra,
                                       cohorts["GSE196756"]["scores"].loc[candidate_scores.index],
                                       cohorts["GSE196756"]["spectra"].loc[candidate_scores.index],
                                       candidate_flags, cohorts["GSE196756"]["flags"], 100000)
        supportive, supportive_rbh = add_rbh_and_split(supportive)
        supportive = supportive.rename(columns={"robust_primary_match": "robust_supportive_match"})
        supportive.to_csv(OUT / "pairwise_similarity_RMP_candidates_vs_GSE196756.tsv", sep="\t", index=False)
    else:
        supportive = pd.DataFrame()

    evidence_rows = []
    for row in robust.itertuples():
        tier = ("TIER_2_DOMINANCE_CAUTIOUS_RECURRENT_PROGRAM" if
                "STRONG_PATIENT_DOMINANCE" in (row.patient_dominance_flag_A, row.patient_dominance_flag_B)
                else "TIER_1_CORE_RECURRENT_PROGRAM")
        support = supportive.loc[(supportive.program_A.eq(row.rmp_id)) & supportive.robust_supportive_match]
        if len(support):
            s = support.iloc[0]
            support_id, support_yes = s.program_B, "YES"
            support_comp, support_fdr = s.composite_similarity, s.permutation_FDR
            support_count = int(s.independent_metric_support_count)
            support_dom = s.patient_dominance_flag_B
        else:
            all_s = supportive.loc[supportive.program_A.eq(row.rmp_id)].sort_values("composite_similarity", ascending=False)
            s = all_s.iloc[0]
            support_id, support_yes = s.program_B, "NO"
            support_comp, support_fdr = s.composite_similarity, s.permutation_FDR
            support_count = int(s.independent_metric_support_count)
            support_dom = s.patient_dominance_flag_B
        evidence_rows.append({
            "rmp_id": row.rmp_id, "evidence_tier": tier, "program_A": row.program_A, "program_B": row.program_B,
            "primary_composite_similarity": row.composite_similarity,
            "primary_independent_metric_support_count": int(row.independent_metric_support_count),
            "primary_permutation_FDR": row.permutation_FDR,
            "patient_dominance_flag_A": row.patient_dominance_flag_A,
            "patient_dominance_flag_B": row.patient_dominance_flag_B,
            "supportive_program": support_id, "supportive_composite_similarity": support_comp,
            "supportive_independent_metric_support_count": support_count,
            "supportive_permutation_FDR": support_fdr,
            "supportive_patient_dominance_flag": support_dom,
            "THREE_COHORT_SUPPORT": support_yes,
        })
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_csv(OUT / "RMP_evidence_table.tsv", sep="\t", index=False)

    matched_a = set(evidence.program_A) if len(evidence) else set()
    matched_b = set(evidence.program_B) if len(evidence) else set()
    unmatched_rows = ([{"cohort": "GSE160269", "program_id": p} for p in cohorts["GSE160269"]["scores"].columns if p not in matched_a] +
                      [{"cohort": "GSE188900", "program_id": p} for p in cohorts["GSE188900"]["scores"].columns if p not in matched_b])
    pd.DataFrame(unmatched_rows).to_csv(OUT / "unmatched_primary_programs.tsv", sep="\t", index=False)

    plot_heatmap(primary, "spearman_rho", "spearman_similarity_heatmap", "GSE160269 × GSE188900 Spearman similarity")
    plot_heatmap(primary, "cosine_similarity", "cosine_similarity_heatmap", "GSE160269 × GSE188900 cosine similarity")
    plot_heatmap(primary, "composite_similarity", "composite_similarity_heatmap", "GSE160269 × GSE188900 composite similarity")
    plot_heatmap(primary, "top50_overlap", "top50_overlap_heatmap", "GSE160269 × GSE188900 top-50 overlap", "d")
    plot_network(primary, evidence)

    split_count = int(primary.POTENTIAL_SPLIT_MERGE.sum())
    tier1 = int(evidence.evidence_tier.eq("TIER_1_CORE_RECURRENT_PROGRAM").sum()) if len(evidence) else 0
    tier2 = int(evidence.evidence_tier.eq("TIER_2_DOMINANCE_CAUTIOUS_RECURRENT_PROGRAM").sum()) if len(evidence) else 0
    three = int(evidence.THREE_COHORT_SUPPORT.eq("YES").sum()) if len(evidence) else 0
    primary_status = "PASS" if len(evidence) else "PARTIAL"
    ready = "YES" if len(evidence) else "NO"
    warnings = [
        "USER_STATE_DISCREPANCY: GSE196756 silhouette was stated as 0.866243 in the Phase C request; "
        "the frozen Phase B1 metrics/report contain 0.8626425352705196 (rounded 0.862643). Frozen value used; no cNMF rerun.",
        "No Phase B1 files were modified. No secondary-K analysis was run.",
        "RECOVERED_PRECOMPUTATION_ERROR: the first Phase C attempt stopped before similarity calculation because "
        "pandas Index.eq was unavailable; replaced with equivalent vectorized comparison and reran from verified B1 inputs.",
        "RECOVERED_IDENTIFIER_AUDIT_ERROR: an intermediate supportive-matching attempt stopped because GSE196756 "
        "formal rows use numeric var_names while frozen var['gene_symbol'] contains symbols; explicit frozen-input "
        "var_names-to-gene_symbol mapping was applied before the final complete run.",
        "RECOVERED_VALIDATOR_EXPECTATION_ERROR: the first independent final-table validator hard-coded 13,917 "
        "three-way shared genes; the computed and matrix-verified universe is 13,925. The validator expectation was "
        "corrected; analysis outputs were not changed.",
    ]
    (OUT / "warnings_errors.txt").write_text("WARNINGS\n" + "\n".join(warnings) + "\n\nERRORS\nNONE\n", encoding="utf-8")

    lines = ["# Phase C — Cross-Cohort Program Matching and Recurrent Malignant Program Definition", "",
             "Date: 2026-08-26 (Asia/Shanghai)", "", "## 1. Executive status", "",
             f"**PRIMARY CROSS-COHORT MATCHING: {primary_status}**", "",
             f"The locked technical rule produced **{len(evidence)}** neutral RMP candidates: {tier1} Tier 1 and {tier2} Tier 2. GSE196756 provided supportive evidence for {three}. No biological annotation or prohibited downstream analysis was performed.", "",
             "## 2. Shared gene universe audit", "",
             "Gene identifiers were stripped and uppercased. Duplicate canonical symbols were collapsed by mean for scores and sum for spectra; no symbol was counted twice. Pairwise shared universes were fixed before metric calculation.", "",
             "| Pair | Shared canonical genes |", "|---|---:|"]
    for pair, count in pair_universes.items(): lines.append(f"| {pair.replace('_vs_', ' vs ')} | {count:,} |")
    lines += ["", "Full-gene cNMF refit scores, rather than the cohort-specific 2,000-HVG lists, were used for rank similarity. Top-gene metrics were independently retained to reduce dependence on full-universe scale.", "",
              "## 3. Pairwise similarity methodology", "",
              "Primary discovery comprised exactly 10 × 8 = 80 GSE160269–GSE188900 pairs. Spearman used shared-gene full-refit scores. Cosine used nonnegative spectra restricted to the shared universe, L1-normalized within each program and then L2-normalized. Top-50 overlap/Jaccard/hypergeometric tests were primary; top-100 was fixed sensitivity. Composite was the equal-weight mean of positive Spearman, cosine, and top-50 overlap coefficient.", "",
              "## 4. Permutation/null methodology", "",
              f"A fixed seed ({SEED}) generated {N_PERM:,} gene-label permutations. Each permutation applied one shared gene-identity shuffle to every program on the second side, preserving program score/spectra distributions, gene-universe size, and within-cohort program relationships. Empirical P=(exceedances+1)/({N_PERM}+1); BH FDR was applied separately to Spearman, cosine, composite, and overlap test families.", "",
              "## 5. Primary GSE160269–GSE188900 matching results", "",
              f"Statistically supported edges: {int(primary.statistically_supported_edge.sum())}/80. Robust RBH candidates: {len(evidence)}. Matching used only frozen technical rules and neutral IDs.", "",
              "## 6. Reciprocal-best matches", ""]
    rbh_only = primary.loc[primary.RBH].sort_values("composite_similarity", ascending=False)
    if len(rbh_only):
        lines += ["| Program A | Program B | Composite | Metrics supported | Composite FDR | Robust |", "|---|---|---:|---:|---:|---|"]
        for r in rbh_only.itertuples():
            lines.append(f"| {r.program_A} | {r.program_B} | {r.composite_similarity:.4f} | {r.independent_metric_support_count} | {r.permutation_FDR:.3g} | {'YES' if r.robust_primary_match else 'NO'} |")
    else: lines.append("No reciprocal-best pairs were found.")
    lines += ["", "## 7. Potential split/merge cases", ""]
    splits = primary.loc[primary.POTENTIAL_SPLIT_MERGE]
    if len(splits):
        for r in splits.itertuples(): lines.append(f"- {r.program_A} ↔ {r.program_B} (`POTENTIAL_SPLIT_MERGE`; composite={r.composite_similarity:.4f})")
        lines += ["", "Targeted secondary-K sensitivity is recommended for these relations, but was not run."]
    else: lines.append("None crossed the locked split/merge rule; secondary-K sensitivity is not currently required.")
    lines += ["", "## 8. Evidence-tier assignment", "", "### TIER 1 — CORE RECURRENT PROGRAM candidates", ""]
    t1 = evidence.loc[evidence.evidence_tier.eq("TIER_1_CORE_RECURRENT_PROGRAM")] if len(evidence) else evidence
    lines.append(", ".join(t1.rmp_id) if len(t1) else "NONE")
    lines += ["", "### TIER 2 — DOMINANCE-CAUTIOUS candidates", ""]
    t2 = evidence.loc[evidence.evidence_tier.eq("TIER_2_DOMINANCE_CAUTIOUS_RECURRENT_PROGRAM")] if len(evidence) else evidence
    lines.append(", ".join(t2.rmp_id) if len(t2) else "NONE")
    if len(evidence):
        lines += ["", "| RMP | Tier | GSE160269 member | GSE188900 member | GSE188900 dominance |", "|---|---|---|---|---|"]
        for r in evidence.itertuples(): lines.append(f"| {r.rmp_id} | {r.evidence_tier} | {r.program_A} | {r.program_B} | {r.patient_dominance_flag_B} |")
    lines += ["", "## 9. GSE196756 supportive evidence", "",
              "GSE196756 was tested only against already-created primary candidates. It did not create any RMP. Its P4/sample dominance remains an evidence limitation.", ""]
    if len(evidence):
        lines += ["| RMP | Best supportive program | Composite | FDR | Three-cohort support |", "|---|---|---:|---:|---|"]
        for r in evidence.itertuples(): lines.append(f"| {r.rmp_id} | {r.supportive_program} | {r.supportive_composite_similarity:.4f} | {r.supportive_permutation_FDR:.3g} | {r.THREE_COHORT_SUPPORT} |")
    lines += ["", "## 10. Unmatched programs", ""]
    unmatched = pd.DataFrame(unmatched_rows)
    for cohort in ("GSE160269", "GSE188900"):
        ids = unmatched.loc[unmatched.cohort.eq(cohort), "program_id"].tolist() if len(unmatched) else []
        lines.append(f"- {cohort}: {', '.join(ids) if ids else 'NONE'}")
    lines += ["", "## 11. Warnings/errors", ""]
    lines += [f"- {warning}" for warning in warnings]
    lines += ["", "No unresolved errors remain in the final complete run.", "",
              "## 12. Readiness gate", "", f"`PRIMARY CROSS-COHORT MATCHING: {primary_status}`", "",
              f"`NUMBER OF TIER 1 CORE RMPs: {tier1}`", "", f"`NUMBER OF TIER 2 DOMINANCE-CAUTIOUS RMPs: {tier2}`", "",
              f"`NUMBER WITH THREE-COHORT SUPPORT: {three}`", "", f"`POTENTIAL SPLIT/MERGE CASES: {split_count}`", "",
              f"`SECONDARY-K SENSITIVITY NEEDED: {'YES' if split_count else 'NO'}`", "",
              f"`READY FOR BIOLOGICAL ANNOTATION: {ready}`", "", f"`READY FOR NEXT PHASE: {ready}`", "",
              "Phase C analysis complete.", ""]
    (OUT / "PHASE_C_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    refresh_output_manifest()


if __name__ == "__main__":
    main()
