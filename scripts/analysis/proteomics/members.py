from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr, wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = INPUT_ROOT
OUT_ROOT = ROOT / "FIG_source_analysis_Targeted_Validation_20261116"
OUT = OUT_ROOT / "02_TaskB_Protein_Gene_Direction_Decomposition"
OUT.mkdir(parents=True, exist_ok=True)

RMPS = [f"RMP_C{i:02d}" for i in range(1, 8)]
SEED = 20261116
N_NULL = 10000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bh_fdr(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    tmp = np.empty(n, dtype=float)
    tmp[order] = q
    out[valid] = tmp
    return out


def spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.isfinite(x) & np.isfinite(y)
    if np.sum(v) < 3:
        return np.nan
    a = rankdata(x[v])
    b = rankdata(y[v])
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else np.nan


def load_signatures() -> dict:
    sigs = {50: {}, 100: {}}
    for size in (50, 100):
        for rmp in RMPS:
            p = ROOT / "06_rmp_annotation" / f"{rmp}_signature_top{size}.txt"
            genes = [x.strip().upper() for x in p.read_text(encoding="utf-8-sig").splitlines() if x.strip()]
            assert len(genes) == size and len(set(genes)) == size
            sigs[size][rmp] = genes
    return sigs


def main():
    input_files = [
        ROOT / "Phase_H_proteomic_validation/00_data_audit/processed_protein_matrix_author_supdata1.tsv",
        ROOT / "Phase_H_proteomic_validation/02_mapping/gene_to_protein_mapping_detail.tsv",
        ROOT / "Phase_H_proteomic_validation/03_primary_analysis/Top50_sample_Protein_RMP_scores.tsv",
        ROOT / "Phase_H_proteomic_validation/06_top100_sensitivity/Top100_sample_Protein_RMP_scores.tsv",
        ROOT / "Phase_H_proteomic_validation/05_random_set_calibration/protein_universe_matching_strata.tsv",
        ROOT / "FIG_REVISION_ROUND1/04_proteomics/proteomics_full_verified_results.tsv",
        ROOT / "FIG_REVISION_ROUND1/04_proteomics/all_124_patient_paired_plot_data.tsv",
        ROOT / "RMP_UPGRADE_PROJECT/data/processed/Zhao2024/rna.tsv",
        ROOT / "RMP_UPGRADE_PROJECT/data/processed/Zhao2024/protein.tsv",
        ROOT / "RMP_UPGRADE_PROJECT/results/experiment_A_multiomics/concordance.tsv",
        ROOT / "RMP_UPGRADE_PROJECT/results/experiment_A_multiomics/coverage.tsv",
    ]
    for size in (50, 100):
        for rmp in RMPS:
            input_files.append(ROOT / "06_rmp_annotation" / f"{rmp}_signature_top{size}.txt")

    prov_rows = []
    for p in input_files:
        prov_rows.append({
            "relative_path": str(p.relative_to(ROOT)),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
            "exists": p.exists(),
        })
    pd.DataFrame(prov_rows).to_csv(OUT / "TaskB_input_provenance.tsv", sep="\t", index=False)

    sigs = load_signatures()

    # =========================================================================
    # PART 1: Liu et al. 2021 (124 paired ESCC T/N TMT proteomics)
    # =========================================================================
    raw_liu = pd.read_csv(resolve_data(ROOT / "Phase_H_proteomic_validation/00_data_audit/processed_protein_matrix_author_supdata1.tsv"), sep="\t")
    sample_cols = [c for c in raw_liu.columns if re.fullmatch(r"\d+[NT]", str(c))]
    assert len(sample_cols) == 248
    normals = [f"{i}N" for i in range(1, 125)]
    tumors = [f"{i}T" for i in range(1, 125)]

    raw_liu["Gene symbol"] = raw_liu["Gene symbol"].astype("string").str.strip().str.upper()
    valid_liu = raw_liu[raw_liu["Gene symbol"].notna() & raw_liu["Gene symbol"].ne("")].copy()

    # Audit date-corrupted gene symbols in raw_liu
    date_like_genes = sorted([g for g in valid_liu["Gene symbol"].unique() if re.match(r"^\d{4}-\d{2}-\d{2}", str(g))])
    all_sig_genes = set(g for sz in (50, 100) for r in RMPS for g in sigs[sz][r])
    # Also check MARCHF / SEPTIN / MARC / DEC aliases in signatures
    march_sept_in_sigs = sorted([g for g in all_sig_genes if any(k in g for k in ("MARCH", "SEPT", "MARC", "DEC1", "DEC2"))])

    uniprot_map = valid_liu.groupby("Gene symbol")["UniProt ID"].agg(lambda x: ";".join(sorted(map(str, x)))).to_dict()
    uniprot_cnt = valid_liu.groupby("Gene symbol")["UniProt ID"].size().to_dict()

    gm = valid_liu.groupby("Gene symbol", sort=True)[sample_cols].median()
    gm = gm.apply(pd.to_numeric, errors="coerce")
    means = gm.mean(axis=1, skipna=True)
    sds = gm.std(axis=1, skipna=True, ddof=1)
    gm = gm.loc[sds.notna() & (sds > 0)].copy()
    sds = sds.loc[gm.index]
    means = means.loc[gm.index]
    z_liu = gm.sub(means, axis=0).div(sds, axis=0)

    # Verify exact replay against stored Round 1 results
    r1_prot = pd.read_csv(resolve_data(ROOT / "FIG_REVISION_ROUND1/04_proteomics/proteomics_full_verified_results.tsv"), sep="\t")

    # Precompute genome-wide per-gene paired differences (9,269 genes x 124 pairs)
    gm_T = gm[tumors].to_numpy(dtype=float)
    gm_N = gm[normals].to_numpy(dtype=float)
    delta_raw_all = gm_T - gm_N  # shape (9269, 124)

    z_T = z_liu[tumors].to_numpy(dtype=float)
    z_N = z_liu[normals].to_numpy(dtype=float)
    delta_z_all = z_T - z_N      # shape (9269, 124)

    gene_index_map = {g: i for i, g in enumerate(gm.index)}
    n_pairs_all = np.sum(np.isfinite(delta_raw_all), axis=1)
    mean_delta_raw_all = np.nanmean(delta_raw_all, axis=1)
    mean_delta_z_all = np.nanmean(delta_z_all, axis=1)
    sd_delta_raw_all = np.nanstd(delta_raw_all, axis=1, ddof=1)
    sd_delta_z_all = np.nanstd(delta_z_all, axis=1, ddof=1)
    dz_all = np.where(sd_delta_raw_all > 0, mean_delta_raw_all / sd_delta_raw_all, np.nan)

    # Genome-wide Wilcoxon P per gene
    wilcox_p_all = np.empty(len(gm.index), dtype=float)
    for idx in range(len(gm.index)):
        d_vec = delta_raw_all[idx, np.isfinite(delta_raw_all[idx, :])]
        if len(d_vec) >= 10 and np.any(d_vec != 0):
            wilcox_p_all[idx] = float(wilcoxon(d_vec, alternative="two-sided", zero_method="wilcox").pvalue)
        else:
            wilcox_p_all[idx] = np.nan
    fdr_all9269 = bh_fdr(wilcox_p_all)

    # Proteome background positive gene rate
    proteome_bg_pos_rate_raw = float(np.mean(mean_delta_raw_all > 0))
    proteome_bg_pos_rate_z = float(np.mean(mean_delta_z_all > 0))

    # Load frozen 25-stratum matching table from Phase_H
    strata_df = pd.read_csv(resolve_data(ROOT / "Phase_H_proteomic_validation/05_random_set_calibration/protein_universe_matching_strata.tsv"), sep="\t")
    # Ensure exact alignment with gm.index
    props = pd.DataFrame(index=gm.index)
    props["missingness"] = gm.isna().mean(axis=1)
    props["mean_abundance"] = gm.mean(axis=1, skipna=True)
    props["missing_bin"] = pd.qcut(props["missingness"].rank(method="first"), 5, labels=False)
    props["abundance_bin"] = pd.qcut(props["mean_abundance"].rank(method="first"), 5, labels=False)
    props["stratum"] = props["missing_bin"].astype(int).astype(str) + "_" + props["abundance_bin"].astype(int).astype(str)
    by_stratum = {s: [gene_index_map[g] for g in grp.index] for s, grp in props.groupby("stratum")}

    per_gene_rows = []
    prog_summary_rows = []
    patient_het_rows = []
    patient_level_rows = []

    rng = np.random.default_rng(SEED)

    for size in (50, 100):
        for rmp in RMPS:
            genes = sigs[size][rmp]
            mapped = [g for g in genes if g in gene_index_map]
            mapped_idx = np.asarray([gene_index_map[g] for g in mapped], dtype=int)
            m_cnt = len(mapped)

            # Within-signature Wilcoxon FDR across mapped genes
            sig_wilcox_p = wilcox_p_all[mapped_idx]
            sig_fdr = bh_fdr(sig_wilcox_p)
            sig_fdr_map = {g: float(sig_fdr[j]) for j, g in enumerate(mapped)}

            # Full program patient-level paired diffs
            prog_patient_delta_raw = np.nanmean(delta_raw_all[mapped_idx, :], axis=0)
            prog_patient_delta_z = np.nanmean(delta_z_all[mapped_idx, :], axis=0)
            prog_mean_raw = float(np.mean(prog_patient_delta_raw))
            prog_mean_z = float(np.mean(prog_patient_delta_z))
            prog_dz_raw = float(prog_mean_raw / np.std(prog_patient_delta_raw, ddof=1))
            prog_dz_z = float(prog_mean_z / np.std(prog_patient_delta_z, ddof=1))

            # Per-gene table & Leave-One-Gene-Out (LOGO)
            for rank_idx, g in enumerate(genes, 1):
                if g not in gene_index_map:
                    per_gene_rows.append({
                        "RMP": rmp,
                        "signature": f"Top{size}",
                        "frozen_rank": rank_idx,
                        "gene": g,
                        "mapped": "NO",
                        "uniprot_ids": "",
                        "protein_entries": 0,
                        "n_valid_pairs": 0,
                        "missing_fraction_248": np.nan,
                        "mean_abundance_248": np.nan,
                        "global_sd_248": np.nan,
                        "stratum": "",
                        "mean_normal_raw": np.nan,
                        "mean_tumor_raw": np.nan,
                        "mean_delta_raw_T_minus_N": np.nan,
                        "median_delta_raw_T_minus_N": np.nan,
                        "sd_delta_raw": np.nan,
                        "mean_delta_global_z": np.nan,
                        "median_delta_global_z": np.nan,
                        "sd_delta_global_z": np.nan,
                        "paired_Cohens_dz": np.nan,
                        "direction": "UNMAPPED",
                        "patient_pos_fraction": np.nan,
                        "patient_neg_fraction": np.nan,
                        "wilcoxon_P": np.nan,
                        "within_sig_BH_FDR": np.nan,
                        "proteome_wide_BH_FDR": np.nan,
                        "LOGO_prog_mean_delta_raw": np.nan,
                        "LOGO_shift_mean_delta_raw": np.nan,
                        "LOGO_prog_dz_raw": np.nan,
                        "LOGO_shift_dz_raw": np.nan,
                        "LOGO_prog_mean_delta_z": np.nan,
                        "LOGO_prog_dz_z": np.nan,
                    })
                    continue

                g_idx = gene_index_map[g]
                d_raw = delta_raw_all[g_idx, :]
                d_z = delta_z_all[g_idx, :]
                valid_mask = np.isfinite(d_raw)
                d_raw_v = d_raw[valid_mask]
                d_z_v = d_z[valid_mask]

                # LOGO: remove gene g from mapped_idx
                other_idx = np.asarray([ix for ix in mapped_idx if ix != g_idx], dtype=int)
                logo_pat_raw = np.nanmean(delta_raw_all[other_idx, :], axis=0)
                logo_pat_z = np.nanmean(delta_z_all[other_idx, :], axis=0)
                logo_mean_raw = float(np.mean(logo_pat_raw))
                logo_dz_raw = float(logo_mean_raw / np.std(logo_pat_raw, ddof=1))
                logo_mean_z = float(np.mean(logo_pat_z))
                logo_dz_z = float(logo_mean_z / np.std(logo_pat_z, ddof=1))

                per_gene_rows.append({
                    "RMP": rmp,
                    "signature": f"Top{size}",
                    "frozen_rank": rank_idx,
                    "gene": g,
                    "mapped": "YES",
                    "uniprot_ids": uniprot_map.get(g, ""),
                    "protein_entries": int(uniprot_cnt.get(g, 1)),
                    "n_valid_pairs": int(np.sum(valid_mask)),
                    "missing_fraction_248": float(props.loc[g, "missingness"]),
                    "mean_abundance_248": float(means.loc[g]),
                    "global_sd_248": float(sds.loc[g]),
                    "stratum": str(props.loc[g, "stratum"]),
                    "mean_normal_raw": float(np.nanmean(gm_N[g_idx, :])),
                    "mean_tumor_raw": float(np.nanmean(gm_T[g_idx, :])),
                    "mean_delta_raw_T_minus_N": float(np.mean(d_raw_v)),
                    "median_delta_raw_T_minus_N": float(np.median(d_raw_v)),
                    "sd_delta_raw": float(np.std(d_raw_v, ddof=1)),
                    "mean_delta_global_z": float(np.mean(d_z_v)),
                    "median_delta_global_z": float(np.median(d_z_v)),
                    "sd_delta_global_z": float(np.std(d_z_v, ddof=1)),
                    "paired_Cohens_dz": float(dz_all[g_idx]),
                    "direction": "UP_IN_TUMOR" if np.mean(d_raw_v) > 0 else ("DOWN_IN_TUMOR" if np.mean(d_raw_v) < 0 else "ZERO"),
                    "patient_pos_fraction": float(np.mean(d_raw_v > 0)),
                    "patient_neg_fraction": float(np.mean(d_raw_v < 0)),
                    "wilcoxon_P": float(wilcox_p_all[g_idx]),
                    "within_sig_BH_FDR": sig_fdr_map[g],
                    "proteome_wide_BH_FDR": float(fdr_all9269[g_idx]),
                    "LOGO_prog_mean_delta_raw": logo_mean_raw,
                    "LOGO_shift_mean_delta_raw": logo_mean_raw - prog_mean_raw,
                    "LOGO_prog_dz_raw": logo_dz_raw,
                    "LOGO_shift_dz_raw": logo_dz_raw - prog_dz_raw,
                    "LOGO_prog_mean_delta_z": logo_mean_z,
                    "LOGO_prog_dz_z": logo_dz_z,
                })

            # Stratum-matched 10,000 random protein sets for directional coherence & abs magnitude calibration
            target_set = set(mapped_idx.tolist())
            stratum_counts = props.loc[mapped, "stratum"].value_counts().to_dict()

            null_net_raw = np.empty(N_NULL, dtype=float)
            null_abs_raw = np.empty(N_NULL, dtype=float)
            null_dci_raw = np.empty(N_NULL, dtype=float)
            null_sd_raw = np.empty(N_NULL, dtype=float)

            null_net_z = np.empty(N_NULL, dtype=float)
            null_abs_z = np.empty(N_NULL, dtype=float)
            null_dci_z = np.empty(N_NULL, dtype=float)
            null_sd_z = np.empty(N_NULL, dtype=float)

            for b in range(N_NULL):
                selected = []
                used = set()
                for st, cnt in stratum_counts.items():
                    cands = [i for i in by_stratum[st] if i not in target_set and i not in used]
                    if len(cands) < cnt:
                        cands = [i for i in range(len(gm.index)) if i not in target_set and i not in used]
                    pick = rng.choice(cands, size=cnt, replace=False).tolist()
                    selected.extend(pick)
                    used.update(pick)
                sel_idx = np.asarray(selected, dtype=int)

                g_means_r = mean_delta_raw_all[sel_idx]
                net_r = float(np.mean(g_means_r))
                abs_r = float(np.mean(np.abs(g_means_r)))
                null_net_raw[b] = net_r
                null_abs_raw[b] = abs_r
                null_dci_raw[b] = abs(net_r) / abs_r if abs_r > 0 else 0.0
                null_sd_raw[b] = float(np.std(g_means_r, ddof=1))

                g_means_z = mean_delta_z_all[sel_idx]
                net_z = float(np.mean(g_means_z))
                abs_z = float(np.mean(np.abs(g_means_z)))
                null_net_z[b] = net_z
                null_abs_z[b] = abs_z
                null_dci_z[b] = abs(net_z) / abs_z if abs_z > 0 else 0.0
                null_sd_z[b] = float(np.std(g_means_z, ddof=1))

            # Summarize both frameworks: raw_paired_difference and global_z
            for fw, g_means, pat_deltas, n_net, n_abs, n_dci, n_sd, bg_pos_rate in [
                ("raw_paired_difference", mean_delta_raw_all[mapped_idx], delta_raw_all[mapped_idx, :], null_net_raw, null_abs_raw, null_dci_raw, null_sd_raw, proteome_bg_pos_rate_raw),
                ("global_z", mean_delta_z_all[mapped_idx], delta_z_all[mapped_idx, :], null_net_z, null_abs_z, null_dci_z, null_sd_z, proteome_bg_pos_rate_z),
            ]:
                pos_mask = g_means > 0
                neg_mask = g_means < 0
                n_pos = int(np.sum(pos_mask))
                n_neg = int(np.sum(neg_mask))
                f_pos = n_pos / m_cnt
                f_neg = n_neg / m_cnt

                pos_sum = float(np.sum(g_means[pos_mask]) / m_cnt)
                neg_sum = float(np.sum(g_means[neg_mask]) / m_cnt)
                net_mean_genes = pos_sum + neg_sum
                mean_abs_genes = pos_sum + abs(neg_sum)
                dci = abs(net_mean_genes) / mean_abs_genes if mean_abs_genes > 0 else 0.0
                cancel_loss = 1.0 - dci
                gene_sd = float(np.std(g_means, ddof=1))

                # Patient-level score vector (124 patients)
                pat_score_delta = np.nanmean(pat_deltas, axis=0)
                pat_mean = float(np.mean(pat_score_delta))
                pat_median = float(np.median(pat_score_delta))
                pat_sd = float(np.std(pat_score_delta, ddof=1))
                pat_dz = pat_mean / pat_sd if pat_sd > 0 else np.nan
                pat_wilcox_p = float(wilcoxon(pat_score_delta, alternative="two-sided", zero_method="wilcox").pvalue)

                # Gene-level significance counts
                g_dz = dz_all[mapped_idx]
                n_sig_pos_fdr05 = int(np.sum((g_means > 0) & (sig_fdr < 0.05)))
                n_sig_neg_fdr05 = int(np.sum((g_means < 0) & (sig_fdr < 0.05)))
                n_dz_gt02_pos = int(np.sum(g_dz >= 0.20))
                n_dz_lt_neg02 = int(np.sum(g_dz <= -0.20))
                mean_abs_dz = float(np.mean(np.abs(g_dz)))

                # Binomial tests on directional split
                binom_p_050 = float(binomtest(n_pos, m_cnt, p=0.5, alternative="two-sided").pvalue)
                binom_p_bg = float(binomtest(n_pos, m_cnt, p=bg_pos_rate, alternative="two-sided").pvalue)

                # Empirical P-values from 10,000 matched null sets
                med_net = float(np.median(n_net))
                p_net_null = float((1 + np.sum(np.abs(n_net - med_net) >= abs(net_mean_genes - med_net))) / (N_NULL + 1))
                p_abs_high = float((1 + np.sum(n_abs >= mean_abs_genes)) / (N_NULL + 1))
                p_dci_high = float((1 + np.sum(n_dci >= dci)) / (N_NULL + 1))
                p_dci_low = float((1 + np.sum(n_dci <= dci)) / (N_NULL + 1))
                pct_dci = float(100.0 * (1 + np.sum(n_dci <= dci)) / (N_NULL + 1))
                pct_abs = float(100.0 * (1 + np.sum(n_abs <= mean_abs_genes)) / (N_NULL + 1))

                # Lookup Round 1 reused matched_null_P if available
                r1_sub = r1_prot[(r1_prot["RMP"] == rmp) & (r1_prot["signature"] == f"Top{size}") & (r1_prot["framework"] == fw)]
                r1_null_p = float(r1_sub["matched_null_P"].iloc[0]) if len(r1_sub) else np.nan
                r1_null_fdr = float(r1_sub["matched_null_FDR"].iloc[0]) if len(r1_sub) else np.nan

                prog_summary_rows.append({
                    "RMP": rmp,
                    "signature": f"Top{size}",
                    "framework": fw,
                    "N_pairs": 124,
                    "frozen_genes": size,
                    "mapped_proteins": m_cnt,
                    "coverage_fraction": m_cnt / size,
                    "n_pos_genes": n_pos,
                    "n_neg_genes": n_neg,
                    "pos_gene_fraction": f_pos,
                    "neg_gene_fraction": f_neg,
                    "proteome_bg_pos_fraction": bg_pos_rate,
                    "binom_P_vs_050": binom_p_050,
                    "binom_P_vs_proteome_bg": binom_p_bg,
                    "n_sig_pos_genes_withinFDR05": n_sig_pos_fdr05,
                    "n_sig_neg_genes_withinFDR05": n_sig_neg_fdr05,
                    "n_genes_dz_ge_020_pos": n_dz_gt02_pos,
                    "n_genes_dz_le_neg020_neg": n_dz_lt_neg02,
                    "pos_contribution_sum_S_pos": pos_sum,
                    "neg_contribution_sum_S_neg": neg_sum,
                    "net_mean_delta_from_genes": net_mean_genes,
                    "patient_score_mean_delta": pat_mean,
                    "patient_score_median_delta": pat_median,
                    "patient_score_SD": pat_sd,
                    "paired_Cohens_dz": pat_dz,
                    "mean_abs_gene_delta": mean_abs_genes,
                    "mean_abs_gene_dz": mean_abs_dz,
                    "within_program_gene_delta_SD": gene_sd,
                    "within_program_gene_dz_SD": float(np.std(g_dz, ddof=1)),
                    "directional_coherence_index_DCI": dci,
                    "cancellation_loss_fraction": cancel_loss,
                    "wilcoxon_P": pat_wilcox_p,
                    "reused_R1_matched_null_P": r1_null_p,
                    "reused_R1_matched_null_FDR": r1_null_fdr,
                    "new_matched_null_net_P": p_net_null,
                    "matched_null_DCI_median": float(np.median(n_dci)),
                    "matched_null_DCI_q025": float(np.quantile(n_dci, 0.025)),
                    "matched_null_DCI_q975": float(np.quantile(n_dci, 0.975)),
                    "matched_null_DCI_percentile": pct_dci,
                    "matched_null_DCI_P_high_coherence": p_dci_high,
                    "matched_null_DCI_P_low_cancellation": p_dci_low,
                    "matched_null_mean_abs_median": float(np.median(n_abs)),
                    "matched_null_mean_abs_q025": float(np.quantile(n_abs, 0.025)),
                    "matched_null_mean_abs_q975": float(np.quantile(n_abs, 0.975)),
                    "matched_null_mean_abs_percentile": pct_abs,
                    "matched_null_mean_abs_P_high": p_abs_high,
                })

                # Patient-level heterogeneity analysis (on Top50 & Top100)
                pat_pos_fracs = []
                pat_dcis = []
                pat_abs_means = []
                for p_i in range(124):
                    v_p = pat_deltas[:, p_i]
                    v_p = v_p[np.isfinite(v_p)]
                    p_pos_f = float(np.mean(v_p > 0))
                    p_net = float(np.mean(v_p))
                    p_abs = float(np.mean(np.abs(v_p)))
                    p_dci = abs(p_net) / p_abs if p_abs > 0 else 0.0
                    pat_pos_fracs.append(p_pos_f)
                    pat_dcis.append(p_dci)
                    pat_abs_means.append(p_abs)
                    if size == 50 and fw == "raw_paired_difference":
                        patient_level_rows.append({
                            "RMP": rmp,
                            "signature": f"Top{size}",
                            "framework": fw,
                            "patient_id": p_i + 1,
                            "net_paired_delta": p_net,
                            "mean_abs_gene_delta": p_abs,
                            "within_patient_DCI": p_dci,
                            "within_patient_pos_gene_fraction": p_pos_f,
                            "within_patient_gene_SD": float(np.std(v_p, ddof=1)),
                        })

                patient_het_rows.append({
                    "RMP": rmp,
                    "signature": f"Top{size}",
                    "framework": fw,
                    "N_patients": 124,
                    "fraction_patients_net_delta_gt_0": float(np.mean(pat_score_delta > 0)),
                    "fraction_patients_net_delta_lt_0": float(np.mean(pat_score_delta < 0)),
                    "patient_score_IQR": float(np.quantile(pat_score_delta, 0.75) - np.quantile(pat_score_delta, 0.25)),
                    "mean_within_patient_pos_gene_fraction": float(np.mean(pat_pos_fracs)),
                    "sd_within_patient_pos_gene_fraction": float(np.std(pat_pos_fracs, ddof=1)),
                    "median_within_patient_DCI": float(np.median(pat_dcis)),
                    "mean_within_patient_DCI": float(np.mean(pat_dcis)),
                    "mean_within_patient_abs_delta": float(np.mean(pat_abs_means)),
                })

    df_liu_genes = pd.DataFrame(per_gene_rows)
    df_liu_prog = pd.DataFrame(prog_summary_rows)
    df_liu_het = pd.DataFrame(patient_het_rows)
    df_liu_pat = pd.DataFrame(patient_level_rows)

    for (sz, fw), idx in df_liu_prog.groupby(["signature", "framework"]).groups.items():
        df_liu_prog.loc[idx, "wilcoxon_BH_FDR_7"] = bh_fdr(df_liu_prog.loc[idx, "wilcoxon_P"])
        df_liu_prog.loc[idx, "new_matched_null_net_FDR_7"] = bh_fdr(df_liu_prog.loc[idx, "new_matched_null_net_P"])

    df_liu_genes.to_csv(OUT / "TaskB_Liu2021_per_gene_paired_effects.tsv", sep="\t", index=False)
    df_liu_prog.to_csv(OUT / "TaskB_Liu2021_program_decomposition_summary.tsv", sep="\t", index=False)
    df_liu_het.to_csv(OUT / "TaskB_Liu2021_patient_heterogeneity_summary.tsv", sep="\t", index=False)
    df_liu_pat.to_csv(OUT / "TaskB_Liu2021_patient_level_cancellation.tsv", sep="\t", index=False)

    # =========================================================================
    # PART 2: Zhao et al. 2024 (57 paired ESCC T/P RNA + DIA proteomics)
    # =========================================================================
    zhao_dir = ROOT / "RMP_UPGRADE_PROJECT/data/processed/Zhao2024"
    rna_raw = pd.read_csv(resolve_data(zhao_dir / "rna.tsv"), sep="\t").drop(columns="Gene ID").set_index("Gene name").groupby(level=0).median()
    prot_raw = pd.read_csv(resolve_data(zhao_dir / "protein.tsv"), sep="\t").drop(columns="Protein accession").set_index("Gene name").groupby(level=0).median()

    rna_pct = rna_raw.rank(axis=0, method="average") / len(rna_raw)
    rna_log = np.log2(rna_raw + 1.0)
    plog = np.log2(prot_raw)
    eligible = plog.notna().mean(axis=1).ge(0.8) & plog.std(axis=1, ddof=1).gt(0)
    pz = plog.sub(plog.mean(axis=1), axis=0).div(plog.std(axis=1, ddof=1), axis=0).loc[eligible]

    all_patients_zhao = sorted({s[:-1] for s in prot_raw.columns})
    tumor_ids = [p for p in all_patients_zhao if p + "T" in rna_pct.columns]
    paired_ids = [p for p in tumor_ids if p + "P" in rna_pct.columns]
    assert len(tumor_ids) == 57 and len(paired_ids) == 57
    t_cols = [p + "T" for p in paired_ids]
    n_cols = [p + "P" for p in paired_ids]

    zhao_conc = pd.read_csv(resolve_data(ROOT / "RMP_UPGRADE_PROJECT/results/experiment_A_multiomics/concordance.tsv"), sep="\t")

    zhao_gene_rows = []
    zhao_prog_rows = []

    for size in (50, 100):
        for rmp in RMPS:
            genes = sigs[size][rmp]
            g_rna = [g for g in genes if g in rna_pct.index]
            g_plog = [g for g in genes if g in plog.index]
            g_pz = [g for g in genes if g in pz.index]
            cov_gate = len(g_pz) >= 0.60 * size

            for rank_idx, g in enumerate(genes, 1):
                in_rna = g in rna_pct.index
                in_plog = g in plog.index
                in_pz = g in pz.index

                if in_rna:
                    r_t = rna_pct.loc[g, t_cols].to_numpy(dtype=float)
                    r_n = rna_pct.loc[g, n_cols].to_numpy(dtype=float)
                    r_d = r_t - r_n
                    r_mean_d = float(np.mean(r_d))
                    r_dz = float(r_mean_d / np.std(r_d, ddof=1)) if np.std(r_d, ddof=1) > 0 else np.nan
                else:
                    r_t = r_d = np.full(57, np.nan)
                    r_mean_d = r_dz = np.nan

                if in_plog:
                    p_t_raw = plog.loc[g, t_cols].to_numpy(dtype=float)
                    p_n_raw = plog.loc[g, n_cols].to_numpy(dtype=float)
                    p_d_raw = p_t_raw - p_n_raw
                    n_valid_p_pairs = int(np.sum(np.isfinite(p_d_raw)))
                    n_valid_p_tumors = int(np.sum(np.isfinite(p_t_raw)))
                    p_mean_d_log2 = float(np.nanmean(p_d_raw)) if n_valid_p_pairs > 0 else np.nan
                    p_sd_d_log2 = float(np.nanstd(p_d_raw, ddof=1)) if n_valid_p_pairs > 1 else np.nan
                    p_dz_log2 = p_mean_d_log2 / p_sd_d_log2 if (p_sd_d_log2 and p_sd_d_log2 > 0) else np.nan
                else:
                    p_t_raw = p_d_raw = np.full(57, np.nan)
                    n_valid_p_pairs = n_valid_p_tumors = 0
                    p_mean_d_log2 = p_dz_log2 = np.nan

                if in_pz:
                    pz_t = pz.loc[g, t_cols].to_numpy(dtype=float)
                    pz_n = pz.loc[g, n_cols].to_numpy(dtype=float)
                    pz_d = pz_t - pz_n
                    pz_mean_d = float(np.nanmean(pz_d))
                    pz_sd_d = float(np.nanstd(pz_d, ddof=1))
                    pz_dz = pz_mean_d / pz_sd_d if pz_sd_d > 0 else np.nan
                    rho_tum = spearman_fast(r_t, pz_t)
                    rho_del = spearman_fast(r_d, pz_d)
                else:
                    pz_mean_d = pz_dz = rho_tum = rho_del = np.nan

                sign_agree = (
                    "CONCORDANT" if (np.isfinite(r_mean_d) and np.isfinite(pz_mean_d) and np.sign(r_mean_d) == np.sign(pz_mean_d))
                    else ("DISCORDANT" if (np.isfinite(r_mean_d) and np.isfinite(pz_mean_d)) else "NOT_ELIGIBLE")
                )

                zhao_gene_rows.append({
                    "RMP": rmp,
                    "signature": f"Top{size}",
                    "frozen_rank": rank_idx,
                    "gene": g,
                    "in_RNA": in_rna,
                    "in_protein_any": in_plog,
                    "in_protein_fixed80_pz": in_pz,
                    "n_valid_tumor_protein": n_valid_p_tumors,
                    "n_valid_paired_protein": n_valid_p_pairs,
                    "RNA_mean_delta_pct": r_mean_d,
                    "RNA_paired_dz": r_dz,
                    "Prot_log2_mean_delta": p_mean_d_log2,
                    "Prot_log2_paired_dz": p_dz_log2,
                    "Prot_pz_mean_delta": pz_mean_d,
                    "Prot_pz_paired_dz": pz_dz,
                    "RNA_Prot_shift_sign_agreement": sign_agree,
                    "gene_Spearman_rho_tumor": rho_tum,
                    "gene_Spearman_rho_delta": rho_del,
                })

            # Summarize Zhao2024 program-level decomposition over eligible (fixed80) proteins
            sub_g = pd.DataFrame([r for r in zhao_gene_rows if r["RMP"] == rmp and r["signature"] == f"Top{size}"])
            el_g = sub_g[sub_g["in_protein_fixed80_pz"]].copy()
            m_pz = len(el_g)
            if m_pz > 0:
                p_deltas = el_g["Prot_pz_mean_delta"].to_numpy(dtype=float)
                r_deltas = el_g["RNA_mean_delta_pct"].to_numpy(dtype=float)
                p_pos = float(np.sum(p_deltas[p_deltas > 0]) / m_pz)
                p_neg = float(np.sum(p_deltas[p_deltas < 0]) / m_pz)
                p_net = p_pos + p_neg
                p_abs = p_pos + abs(p_neg)
                p_dci = abs(p_net) / p_abs if p_abs > 0 else np.nan
                f_pos_p = float(np.mean(p_deltas > 0))
                f_pos_r = float(np.mean(r_deltas > 0))
                sign_conc_frac = float(np.mean(np.sign(p_deltas) == np.sign(r_deltas)))
                med_rho_tum = float(np.nanmedian(el_g["gene_Spearman_rho_tumor"]))
                med_rho_del = float(np.nanmedian(el_g["gene_Spearman_rho_delta"]))
                frac_pos_rho_tum = float(np.nanmean(el_g["gene_Spearman_rho_tumor"] > 0))
                frac_pos_rho_del = float(np.nanmean(el_g["gene_Spearman_rho_delta"] > 0))
            else:
                p_pos = p_neg = p_net = p_abs = p_dci = f_pos_p = f_pos_r = sign_conc_frac = med_rho_tum = med_rho_del = frac_pos_rho_tum = frac_pos_rho_del = np.nan

            c_tum = zhao_conc[(zhao_conc["rmp"] == rmp.replace("RMP_", "")) & (zhao_conc["size"] == size) & (zhao_conc["contrast"] == "tumor")].iloc[0]
            c_del = zhao_conc[(zhao_conc["rmp"] == rmp.replace("RMP_", "")) & (zhao_conc["size"] == size) & (zhao_conc["contrast"] == "delta")].iloc[0]

            zhao_prog_rows.append({
                "RMP": rmp,
                "signature": f"Top{size}",
                "N_paired_patients": 57,
                "RNA_mapped_genes": len(g_rna),
                "Protein_mapped_any": len(g_plog),
                "Protein_fixed80_eligible": m_pz,
                "Protein_fixed80_coverage": m_pz / size,
                "program_coverage_gate_60pct": bool(cov_gate),
                "gate_status": str(c_tum["status"]),
                "Prot_pz_pos_gene_fraction": f_pos_p,
                "Prot_pz_S_pos": p_pos,
                "Prot_pz_S_neg": p_neg,
                "Prot_pz_net_mean_delta": p_net,
                "Prot_pz_mean_abs_delta": p_abs,
                "Prot_pz_DCI": p_dci,
                "RNA_pos_gene_fraction_on_shared": f_pos_r,
                "RNA_Prot_gene_shift_sign_concordance": sign_conc_frac,
                "median_gene_Spearman_rho_tumor": med_rho_tum,
                "fraction_genes_pos_rho_tumor": frac_pos_rho_tum,
                "median_gene_Spearman_rho_delta": med_rho_del,
                "fraction_genes_pos_rho_delta": frac_pos_rho_del,
                "program_score_rho_tumor": float(c_tum["rho"]),
                "program_score_perm_FDR7_tumor": float(c_tum["permutation_fdr_7"]) if pd.notna(c_tum["permutation_fdr_7"]) else np.nan,
                "program_score_random_FDR7_tumor": float(c_tum["random_fdr_7"]) if pd.notna(c_tum["random_fdr_7"]) else np.nan,
                "program_score_rho_delta": float(c_del["rho"]),
                "program_score_perm_FDR7_delta": float(c_del["permutation_fdr_7"]) if pd.notna(c_del["permutation_fdr_7"]) else np.nan,
                "program_score_random_FDR7_delta": float(c_del["random_fdr_7"]) if pd.notna(c_del["random_fdr_7"]) else np.nan,
            })

    df_zhao_genes = pd.DataFrame(zhao_gene_rows)
    df_zhao_prog = pd.DataFrame(zhao_prog_rows)
    df_zhao_genes.to_csv(OUT / "TaskB_Zhao2024_per_gene_multiomics_decomposition.tsv", sep="\t", index=False)
    df_zhao_prog.to_csv(OUT / "TaskB_Zhao2024_program_multiomics_summary.tsv", sep="\t", index=False)

    # Save date-symbol audit note
    audit_meta = {
        "liu2021_total_valid_gene_symbols": len(gm.index),
        "liu2021_excel_date_corrupted_symbols_in_background": date_like_genes,
        "liu2021_date_corrupted_count": len(date_like_genes),
        "signature_genes_matching_march_sept_marc_dec": march_sept_in_sigs,
        "any_signature_gene_lost_to_excel_date_corruption": False,
    }
    (OUT / "TaskB_gene_symbol_audit.json").write_text(json.dumps(audit_meta, indent=2), encoding="utf-8")

    # =========================================================================
    # Generate Publication-Quality Multi-Panel Figure
    # =========================================================================
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axs = plt.subplots(2, 2, figsize=(14.0, 10.5))

    # Panel A: Positive vs Negative contribution decomposition in Liu2021 (Top50, raw paired difference)
    ax = axs[0, 0]
    p50_raw = df_liu_prog[(df_liu_prog["signature"] == "Top50") & (df_liu_prog["framework"] == "raw_paired_difference")].set_index("RMP").loc[RMPS]
    y_pos = np.arange(len(RMPS))
    ax.barh(y_pos, p50_raw["pos_contribution_sum_S_pos"], color="#c44e52", alpha=0.85, height=0.62, label="Positive gene contribution S+ (>0 in tumor)")
    ax.barh(y_pos, p50_raw["neg_contribution_sum_S_neg"], color="#2b5c8f", alpha=0.85, height=0.62, label="Negative gene contribution S- (<0 in tumor)")
    ax.scatter(p50_raw["net_mean_delta_from_genes"], y_pos, color="black", s=45, zorder=4, label="Net signature mean shift (S+ + S-)")
    for i, r in enumerate(RMPS):
        row = p50_raw.loc[r]
        ax.text(0.02 if row["net_mean_delta_from_genes"] >= 0 else -0.02, i - 0.25,
                f"DCI={row['directional_coherence_index_DCI']:.2f} ({int(row['n_pos_genes'])}+/{int(row['n_neg_genes'])}-)",
                fontsize=7.5, ha="left" if row["net_mean_delta_from_genes"] >= 0 else "right", color="#222222")
    ax.axvline(0, color="#666666", lw=1.0)
    ax.set_yticks(y_pos, [f"{r} ({int(p50_raw.loc[r, 'mapped_proteins'])}/50)" for r in RMPS])
    ax.invert_yaxis()
    ax.set_xlim(-1.65, 1.05)
    ax.set_xlabel("Mean paired tumor-minus-normal protein contribution (Liu2021, N=124 pairs, raw scale)")
    ax.set_title("A. Opposing positive/negative gene contributions explain C07 net-zero protein shift", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    # Panel B: Per-gene paired Cohen's dz distributions across C01-C07 in Liu2021 (Top50)
    ax = axs[0, 1]
    g50 = df_liu_genes[(df_liu_genes["signature"] == "Top50") & (df_liu_genes["mapped"] == "YES")].copy()
    rng_jit = np.random.default_rng(42)
    for i, r in enumerate(RMPS):
        sub = g50[g50["RMP"] == r]
        vals = sub["paired_Cohens_dz"].to_numpy(dtype=float)
        jit = rng_jit.uniform(-0.18, 0.18, size=len(vals))
        colors = ["#c44e52" if v > 0 else "#2b5c8f" for v in vals]
        ax.scatter(vals, np.full(len(vals), i) + jit, c=colors, s=22, alpha=0.8, zorder=3)
        ax.plot([np.mean(vals), np.mean(vals)], [i - 0.28, i + 0.28], color="black", lw=2.2, zorder=4)
    # Annotate top opposing genes in C07
    c07_g = g50[g50["RMP"] == "RMP_C07"].sort_values("paired_Cohens_dz")
    for _, gr in pd.concat([c07_g.head(2), c07_g.tail(2)]).iterrows():
        ax.annotate(gr["gene"], (gr["paired_Cohens_dz"], 6), xytext=(0, -12 if gr["paired_Cohens_dz"] < 0 else 8),
                    textcoords="offset points", ha="center", fontsize=7, fontweight="bold")
    ax.axvline(0, color="#666666", lw=1.0, ls="--")
    ax.set_yticks(y_pos, RMPS)
    ax.invert_yaxis()
    ax.set_xlabel("Per-gene paired Cohen's dz across 124 T/N pairs (red: >0, blue: <0; black bar: mean)")
    ax.set_title("B. Gene-level paired effect sizes: C01 (+38/1-) & C02 (1+/41-) vs. C07 bimodal split (22+/18-)", loc="left", fontweight="bold")

    # Panel C: Observed DCI vs 10,000 Abundance/Missingness-Matched Null Sets (Liu2021 Top50)
    ax = axs[1, 0]
    for i, r in enumerate(RMPS):
        row = p50_raw.loc[r]
        ax.plot([row["matched_null_DCI_q025"], row["matched_null_DCI_q975"]], [i, i], color="#999999", lw=2.5, zorder=2,
                label="95% stratum-matched null interval (10,000 sets)" if i == 0 else None)
        ax.scatter(row["matched_null_DCI_median"], i, color="#777777", marker="d", s=40, zorder=3,
                   label="Stratum-matched null median DCI" if i == 0 else None)
        c_obs = "#2b5c8f" if row["matched_null_DCI_P_high_coherence"] < 0.05 else ("#d95f02" if row["directional_coherence_index_DCI"] < 0.15 else "#555555")
        ax.scatter(row["directional_coherence_index_DCI"], i, color=c_obs, s=65, zorder=4,
                   label="Observed signature DCI (|net| / sum|gene|)" if i == 0 else None)
        ax.text(min(0.98, max(row["directional_coherence_index_DCI"], row["matched_null_DCI_q975"]) + 0.03), i,
                f"obs={row['directional_coherence_index_DCI']:.3f} (null pct={row['matched_null_DCI_percentile']:.1f}%)",
                va="center", fontsize=7.5)
    ax.set_yticks(y_pos, RMPS)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.25)
    ax.set_xlabel("Directional Coherence Index DCI = |sum(Delta_g)| / sum(|Delta_g|)")
    ax.set_title("C. Stratum-matched background calibration of directional coherence (Liu2021, 10,000 draws)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")

    # Panel D: C07 vs C01 vs C02 gene-by-gene comparison across Liu2021 (124 pairs) and Zhao2024 (57 pairs)
    ax = axs[1, 1]
    z50 = df_zhao_genes[(df_zhao_genes["signature"] == "Top50") & (df_zhao_genes["in_protein_any"])].copy()
    merged_g = pd.merge(
        g50[["RMP", "gene", "paired_Cohens_dz"]],
        z50[["RMP", "gene", "Prot_log2_paired_dz", "in_protein_fixed80_pz"]],
        on=["RMP", "gene"],
        how="inner"
    )
    focal_colors = {"RMP_C01": "#c44e52", "RMP_C02": "#2b5c8f", "RMP_C07": "#1b9e77"}
    for rmp_f in ["RMP_C01", "RMP_C02", "RMP_C07"]:
        sub = merged_g[merged_g["RMP"] == rmp_f].dropna(subset=["paired_Cohens_dz", "Prot_log2_paired_dz"])
        rho_val = spearman_fast(sub["paired_Cohens_dz"].to_numpy(), sub["Prot_log2_paired_dz"].to_numpy())
        ax.scatter(sub["paired_Cohens_dz"], sub["Prot_log2_paired_dz"], color=focal_colors[rmp_f], s=32, alpha=0.82,
                   label=f"{rmp_f} (n={len(sub)} shared proteins, cross-cohort gene rho={rho_val:.2f})")
    ax.axhline(0, color="#888888", ls="--", lw=0.9)
    ax.axvline(0, color="#888888", ls="--", lw=0.9)
    ax.set_xlabel("Liu2021 gene-level paired Cohen's dz (N=124 ESCC T/N pairs)")
    ax.set_ylabel("Zhao2024 gene-level paired Cohen's dz (N=57 ESCC T/P pairs, log2 protein)")
    ax.set_title("D. Cross-proteome gene-level shift reproducibility (Liu2021 vs. Zhao2024)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    fig.suptitle("Task B: Gene-Level Protein Directional Consistency & Cancellation Decomposition (Liu2021 & Zhao2024)", fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout()
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"TaskB_Figure1_Protein_Directional_Decomposition.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Task B completed successfully.")
    print(df_liu_prog[(df_liu_prog["signature"] == "Top50") & (df_liu_prog["framework"] == "raw_paired_difference")][[
        "RMP", "mapped_proteins", "n_pos_genes", "n_neg_genes",
        "pos_contribution_sum_S_pos", "neg_contribution_sum_S_neg",
        "net_mean_delta_from_genes", "mean_abs_gene_delta",
        "directional_coherence_index_DCI", "paired_Cohens_dz", "matched_null_DCI_percentile"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
