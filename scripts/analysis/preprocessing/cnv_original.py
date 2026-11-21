#!/usr/bin/env python
"""Locked Phase A epithelial/CNV preparation for GSE196756; no cNMF."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import argparse
import gzip
import re
from pathlib import Path

import anndata as ad
import infercnvpy as cnv
import numpy as np
import pandas as pd
from scipy import sparse


EPITHELIAL_MARKERS = ("EPCAM", "SFN", "KRT4", "KRT5", "KRT8", "KRT14", "KRT18", "KRT19", "KRT17", "KRT13", "KRT16")
AUTOSOMES = {f"chr{i}" for i in range(1, 23)}


def parse_gtf(path: Path, gene_ids: set[str]) -> pd.DataFrame:
    positions: dict[str, tuple[str, int, int]] = {}
    pattern = re.compile(r'gene_id "([^"]+)"')
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            match = pattern.search(fields[8])
            if match is None:
                continue
            gene_id = match.group(1).split(".")[0]
            if gene_id in gene_ids and gene_id not in positions:
                positions[gene_id] = (f"chr{fields[0]}", int(fields[3]), int(fields[4]))
    return pd.DataFrame(
        ((gene, chrom, start, end) for gene, (chrom, start, end) in positions.items()),
        columns=["original_gene_id", "chromosome", "start", "end"],
    )


def marker_count(adata: ad.AnnData) -> np.ndarray:
    locations = np.flatnonzero(adata.var["gene_symbol"].astype(str).str.upper().isin(EPITHELIAL_MARKERS).to_numpy())
    values = adata.X[:, locations]
    return np.asarray((values > 0).sum(axis=1)).ravel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite malignant-ready output: {args.output}")

    source = ad.read_h5ad(args.preflight)
    qc = (source.obs["n_genes_by_counts"].astype(float) >= 500) & (source.obs["pct_counts_mt"].astype(float) <= 20.0)
    source.obs["qc_pass_phase_a"] = qc.to_numpy()
    source.obs["epithelial_marker_count"] = marker_count(source)
    epithelial = qc.to_numpy() & (source.obs["epithelial_marker_count"].to_numpy() >= 2)
    source.obs["epithelial_flag"] = np.where(epithelial, "TRUE", "FALSE")

    candidates = source[epithelial].copy()
    candidates.obs["cnv_reference"] = np.where(
        candidates.obs["tissue"].astype(str).eq("adjacent_normal"),
        "normal_epithelial", "tumor_epithelial",
    )
    gene_ids = set(candidates.var["original_gene_id"].astype(str).str.split(".").str[0])
    coordinates = parse_gtf(args.gtf, gene_ids)
    candidates.var["original_gene_id_base"] = candidates.var["original_gene_id"].astype(str).str.split(".").str[0].to_numpy()
    candidates.var = candidates.var.merge(coordinates, how="left", on="original_gene_id", suffixes=("", "_mapped"))
    candidates.var.index = candidates.var_names
    # Merge uses the raw Ensembl IDs without versions in this dataset; retain only ordered autosomal genes.
    if "chromosome" not in candidates.var or candidates.var["chromosome"].isna().all():
        coordinates = coordinates.rename(columns={"original_gene_id": "original_gene_id_base"})
        candidates.var = candidates.var.drop(columns=[column for column in ("chromosome", "start", "end") if column in candidates.var]).merge(
            coordinates, how="left", on="original_gene_id_base"
        )
        candidates.var.index = candidates.var_names
    ordered = candidates.var["chromosome"].isin(AUTOSOMES).to_numpy() & candidates.var["start"].notna().to_numpy()
    candidates = candidates[:, ordered].copy()
    candidates.var["start"] = candidates.var["start"].astype(int)
    candidates.var["end"] = candidates.var["end"].astype(int)

    outputs: list[ad.AnnData] = []
    summary: list[dict[str, object]] = []
    for patient in sorted(candidates.obs["patient_id"].astype(str).unique()):
        subset = candidates[candidates.obs["patient_id"].astype(str).eq(patient)].copy()
        normal = subset.obs["cnv_reference"].astype(str).eq("normal_epithelial").to_numpy()
        tumor = subset.obs["cnv_reference"].astype(str).eq("tumor_epithelial").to_numpy()
        if normal.sum() < 50 or tumor.sum() == 0:
            raise ValueError(f"{patient}: insufficient paired epithelial candidates for locked CNV rule (normal={normal.sum()}, tumor={tumor.sum()})")
        cnv.tl.infercnv(
            subset,
            reference_key="cnv_reference",
            reference_cat="normal_epithelial",
            lfc_clip=3,
            window_size=100,
            step=10,
            dynamic_threshold=1.5,
            exclude_chromosomes=("chrX", "chrY"),
            chunksize=2000,
            n_jobs=4,
            calculate_gene_values=False,
            inplace=True,
        )
        matrix = subset.obsm["X_cnv"]
        scores = np.asarray(np.abs(matrix).mean(axis=1)).ravel()
        subset.obs["cnv_score"] = scores
        threshold = float(np.quantile(scores[normal], 0.99))
        subset.obs["cnv_threshold_patient_99pct_normal"] = threshold
        malignant = tumor & (scores > threshold)
        selected = subset[malignant].copy()
        selected.obs["malignant_flag_initial"] = "TRUE"
        selected.obs["author_celltype"] = "marker_defined_epithelial_with_paired_CNV_evidence"
        selected.obs["epithelial_flag"] = "TRUE"
        selected.obsm.clear()
        outputs.append(selected)
        summary.append({
            "patient": patient, "normal_epithelial_candidates": int(normal.sum()),
            "tumor_epithelial_candidates": int(tumor.sum()), "cnv_threshold": threshold,
            "malignant_cells": int(malignant.sum()),
        })

    result = ad.concat(outputs, join="inner", merge="same", index_unique=None)
    result.X = result.X.tocsr() if sparse.issparse(result.X) else sparse.csr_matrix(result.X)
    result.uns["X_semantics"] = "raw_nonnegative_integer_UMI_counts"
    result.uns["day2_phase_a_complete"] = True
    result.uns["cnv_inference_run_phase_a"] = True
    result.uns["cnv_method"] = "infercnvpy paired normal epithelial reference; autosomal sliding-window CNV"
    result.uns["malignant_rule"] = "tumor epithelial marker candidate with CNV score > paired-normal epithelial 99th percentile"
    result.uns["cnmf_run"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output, compression=None)

    summary_table = "\n".join(
        f"| {row['patient']} | {row['normal_epithelial_candidates']} | {row['tumor_epithelial_candidates']} | {row['cnv_threshold']:.6f} | {row['malignant_cells']} |"
        for row in summary
    )
    text = f"""# GSE196756 Day 2 Phase A

## Status

- **MALIGNANT READY**
- Annotation class: **C** — GEO supplies raw processed 10x matrices but no cell-level annotation.
- Pre-specified identification sequence: QC → canonical epithelial-marker consistency → paired normal epithelial reference → CNV evidence.
- QC: at least 500 detected genes and mitochondrial fraction no greater than 20%; no outcome data were accessed.
- Epithelial candidate rule: at least two detected genes from `{', '.join(EPITHELIAL_MARKERS)}`.
- Malignant rule: tumor epithelial candidate CNV score greater than the patient-matched normal epithelial 99th-percentile score.
- cNMF: **not run**.

## Counts

| Stage | Cells |
|---|---:|
| Input cells | {source.n_obs} |
| QC-pass cells | {int(qc.sum())} |
| Epithelial candidates | {int(epithelial.sum())} |
| Malignant-ready cells | {result.n_obs} |

## Paired CNV evidence

| patient | normal epithelial candidates | tumor epithelial candidates | paired-normal CNV threshold | malignant cells |
|---|---:|---:|---:|---:|
{summary_table}

Output: `{args.output.as_posix()}`  
`.X` retains raw nonnegative integer UMI counts. No program discovery, K/rank choice, treatment, spatial, clinical, or survival analysis was run.
"""
    args.log.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
