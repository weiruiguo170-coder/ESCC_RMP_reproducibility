#!/usr/bin/env python
"""Build one cohort-level preflight AnnData from independent 10x matrices.

This script is intentionally limited to input validation, schema harmonization,
and QC preflight. It does not filter cells, identify programs, or run cNMF.
"""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import argparse
import gzip
import hashlib
import math
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open("rt")


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with open_text(path) as handle:
        return sum(1 for _ in handle)


def matrix_market_dimensions(path: Path) -> tuple[int, int, int]:
    with open_text(path) as handle:
        for line in handle:
            if not line.startswith("%"):
                values = tuple(int(value) for value in line.split())
                if len(values) != 3:
                    raise ValueError(f"Invalid Matrix Market size line in {path}: {line!r}")
                return values
    raise ValueError(f"Missing Matrix Market size line: {path}")


def find_triplet(directory: Path) -> tuple[Path, Path, Path]:
    matrices = list(directory.glob("matrix.mtx*"))
    features = list(directory.glob("features.tsv*")) or list(directory.glob("genes.tsv*"))
    barcodes = list(directory.glob("barcodes.tsv*"))
    if len(matrices) != 1 or len(features) != 1 or len(barcodes) != 1:
        raise FileNotFoundError(f"Expected one 10x triplet in {directory}")
    return matrices[0], features[0], barcodes[0]


def sample_fields(dataset: str, sample: str) -> tuple[str, str, str]:
    if dataset == "GSE188900":
        match = re.fullmatch(r"Patient(\d+)(?:_(ESCC\d+|normal))?", sample)
        if not match:
            raise ValueError(f"Unexpected GSE188900 sample name: {sample}")
        patient = f"P{match.group(1)}"
        tissue = "adjacent_normal" if match.group(2) == "normal" else "tumor"
        return patient, sample, tissue
    if dataset == "GSE196756":
        match = re.fullmatch(r"(GSM\d+)_\d+([TN])", sample)
        if not match:
            raise ValueError(f"Unexpected GSE196756 sample name: {sample}")
        patient_by_gsm = {
            "GSM5900215": "P1", "GSM5900216": "P1",
            "GSM5900217": "P2", "GSM5900218": "P2",
            "GSM5900219": "P4", "GSM5900220": "P4",
        }
        patient = patient_by_gsm.get(match.group(1))
        if patient is None:
            raise ValueError(f"Unmapped GSE196756 GSM accession: {sample}")
        tissue = "tumor" if match.group(2) == "T" else "adjacent_normal"
        return patient, sample, tissue
    raise ValueError(f"Unsupported dataset: {dataset}")


def read_sample(dataset: str, sample: str, directory: Path) -> tuple[ad.AnnData, dict]:
    matrix, features, barcodes = find_triplet(directory)
    if any(".partial" in path.name for path in (matrix, features, barcodes)):
        raise RuntimeError("A .partial file cannot enter the analysis pipeline")
    n_gene_mtx, n_cell_mtx, nnz_header = matrix_market_dimensions(matrix)
    n_gene_file = count_lines(features)
    n_cell_file = count_lines(barcodes)
    if (n_gene_mtx, n_cell_mtx) != (n_gene_file, n_cell_file):
        raise ValueError(
            f"Dimension mismatch for {sample}: matrix={(n_gene_mtx, n_cell_mtx)}, "
            f"features={n_gene_file}, barcodes={n_cell_file}"
        )

    matrix_handle = gzip.open(matrix, "rb") if matrix.suffix == ".gz" else matrix.open("rb")
    with matrix_handle:
        counts = mmread(matrix_handle).T.tocsr()
    with open_text(features) as handle:
        feature_table = pd.read_csv(resolve_data(handle), sep="\t", header=None)
    with open_text(barcodes) as handle:
        barcode_table = pd.read_csv(resolve_data(handle), sep="\t", header=None)
    gene_symbols = feature_table.iloc[:, 1] if feature_table.shape[1] > 1 else feature_table.iloc[:, 0]
    original_gene_ids = feature_table.iloc[:, 0]
    adata = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(index=pd.Index(barcode_table.iloc[:, 0].astype(str), name="barcode")),
        var=pd.DataFrame(
            {"gene_ids": original_gene_ids.astype(str).to_numpy()},
            index=pd.Index(gene_symbols.astype(str), name="gene_symbol"),
        ),
    )
    if adata.shape != (n_cell_mtx, n_gene_mtx):
        raise ValueError(f"Unexpected imported shape for {sample}: {adata.shape}")
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()
    if adata.X.nnz != nnz_header:
        raise ValueError(f"NNZ mismatch for {sample}: {adata.X.nnz} != {nnz_header}")
    if not np.isfinite(adata.X.data).all():
        raise ValueError(f"NaN/Inf detected in {sample}")
    if np.any(adata.X.data < 0) or not np.allclose(adata.X.data, np.round(adata.X.data)):
        raise ValueError(f"Matrix is not a nonnegative integer-count matrix: {sample}")

    original_symbols = pd.Index(adata.var_names.astype(str))
    duplicate_gene_symbols = int(original_symbols.duplicated().sum())
    adata.var["gene_symbol"] = original_symbols.to_numpy()
    if "gene_ids" in adata.var:
        adata.var["original_gene_id"] = adata.var["gene_ids"].astype(str).to_numpy()
    else:
        adata.var["original_gene_id"] = original_symbols.to_numpy()
    adata.var_names_make_unique()
    adata.var_names.name = "feature_name_unique"

    patient_id, sample_id, tissue = sample_fields(dataset, sample)
    adata.obs_names = pd.Index([f"{sample_id}:{barcode}" for barcode in adata.obs_names])
    adata.obs["dataset"] = dataset
    adata.obs["patient_id"] = patient_id
    adata.obs["sample_id"] = sample_id
    adata.obs["tissue"] = tissue
    adata.obs["author_celltype"] = "NA"
    adata.obs["epithelial_flag"] = "TO_VERIFY"
    adata.obs["malignant_flag_initial"] = "TO_VERIFY"

    adata.var["mt"] = adata.var["gene_symbol"].str.upper().str.startswith("MT-").to_numpy()
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    stats = {
        "sample": sample_id,
        "patient": patient_id,
        "tissue": tissue,
        "genes": n_gene_mtx,
        "cells": n_cell_mtx,
        "nnz": nnz_header,
        "duplicate_gene_symbols": duplicate_gene_symbols,
        "duplicate_barcodes_within_sample": int(adata.obs_names.duplicated().sum()),
        "matrix_sha256": sha256(matrix),
        "features_sha256": sha256(features),
        "barcodes_sha256": sha256(barcodes),
    }
    return adata, stats


def discover_samples(dataset: str, root: Path) -> list[tuple[str, Path]]:
    if dataset == "GSE188900":
        samples = [(path.name, path) for path in sorted(root.iterdir()) if path.is_dir()]
    elif dataset == "GSE196756":
        samples = []
        for matrix in sorted(root.glob("*_matrix.mtx*")):
            sample = re.sub(r"_matrix\.mtx(?:\.gz)?$", "", matrix.name)
            sample_dir = root / sample
            sample_dir.mkdir(exist_ok=True)
            for suffix in ("matrix.mtx.gz", "matrix.mtx", "features.tsv", "features.tsv", "genes.tsv", "genes.tsv", "barcodes.tsv", "barcodes.tsv"):
                source = root / f"{sample}_{suffix}"
                target = sample_dir / suffix
                if source.exists() and not target.exists():
                    target.hardlink_to(source)
            samples.append((sample, sample_dir))
    else:
        raise ValueError(dataset)
    if not samples:
        raise FileNotFoundError(f"No samples discovered under {root}")
    return samples


def write_audit(dataset: str, output: Path, audit_path: Path, adata: ad.AnnData, stats: list[dict]) -> None:
    cells = int(adata.n_obs)
    genes = int(adata.n_vars)
    patients = int(adata.obs["patient_id"].nunique())
    samples = int(adata.obs["sample_id"].nunique())
    duplicate_symbols = int(pd.Index(adata.var["gene_symbol"]).duplicated().sum())
    duplicate_barcodes = int(adata.obs_names.duplicated().sum())
    finite = bool(np.isfinite(adata.X.data).all())
    estimated_ram_gib = (adata.X.data.nbytes + adata.X.indices.nbytes + adata.X.indptr.nbytes) / (1024**3)
    def summary(column: str) -> str:
        values = adata.obs[column].astype(float)
        return (
            f"min={values.min():.3f}; median={values.median():.3f}; "
            f"mean={values.mean():.3f}; max={values.max():.3f}"
        )
    table = "\n".join(
        f"| {item['sample']} | {item['patient']} | {item['tissue']} | {item['genes']} | "
        f"{item['cells']} | {item['nnz']} |"
        for item in stats
    )
    text = f"""# {dataset} Input Audit

Audit date: 2026-08-26 (Asia/Shanghai)  
Source script: `scripts/01_preprocessing/build_10x_preflight.py`

## Decision

- Input status: **READY**
- Malignant-cell annotation class: **C — cell-level author annotation is not present in the downloaded 10x archive**
- Day 2 action: implement epithelial marker consistency first; use CNV evidence only if needed and feasible (`CNV_INFERENCE_REQUIRED` for epithelial candidates lacking reliable author malignant labels).
- No filtering threshold, program discovery, treatment, spatial, or clinical analysis was run.

## Matrix audit

| Field | Value |
|---|---:|
| Matrix dimensions (cells × genes) | {cells} × {genes} |
| Storage | CSR sparse matrix |
| Value semantics | raw, nonnegative integer UMI counts from processed 10x matrices |
| Genes | {genes} |
| Cells | {cells} |
| Nonzero entries | {int(adata.X.nnz)} |
| Gene identifiers | gene symbols in `var_names`; original Ensembl IDs retained in `original_gene_id` |
| Duplicate source gene symbols | {duplicate_symbols} (made unique only for `var_names`; original symbols retained) |
| Duplicated unified barcodes | {duplicate_barcodes} |
| NaN/Inf | {'none' if finite else 'DETECTED'} |
| Metadata rows | {cells} |
| Patients | {patients} |
| Samples | {samples} |
| Author cell-type annotation | unavailable in this matrix archive |
| Epithelial annotation | unavailable; `TO_VERIFY` |
| Malignant annotation | unavailable; `TO_VERIFY` |
| Obvious corruption | none detected |
| Sparse matrix RAM footprint | {estimated_ram_gib:.2f} GiB (object overhead and analyses require additional RAM) |
| Standard output | `{output.as_posix()}` |

## Sample-level dimension checks

| sample | patient | tissue | genes | cells | nnz |
|---|---|---|---:|---:|---:|
{table}

Every Matrix Market header matched its features and barcodes line counts. Every imported sparse matrix matched the declared nonzero count. `.partial` inputs are explicitly rejected by the loader.

## Unfiltered QC preflight

| Metric | Summary |
|---|---|
| Detected genes per cell | {summary('n_genes_by_counts')} |
| Total UMI counts per cell | {summary('total_counts')} |
| Mitochondrial fraction (%) | {summary('pct_counts_mt')} |

These are descriptive preflight values only. No cell was filtered and no threshold was selected.
"""
    audit_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["GSE188900", "GSE196756"])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    sample_data = []
    sample_stats = []
    for sample, directory in discover_samples(args.dataset, args.input_root):
        adata, stats = read_sample(args.dataset, sample, directory)
        sample_data.append(adata)
        sample_stats.append(stats)

    cohort = ad.concat(sample_data, join="inner", merge="same", index_unique=None)
    cohort.X = cohort.X.tocsr()
    for column in (
        "dataset", "patient_id", "sample_id", "tissue", "author_celltype",
        "epithelial_flag", "malignant_flag_initial",
    ):
        cohort.obs[column] = cohort.obs[column].astype("category")
    cohort.uns["preflight_schema_version"] = "1.0"
    cohort.uns["X_semantics"] = "raw_nonnegative_integer_UMI_counts"
    cohort.uns["discovery_policy"] = "cohort_independent_no_concatenation_across_datasets"
    cohort.uns["filtering_applied"] = False
    cohort.uns["cnmf_run"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_h5ad(args.output, compression=None)
    write_audit(args.dataset, args.output, args.audit, cohort, sample_stats)
    print(f"WROTE {args.output} shape={cohort.shape} nnz={cohort.X.nnz}")


if __name__ == "__main__":
    main()
