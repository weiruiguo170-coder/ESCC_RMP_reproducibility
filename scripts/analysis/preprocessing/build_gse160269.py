#!/usr/bin/env python
"""Build the independent GSE160269 epithelial preflight AnnData.

The author-provided CD45-negative UMI matrix is streamed gene-by-gene and
subset using the author epithelial annotation. No QC filtering or program
discovery is performed.
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
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stream_epithelial_matrix(matrix_path: Path, metadata: pd.DataFrame):
    if ".partial" in matrix_path.name:
        raise RuntimeError("A .partial file cannot enter the analysis pipeline")
    with gzip.open(matrix_path, "rt") as handle:
        barcodes = pd.Index(handle.readline().split(), name="cell_barcode")
        if len(barcodes) != len(metadata):
            raise ValueError(f"Header/metadata length mismatch: {len(barcodes)} != {len(metadata)}")
        if not np.array_equal(barcodes.to_numpy(), metadata["cell"].astype(str).to_numpy()):
            raise ValueError("Matrix header and metadata cell rows are not in identical order")
        if barcodes.duplicated().any():
            raise ValueError("Duplicated barcodes in full matrix header")

        selected_source_indices = np.flatnonzero(metadata["annotated_type"].eq("Epithelial").to_numpy())
        selected_barcodes = barcodes[selected_source_indices]
        gene_names: list[str] = []
        chunks: list[sparse.csr_matrix] = []
        current_rows: list[sparse.csr_matrix] = []
        raw_value_min = np.inf
        raw_value_max = -np.inf

        for line_number, line in enumerate(handle, start=2):
            try:
                gene, payload = line.rstrip("\n").split(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed expression row {line_number}") from exc
            values = np.fromstring(payload, sep=" ", dtype=np.int64)
            if values.size != barcodes.size:
                raise ValueError(
                    f"Expression row {line_number} has {values.size} values; expected {barcodes.size}"
                )
            selected = values[selected_source_indices]
            if selected.size:
                raw_value_min = min(raw_value_min, int(selected.min()))
                raw_value_max = max(raw_value_max, int(selected.max()))
            current_rows.append(sparse.csr_matrix(selected.reshape(1, -1), dtype=np.int32))
            gene_names.append(gene)
            if len(current_rows) == 256:
                chunks.append(sparse.vstack(current_rows, format="csr"))
                current_rows.clear()
        if current_rows:
            chunks.append(sparse.vstack(current_rows, format="csr"))

    gene_by_cell = sparse.vstack(chunks, format="csr")
    return gene_by_cell.T.tocsr(), selected_barcodes, pd.Index(gene_names), raw_value_min, raw_value_max


def qc_summary(series: pd.Series) -> str:
    values = series.astype(float)
    return (
        f"min={values.min():.3f}; median={values.median():.3f}; "
        f"mean={values.mean():.3f}; max={values.max():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()

    if any(".partial" in path.name for path in (args.matrix, args.metadata)):
        raise RuntimeError("A .partial file cannot enter the analysis pipeline")
    metadata = pd.read_csv(resolve_data(args.metadata), sep=r"\s+")
    expected_columns = ["cell", "sample", "annotated_type"]
    if metadata.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected metadata columns: {metadata.columns.tolist()}")
    if metadata["cell"].duplicated().any():
        raise ValueError("Duplicated metadata cell identifiers")

    X, barcodes, genes, value_min, value_max = stream_epithelial_matrix(args.matrix, metadata)
    selected_metadata = metadata.loc[metadata["annotated_type"].eq("Epithelial")].copy()
    selected_metadata.index = barcodes
    if X.shape != (len(selected_metadata), len(genes)):
        raise ValueError(f"Final matrix shape mismatch: {X.shape}")
    if not np.isfinite(X.data).all() or value_min < 0:
        raise ValueError("Invalid count values detected")

    obs = pd.DataFrame(index=barcodes)
    obs["dataset"] = "GSE160269"
    obs["patient_id"] = selected_metadata["sample"].str.extract(r"^(P\d+)[TN]$", expand=False).to_numpy()
    if obs["patient_id"].isna().any():
        raise ValueError("Could not derive patient IDs from all sample labels")
    obs["sample_id"] = selected_metadata["sample"].astype(str).to_numpy()
    obs["tissue"] = np.where(obs["sample_id"].str.endswith("N"), "adjacent_normal", "tumor")
    obs["author_celltype"] = selected_metadata["annotated_type"].astype(str).to_numpy()
    obs["epithelial_flag"] = "TRUE"
    obs["malignant_flag_initial"] = "TO_VERIFY"
    for column in (
        "dataset", "patient_id", "sample_id", "tissue", "author_celltype",
        "epithelial_flag", "malignant_flag_initial",
    ):
        obs[column] = obs[column].astype("category")

    var = pd.DataFrame(index=genes.astype(str))
    var.index.name = "feature_name_unique"
    var["gene_symbol"] = genes.astype(str).to_numpy()
    duplicate_genes = int(genes.duplicated().sum())
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.var_names.name = "feature_name_unique"
    adata.var["mt"] = adata.var["gene_symbol"].str.upper().str.startswith("MT-").to_numpy()
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata.uns["preflight_schema_version"] = "1.0"
    adata.uns["X_semantics"] = "raw_nonnegative_integer_UMI_counts"
    adata.uns["source_compartment"] = "author_annotated_CD45_negative_epithelial_cells"
    adata.uns["discovery_policy"] = "cohort_independent_no_concatenation_across_datasets"
    adata.uns["filtering_applied"] = False
    adata.uns["cnmf_run"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output, compression=None)

    sparse_ram = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024**3)
    audit = f"""# GSE160269 Input Audit

Audit date: 2026-08-26 (Asia/Shanghai)  
Source script: `scripts/01_preprocessing/build_gse160269_preflight.py`

## Decision

- Input status: **READY**
- Malignant-cell annotation class: **B — the author provides an epithelial annotation, but no cell-level malignant flag in the downloaded metadata**
- Day 2 action: confirm epithelial-marker consistency; tumor-derived epithelial cells remain `TO_VERIFY` for malignancy. Apply CNV evidence only if required and feasible (`CNV_INFERENCE_REQUIRED`), without treating tumor origin alone as proof of malignancy.
- No QC filtering, program discovery, treatment, spatial, or clinical analysis was run.

## Provenance and integrity

- Expression source: author-provided GEO `GSE160269_CD45neg_UMIs.txt.gz`, retrieved inside a checksum-verified Figshare supporting-data mirror.
- Cell metadata source: `GSE160269_CD45neg_cells.txt.gz`; the independently downloaded official GEO copy and mirrored copy have identical SHA256.
- Matrix gzip SHA256: `{sha256(args.matrix)}`
- Metadata gzip SHA256: `{sha256(args.metadata)}`
- Both gzip streams passed full CRC/decompression before conversion.

## Matrix audit

| Field | Value |
|---|---:|
| Full CD45-negative metadata rows | {len(metadata)} |
| Selected author-annotated epithelial cells | {adata.n_obs} |
| Matrix dimensions (cells × genes) | {adata.n_obs} × {adata.n_vars} |
| Storage | CSR sparse matrix |
| Value semantics | raw, nonnegative integer UMI counts |
| Observed count range | {int(value_min)} to {int(value_max)} |
| Gene identifiers | gene symbols |
| Duplicate source gene symbols | {duplicate_genes} |
| Duplicated selected barcodes | {int(adata.obs_names.duplicated().sum())} |
| NaN/Inf | none |
| Metadata rows corresponding to output | {adata.n_obs} |
| Patients | {adata.obs['patient_id'].nunique()} |
| Samples | {adata.obs['sample_id'].nunique()} |
| Tumor epithelial cells | {int((adata.obs['tissue'] == 'tumor').sum())} |
| Adjacent-normal epithelial cells | {int((adata.obs['tissue'] == 'adjacent_normal').sum())} |
| Author epithelial annotation | available for every output cell |
| Author malignant annotation | unavailable |
| Obvious corruption | none detected |
| Sparse matrix RAM footprint | {sparse_ram:.2f} GiB (analysis overhead requires additional RAM) |
| Standard output | `{args.output.as_posix()}` |

## Unfiltered QC preflight

| Metric | Summary |
|---|---|
| Detected genes per cell | {qc_summary(adata.obs['n_genes_by_counts'])} |
| Total UMI counts per cell | {qc_summary(adata.obs['total_counts'])} |
| Mitochondrial fraction (%) | {qc_summary(adata.obs['pct_counts_mt'])} |

The matrix header matched all {len(metadata)} metadata rows in identical order before epithelial subsetting. `.partial` inputs are explicitly rejected by the loader.
"""
    args.audit.write_text(audit, encoding="utf-8")
    print(f"WROTE {args.output} shape={adata.shape} nnz={adata.X.nnz}")


if __name__ == "__main__":
    main()
