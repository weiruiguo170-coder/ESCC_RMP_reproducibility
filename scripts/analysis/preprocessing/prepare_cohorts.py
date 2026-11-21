#!/usr/bin/env python
"""Outcome-blind Day 2 Phase A preparation for independent ESCC cohorts.

This script performs only input QC, epithelial-boundary confirmation and the
pre-specified malignant-cell boundary. It never normalizes data, selects HVGs,
runs CNV inference, chooses NMF rank, or runs cNMF.
"""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


EPITHELIAL_MARKERS = ("EPCAM", "SFN", "KRT4", "KRT5", "KRT8", "KRT14", "KRT18", "KRT19")
GSE188_GROUP_TO_SAMPLE = {
    "Heso_1": "Patient1",
    "Heso_2": "Patient2",
    "Heso_3": "Patient3",
    "Heso_4": "Patient4",
    "Heso_5": "Patient5",
    "Heso_6_para": "Patient6_ESCC1",
    "Heso_6_cancer": "Patient6_ESCC2",
    "Heso6_normal": "Patient6_normal",
}


def robust_upper_flag(values: pd.Series) -> np.ndarray:
    x = values.astype(float).to_numpy()
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    if mad == 0:
        return np.zeros(x.size, dtype=bool)
    return x > median + 5.0 * mad


def add_qc_flags(adata: ad.AnnData) -> None:
    genes = adata.obs["n_genes_by_counts"].astype(float)
    mito = adata.obs["pct_counts_mt"].astype(float)
    adata.obs["qc_pass_phase_a"] = (genes >= 500) & (mito <= 20.0)
    high = np.zeros(adata.n_obs, dtype=bool)
    sample_values = adata.obs["sample_id"].astype(str).to_numpy()
    for sample in pd.unique(sample_values):
        loc = np.flatnonzero(sample_values == sample)
        gene_flag = robust_upper_flag(adata.obs.iloc[loc]["n_genes_by_counts"])
        count_flag = robust_upper_flag(adata.obs.iloc[loc]["total_counts"])
        high[loc] = gene_flag | count_flag
    adata.obs["doublet_like_qc_flag"] = high


def marker_consistency(adata: ad.AnnData) -> tuple[np.ndarray, list[str]]:
    symbols = adata.var["gene_symbol"].astype(str).str.upper()
    positions = np.flatnonzero(symbols.isin(EPITHELIAL_MARKERS).to_numpy())
    present = sorted(set(symbols.iloc[positions]).intersection(EPITHELIAL_MARKERS))
    if not len(positions):
        return np.zeros(adata.n_obs, dtype=bool), present
    detected = adata.X[:, positions]
    if sparse.issparse(detected):
        flag = np.asarray((detected > 0).sum(axis=1)).ravel() > 0
    else:
        flag = (np.asarray(detected) > 0).sum(axis=1) > 0
    return flag, present


def map_gse188_metadata(path: Path, obs_names: pd.Index) -> pd.DataFrame:
    table = pd.read_csv(resolve_data(path))
    required = {"cell_id", "group"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} lacks required columns: {sorted(required)}")
    sample = table["group"].map(GSE188_GROUP_TO_SAMPLE)
    if sample.isna().any():
        missing = sorted(table.loc[sample.isna(), "group"].astype(str).unique())
        raise ValueError(f"Unmapped author group labels: {missing}")
    barcode = []
    for cell, group in zip(table["cell_id"].astype(str), table["group"].astype(str)):
        cell_group = "Heso_6_normal" if group == "Heso6_normal" else group
        prefix = f"{cell_group}_"
        if not cell.startswith(prefix):
            raise ValueError(f"Unexpected author cell identifier: {cell}")
        value = cell[len(prefix):]
        barcode.append(value if value.endswith("-1") else f"{value}-1")
    table = table.copy()
    table["preflight_cell_id"] = sample.astype(str) + ":" + pd.Series(barcode, index=table.index)
    if table["preflight_cell_id"].duplicated().any():
        raise ValueError(f"Duplicated mapped author cells in {path}")
    missing_cells = ~table["preflight_cell_id"].isin(obs_names)
    missing_count = int(missing_cells.sum())
    missing_examples = table.loc[missing_cells, "preflight_cell_id"].head().tolist()
    table = table.loc[~missing_cells].copy().set_index("preflight_cell_id", drop=False)
    table.attrs["missing_raw_count"] = missing_count
    table.attrs["missing_raw_examples"] = missing_examples
    return table


def patient_table(obs: pd.DataFrame) -> str:
    counts = obs.groupby("patient_id", observed=True).size().sort_index()
    return "\n".join(f"| {patient} | {int(count)} |" for patient, count in counts.items())


def write_output(source: ad.AnnData, malignant: np.ndarray, output: Path) -> ad.AnnData:
    result = source[malignant].copy()
    result.X = result.X.tocsr() if sparse.issparse(result.X) else sparse.csr_matrix(result.X)
    result.obs["dataset"] = result.obs["dataset"].astype(str).astype("category")
    result.obs["epithelial_flag"] = "TRUE"
    result.obs["malignant_flag_initial"] = "TRUE"
    result.obs["epithelial_flag"] = result.obs["epithelial_flag"].astype("category")
    result.obs["malignant_flag_initial"] = result.obs["malignant_flag_initial"].astype("category")
    result.uns["X_semantics"] = "raw_nonnegative_integer_UMI_counts"
    result.uns["day2_phase_a_complete"] = True
    result.uns["cnv_inference_run_phase_a"] = False
    result.uns["cnmf_run"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(output, compression=None)
    return result


def run_gse160(adata: ad.AnnData, output: Path, log: Path) -> None:
    add_qc_flags(adata)
    marker_flag, markers_present = marker_consistency(adata)
    adata.obs["epithelial_marker_consistent"] = marker_flag
    epithelial = adata.obs["epithelial_flag"].astype(str).eq("TRUE").to_numpy()
    malignant = (
        adata.obs["qc_pass_phase_a"].to_numpy()
        & epithelial
        & adata.obs["tissue"].astype(str).eq("tumor").to_numpy()
    )
    result = write_output(adata, malignant, output)
    result.uns["malignant_identification_method"] = (
        "author epithelial annotation + tumor tissue identity + author-reported cohort-level CNV/WGS evidence"
    )
    result.write_h5ad(output, compression=None)
    text = f"""# GSE160269 Day 2 Phase A

Date: 2026-08-26 (Asia/Shanghai)

## Status

- **MALIGNANT READY**
- Annotation class: **A/B hybrid** — cell-level epithelial annotation is author-provided; the primary publication treats tumor epithelial cells as malignant-program input and reports severe transcriptome-inferred CNVs supported by bulk WGS, while noting that most epithelial cells are malignant.
- CNV inference in this phase: **not rerun**. The author boundary and orthogonal cohort-level CNV/WGS evidence are retained; no outcome was consulted.
- cNMF: **not run**.

## Outcome-blind QC

- Author rule reproduced: retain cells with at least 500 detected genes and mitochondrial fraction no greater than 20%.
- No upper count/gene cutoff was used for exclusion. Per-sample median + 5 MAD upper outliers are retained and recorded only as `doublet_like_qc_flag`, because no author doublet calls are available.
- Canonical epithelial panel present in the matrix: {', '.join(markers_present)}. Marker consistency is recorded but does not override the author annotation.

| Stage | Cells |
|---|---:|
| Input author-annotated epithelial cells | {adata.n_obs} |
| QC-pass epithelial cells | {int((adata.obs['qc_pass_phase_a'].to_numpy() & epithelial).sum())} |
| Excluded by QC | {int((~adata.obs['qc_pass_phase_a'].to_numpy()).sum())} |
| Epithelial-marker-consistent | {int(marker_flag.sum())} |
| Retained doublet-like QC flags (not excluded) | {int(adata.obs['doublet_like_qc_flag'].sum())} |
| Malignant-ready tumor epithelial cells | {result.n_obs} |

Patients represented in malignant-ready output: **{result.obs['patient_id'].nunique()}**.

## Malignant cells per patient

| patient_id | malignant_cells |
|---|---:|
{patient_table(result.obs)}

Output: `{output.as_posix()}`  
`.X`: raw nonnegative integer UMI counts. No normalization or program discovery was performed.
"""
    log.write_text(text, encoding="utf-8")


def run_gse188(
    adata: ad.AnnData,
    epithelial_metadata: Path,
    malignant_metadata: Path,
    output: Path,
    log: Path,
) -> None:
    add_qc_flags(adata)
    marker_flag, markers_present = marker_consistency(adata)
    adata.obs["epithelial_marker_consistent"] = marker_flag
    epithelial_table = map_gse188_metadata(epithelial_metadata, adata.obs_names)
    malignant_table = map_gse188_metadata(malignant_metadata, adata.obs_names)
    epithelial = adata.obs_names.isin(epithelial_table.index)
    malignant_author = adata.obs_names.isin(malignant_table.index)
    if np.any(malignant_author & ~epithelial):
        raise ValueError("Author malignant set is not a subset of the author epithelial set")
    adata.obs["epithelial_flag"] = np.where(epithelial, "TRUE", "FALSE")
    adata.obs["malignant_flag_initial"] = np.where(malignant_author, "TRUE", "FALSE")
    malignant = malignant_author & adata.obs["qc_pass_phase_a"].to_numpy()
    result = write_output(adata, malignant, output)
    result.uns["malignant_identification_method"] = "author-provided malignant squamous-cell Seurat object"
    result.write_h5ad(output, compression=None)
    text = f"""# GSE188900 Day 2 Phase A

Date: 2026-08-26 (Asia/Shanghai)

## Status

- **MALIGNANT READY**
- Annotation class: **A** — the published author processed objects explicitly provide squamous/cancer cells and malignant squamous cells.
- The author malignant set mapped one-to-one to the independent raw-count AnnData and is a strict subset of the mapped epithelial set. Any epithelial-object cell absent from the GEO matrix is counted below and excluded transparently.
- CNV inference in this phase: **not used** because a direct author malignant annotation is available.
- cNMF: **not run**.

## Outcome-blind QC

- Mechanical validation screen: at least 500 detected genes and mitochondrial fraction no greater than 20%.
- No upper count/gene cutoff was used for exclusion. Per-sample median + 5 MAD upper outliers are retained and recorded only as `doublet_like_qc_flag`.
- Canonical epithelial panel present in the matrix: {', '.join(markers_present)}. Marker consistency is recorded but does not override the author cell-level boundary.

| Stage | Cells |
|---|---:|
| Full independent raw-count input | {adata.n_obs} |
| QC-pass cells (all compartments) | {int(adata.obs['qc_pass_phase_a'].sum())} |
| Author epithelial/squamous compartment | {int(epithelial.sum())} |
| Author epithelial cells absent from GEO raw matrix | {epithelial_table.attrs['missing_raw_count']} |
| QC-pass author epithelial cells | {int((epithelial & adata.obs['qc_pass_phase_a'].to_numpy()).sum())} |
| Author malignant squamous cells | {int(malignant_author.sum())} |
| Author malignant cells absent from GEO raw matrix | {malignant_table.attrs['missing_raw_count']} |
| Malignant-ready after QC validation | {result.n_obs} |
| Retained doublet-like QC flags among malignant cells | {int(result.obs['doublet_like_qc_flag'].sum())} |

Patients represented in malignant-ready output: **{result.obs['patient_id'].nunique()}**.

## Malignant cells per patient

| patient_id | malignant_cells |
|---|---:|
{patient_table(result.obs)}

Output: `{output.as_posix()}`  
`.X`: raw nonnegative integer UMI counts. No normalization or program discovery was performed.
"""
    log.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("GSE160269", "GSE188900"), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--author-epithelial-metadata", type=Path)
    parser.add_argument("--author-malignant-metadata", type=Path)
    args = parser.parse_args()
    adata = ad.read_h5ad(args.preflight)
    if args.dataset == "GSE160269":
        run_gse160(adata, args.output, args.log)
    else:
        if args.author_epithelial_metadata is None or args.author_malignant_metadata is None:
            parser.error("GSE188900 requires both author metadata files")
        run_gse188(
            adata,
            args.author_epithelial_metadata,
            args.author_malignant_metadata,
            args.output,
            args.log,
        )


if __name__ == "__main__":
    main()
