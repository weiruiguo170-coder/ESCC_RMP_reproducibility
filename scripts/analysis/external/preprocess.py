from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import hashlib
import json
import re
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
PROJECT_ROOT = INPUT_ROOT
PHASE_E = PROJECT_ROOT / 'results' / 'external_reproducibility'
RAW = PROJECT_ROOT / 'data' / 'raw'
PROCESSED = PHASE_E / '03_processed_external_data'
TABLES = PHASE_E / '10_tables'
SEED = 20260827

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def canonical(symbol: object) -> str:
    return str(symbol).strip().upper()

def collapse_duplicate_genes(adata: ad.AnnData) -> ad.AnnData:
    symbols = pd.Index([canonical(x) for x in adata.var_names])
    keep = (symbols != '') & (symbols != 'NAN')
    adata = adata[:, keep].copy()
    symbols = symbols[keep]
    unique = pd.Index(pd.unique(symbols))
    codes = unique.get_indexer(symbols)
    aggregator = sparse.csr_matrix((np.ones(len(codes), dtype=np.float32), (np.arange(len(codes)), codes)), shape=(len(codes), len(unique)))
    matrix = sparse.csr_matrix(adata.X) @ aggregator
    obs = adata.obs.copy()
    result = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=unique))
    result.var_names.name = 'gene_symbol'
    return result

def qc_metrics(adata: ad.AnnData) -> pd.DataFrame:
    matrix = sparse.csr_matrix(adata.X)
    n_counts = np.asarray(matrix.sum(axis=1)).ravel()
    n_genes = np.diff(matrix.indptr)
    mt_mask = np.asarray(adata.var_names.str.startswith('MT-'))
    mt_counts = np.asarray(matrix[:, mt_mask].sum(axis=1)).ravel() if mt_mask.any() else np.zeros(adata.n_obs)
    pct_mt = np.divide(mt_counts, n_counts, out=np.zeros_like(mt_counts, dtype=float), where=n_counts > 0) * 100
    return pd.DataFrame({'n_counts': n_counts, 'n_genes': n_genes, 'pct_mito': pct_mt}, index=adata.obs_names)

def read_gse199654() -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    directory = RAW / 'GSE199654'
    matrix_path = directory / 'GSE199654_scTDN_UMI_matrix_epithelial_cells.txt.gz'
    metadata_path = directory / 'GSE199654_scTDN_METADATA_epithelial_cells.txt.gz'
    with pd.io.common.get_handle(matrix_path, 'rt', compression='infer') as handle:
        barcodes = handle.handle.readline().rstrip('\n\r').split('\t')
    frame = pd.read_csv(resolve_data(matrix_path), sep='\t', header=None, skiprows=1, index_col=0, names=['gene'] + barcodes)
    frame = frame.astype(np.float32)
    frame.columns = frame.columns.str.replace('\\.1$', '-1', regex=True)
    matrix = sparse.csr_matrix(frame.to_numpy().T)
    obs = pd.read_csv(resolve_data(metadata_path), sep='\t').set_index('cellID')
    missing = pd.Index(frame.columns).difference(obs.index)
    if len(missing):
        raise RuntimeError(f'GSE199654 matrix/metadata mismatch: {len(missing)} cells')
    obs = obs.loc[frame.columns].copy()
    adata = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=frame.index.astype(str)))
    adata = collapse_duplicate_genes(adata)
    metrics = qc_metrics(adata)
    adata.obs = adata.obs.join(metrics)
    tumor = adata.obs.sampleType.astype(str).eq('T')
    qc = adata.obs.n_genes.ge(200) & adata.obs.n_genes.lt(6000) & adata.obs.pct_mito.lt(5.0)
    selected = tumor & qc
    adata.obs['primary_tumor'] = tumor
    adata.obs['qc_pass'] = qc
    adata.obs['malignant_inclusion'] = selected
    malignant = adata[selected].copy()
    malignant.obs['patient_id'] = malignant.obs.patientID.astype(str)
    malignant.obs['sample_id'] = malignant.obs.sampleID.astype(str)
    malignant.obs['cell_type'] = 'malignant_epithelial'
    malignant.uns['source_accession'] = 'GSE199654'
    malignant.uns['matrix_semantics'] = 'raw nonnegative integer UMI counts'
    malignant.uns['malignant_definition'] = 'author epithelial matrix; tumor-origin cells; published reproducible QC thresholds'
    flow = pd.DataFrame([{'cohort': 'GSE199654', 'step': 'public_epithelial_matrix', 'cells': adata.n_obs}, {'cohort': 'GSE199654', 'step': 'tumor_origin', 'cells': int(tumor.sum())}, {'cohort': 'GSE199654', 'step': 'published_reproducible_qc', 'cells': int((tumor & qc).sum())}, {'cohort': 'GSE199654', 'step': 'malignant_ready', 'cells': malignant.n_obs}])
    return (malignant, flow, adata.obs.reset_index(names='cell_id'))

def marker_score(adata: ad.AnnData, genes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    present = [g for g in genes if g in adata.var_names]
    if not present:
        return (np.zeros(adata.n_obs), np.zeros(adata.n_obs, dtype=int))
    matrix = sparse.csr_matrix(adata[:, present].X).astype(np.float64)
    totals = np.asarray(adata.X.sum(axis=1)).ravel().astype(float)
    scale = np.divide(10000.0, totals, out=np.zeros_like(totals), where=totals > 0)
    normalized = matrix.multiply(scale[:, None])
    normalized.data = np.log1p(normalized.data)
    score = np.asarray(normalized.mean(axis=1)).ravel()
    detected = np.asarray((matrix > 0).sum(axis=1)).ravel()
    return (score, detected)

def read_gse203067() -> tuple[ad.AnnData, ad.AnnData, pd.DataFrame, pd.DataFrame]:
    directory = RAW / 'GSE203067'
    adata = sc.read_10x_mtx(directory, var_names='gene_symbols', make_unique=False, gex_only=True, prefix='GSE203067_')
    adata = collapse_duplicate_genes(adata)
    suffix_map = {'1': ('L1', 'P1', 'L'), '2': ('L2', 'P2', 'L'), '3': ('L3', 'P3', 'L'), '4': ('L4', 'P4', 'L'), '5': ('N1', 'P1', 'N'), '6': ('N2', 'P2', 'N'), '7': ('N3', 'P3', 'N'), '8': ('N4', 'P4', 'N'), '9': ('T1', 'P1', 'T'), '10': ('T2', 'P2', 'T'), '11': ('T3', 'P3', 'T'), '12': ('T4', 'P4', 'T')}
    suffixes = [re.search('-(\\d+)$', x).group(1) if re.search('-(\\d+)$', x) else '' for x in adata.obs_names]
    if any((x not in suffix_map for x in suffixes)):
        raise RuntimeError('GSE203067 barcode suffix outside official map')
    adata.obs['sample_id'] = [suffix_map[x][0] for x in suffixes]
    adata.obs['patient_id'] = [suffix_map[x][1] for x in suffixes]
    adata.obs['tissue'] = [suffix_map[x][2] for x in suffixes]
    metrics = qc_metrics(adata)
    adata.obs = adata.obs.join(metrics)
    log_ratio = np.divide(np.log10(adata.obs.n_genes.clip(lower=1)), np.log10(adata.obs.n_counts.clip(lower=2)))
    adata.obs['log10_genes_per_umi'] = log_ratio
    qc = adata.obs.n_counts.ge(300) & adata.obs.n_genes.ge(200) & adata.obs.log10_genes_per_umi.gt(0.8) & adata.obs.pct_mito.lt(20.0)
    primary = adata.obs.tissue.eq('T')
    primary_qc = adata[primary & qc].copy()
    panels = {'epithelial': ['EPCAM', 'SFN', 'TP63', 'KRT4', 'KRT5', 'KRT13', 'KRT14', 'KRT18', 'KRT19'], 'immune': ['PTPRC', 'CD3D', 'CD3E', 'NKG7', 'MS4A1', 'CD79A', 'LST1', 'TYROBP', 'FCER1G'], 'fibroblast': ['COL1A1', 'COL1A2', 'COL3A1', 'DCN', 'LUM', 'PDGFRA', 'PDGFRB'], 'endothelial': ['PECAM1', 'VWF', 'EMCN', 'KDR', 'RAMP2', 'CLDN5']}
    scores: dict[str, np.ndarray] = {}
    detected: dict[str, np.ndarray] = {}
    for (name, genes) in panels.items():
        (scores[name], detected[name]) = marker_score(primary_qc, genes)
        primary_qc.obs[f'{name}_marker_score'] = scores[name]
        primary_qc.obs[f'{name}_markers_detected'] = detected[name]
    score_matrix = np.column_stack([scores[name] for name in panels])
    largest = np.argmax(score_matrix, axis=1) == 0
    core = [g for g in ['EPCAM', 'SFN', 'TP63', 'KRT5', 'KRT14'] if g in primary_qc.var_names]
    core_detected = np.asarray((sparse.csr_matrix(primary_qc[:, core].X) > 0).sum(axis=1)).ravel()
    ptprc_zero = np.ones(primary_qc.n_obs, dtype=bool)
    if 'PTPRC' in primary_qc.var_names:
        ptprc_zero = sparse.csr_matrix(primary_qc[:, 'PTPRC'].X).toarray().ravel() == 0
    epithelial = largest & (detected['epithelial'] >= 2) & (core_detected >= 1) & ptprc_zero
    primary_qc.obs['cell_type_inference'] = np.select([epithelial, np.argmax(score_matrix, axis=1) == 1, np.argmax(score_matrix, axis=1) == 2, np.argmax(score_matrix, axis=1) == 3], ['malignant_epithelial', 'immune', 'fibroblast', 'endothelial'], default='other_or_ambiguous')
    primary_qc.obs['malignant_inclusion'] = epithelial
    malignant = primary_qc[epithelial].copy()
    malignant.obs['cell_type'] = 'malignant_epithelial'
    malignant.uns['source_accession'] = 'GSE203067'
    malignant.uns['matrix_semantics'] = 'raw nonnegative integer UMI counts'
    malignant.uns['malignant_definition'] = 'primary-tumor origin plus frozen epithelial-marker rule; author publication confirmed malignant epithelium by CNV'
    flow = pd.DataFrame([{'cohort': 'GSE203067', 'step': 'public_all_cell_matrix', 'cells': adata.n_obs}, {'cohort': 'GSE203067', 'step': 'primary_tumor_origin', 'cells': int(primary.sum())}, {'cohort': 'GSE203067', 'step': 'published_qc', 'cells': primary_qc.n_obs}, {'cohort': 'GSE203067', 'step': 'frozen_epithelial_rule', 'cells': int(epithelial.sum())}, {'cohort': 'GSE203067', 'step': 'malignant_ready', 'cells': malignant.n_obs}])
    inclusion = primary_qc.obs.reset_index(names='cell_id')
    return (malignant, primary_qc, flow, inclusion)

def cohort_summary(cohort: str, malignant: ad.AnnData, total_cells: int, samples: int, definition: str) -> dict[str, object]:
    matrix = sparse.csr_matrix(malignant.X)
    return {'cohort': cohort, 'patients': malignant.obs.patient_id.nunique(), 'samples': samples, 'total_public_cells': total_cells, 'malignant_cells': malignant.n_obs, 'genes_in_matrix': malignant.n_vars, 'genes_detected_in_malignant': int(np.asarray((matrix > 0).sum(axis=0)).ravel().astype(bool).sum()), 'median_genes_per_cell': float(np.median(malignant.obs.n_genes)), 'median_umi_per_cell': float(np.median(malignant.obs.n_counts)), 'median_mito_percent': float(np.median(malignant.obs.pct_mito)), 'malignant_definition': definition}

def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    (g199, flow199, inc199) = read_gse199654()
    (g203, g203_all, flow203, inc203) = read_gse203067()
    if g199.obs.patient_id.nunique() < 3 or g203.obs.patient_id.nunique() < 3:
        raise RuntimeError('Patient-level validation gate failed')
    if g199.n_obs == 0 or g203.n_obs == 0:
        raise RuntimeError('Malignant-cell identity gate failed')
    g199.write_h5ad(PROCESSED / 'GSE199654_malignant_ready.h5ad', compression=None)
    g203.write_h5ad(PROCESSED / 'GSE203067_malignant_ready.h5ad', compression=None)
    g203_all.write_h5ad(PROCESSED / 'GSE203067_primary_all_qc.h5ad', compression=None)
    pd.concat([flow199, flow203], ignore_index=True).to_csv(PROCESSED / 'cell_inclusion_flow.tsv', sep='\t', index=False)
    inc199.to_csv(PROCESSED / 'GSE199654_cell_inclusion_audit.tsv', sep='\t', index=False)
    inc203.to_csv(PROCESSED / 'GSE203067_cell_inclusion_audit.tsv', sep='\t', index=False)
    summaries = [cohort_summary('GSE199654', g199, 5070, 14, 'author tumor epithelial + reproducible published QC'), cohort_summary('GSE203067', g203, 85263, 4, 'primary-tumor origin + frozen epithelial marker rule')]
    pd.DataFrame(summaries).to_csv(TABLES / 'external_cohort_characteristics.tsv', sep='\t', index=False)
    patient_counts = []
    for (cohort, data) in (('GSE199654', g199), ('GSE203067', g203)):
        for ((patient, sample), count) in data.obs.groupby(['patient_id', 'sample_id'], observed=True).size().items():
            patient_counts.append({'cohort': cohort, 'patient_id': patient, 'sample_id': sample, 'malignant_cells': int(count)})
    pd.DataFrame(patient_counts).to_csv(TABLES / 'malignant_cells_by_patient_sample.tsv', sep='\t', index=False)
    raw_files = [p for p in RAW.rglob('*') if p.is_file() and (not p.name.endswith(('.partial', '.download')))]
    pd.DataFrame([{'file': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(raw_files)]).to_csv(RAW / 'downloaded_source_checksums.tsv', sep='\t', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.concat([flow199, flow203], ignore_index=True).to_string(index=False))
if __name__ == '__main__':
    main()
