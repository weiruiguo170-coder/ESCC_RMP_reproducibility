# ESCC recurrent malignant programs

**Patient-aware multidimensional evidence assessment of recurrent malignant transcriptional programs in esophageal squamous cell carcinoma**  
Ruiguo Wei and Gaofeng Liang

## Overview

This repository contains frozen transcriptional signatures, patient-level inputs, complete statistical results and figure source data for seven recurrent malignant programs (C01–C07) in esophageal squamous cell carcinoma. The workflow distinguishes recurrence of fitted gene-loading structure from patient representation, external RNA coordination, paired tissue effects and exploratory PDO cisplatin associations. Evidence is reported by domain and measurement condition; the descriptive EDSC bridge is not an independently calibrated universal ranking algorithm.

## Datasets and provenance

Independent discovery uses GSE160269 (60 patients; 44,547 cells) and GSE188900 (six patient analysis groups P1–P6; 6,801 cells). GSE196756 remains an auxiliary tumor-associated epithelial/CNV audit. External RNA uses GSE199654 and GSE203067; the latter has only two of four evaluable patients and is descriptive. Paired bulk RNA comprises GSE53622 and GSE53624 within GSE53625 (60+119=179 pairs). Protein evidence uses PXD021701/IPX0002501000 (124 pairs) and PXD035562/PRJNA861875 (57 pairs). PDO expression is GSE269447, with ten author-AUC models and 12 Day-15 models, nine overlapping the AUC series.

The primary proteome contains the complete seven-program Top50/Top100 patient-first paired-effect families. The second cohort provides the analyses actually used in the manuscript: patient RNA/protein scores, gene effects, coverage and cross-cohort comparisons. Public source identifiers and original analysis labels are preserved. Six GSE188900 groups do not imply an additional independent-donor identity audit.

## Repository structure

| Path | Contents |
|---|---|
| `data/processed/` | Frozen signatures, discovery matrices, patient/null/simulation records and final figure inputs |
| `data/metadata/` | Dataset catalog, current parameters, original configurations and decision-rule inputs |
| `results/tables/` | Submitted results grouped by scientific domain, including null and sensitivity results |
| `scripts/verify.py` | Portable verification and statistical replay |
| `scripts/replay.py` | Original 82-check endpoint reconstruction |
| `scripts/analysis/` | Necessary raw-data generating analyses, organized by subject |
| `scripts/evidence/` | Frozen classification rules and exact parameter-grid reconstruction |
| `scripts/plot_figures.py` | One entry point for all final numbered figures |
| `scripts/plotting/` | Shared final plotting implementations |
| `docs/` | Methods, data definitions, access instructions and table/figure indexes |

No raw sequencing matrix, original clinical workbook or complete third-party gene-set library is redistributed. All included datasets are ordinary text files. Larger tables are split at complete rows below 24 MiB per file; the catalog reader reconstructs their exact logical contents.

## Installation

The portable workflow was tested with Python 3.10.11 and the package versions pinned in `requirements.txt`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts/verify.py
```

On macOS/Linux use `.venv/bin/python` in place of `.venv\Scripts\python`. Raw-data analyses additionally require `requirements-upstream.txt`; the optional author-RDS metadata extractor requires R. Its original R version was not recorded.

## Reproduction workflow

1. Run `scripts/verify.py`. It verifies dataset hashes and dimensions, all 14 ordered signature lists, source contributions and all 80 observed matches; reconstructs the parameter grid; replays patient Q/PC2/PC8, paired proteins and exact PDO tests; and verifies clinical families, domain masks and figure inputs.
2. Optional: use `python scripts/verify.py --output <directory>` to retain regenerated verification tables. The default run uses temporary output and leaves the repository clean.
3. To start from source assays, obtain the materials and preserve the exact preprocessing described in [data access](docs/data_access.md), then follow preprocessing → independent cNMF → matching/signature freeze → patient RNA → tissue/protein/PDO analyses. The source code and frozen configurations are supplied; the portable check does not rerun raw cNMF/CNV fits, full bootstrap model series or count-level simulations.

The archived parameter-grid cache would occupy 124,728,960 uncompressed bytes and is therefore regenerated rather than committed. `python scripts/evidence/grid.py` reconstructs all 967,680 rows and checks their exact original SHA256 while retaining only the 56-row scientific summary. An explicit `--write-full <path>` exports the large plain TSV when needed; keep that generated cache outside Git.

## Output tables and figures

[datasets.tsv](docs/datasets.tsv) lists each dataset once with its scientific purpose, analysis unit, supplement mapping, dimensions and logical checksum. [supplementary_tables.tsv](docs/supplementary_tables.tsv) maps Tables S1–S10 to their current files; filenames no longer depend on a flat list of supplement numbers. The formal supplement comprises one Online Resource 1 with Methods M1–M8, Tables S1–S10 and Figures S1–S13.

[figures.tsv](docs/figures.tsv) maps Main Figures 1–9 and Supplementary Figures S1–S13 to the final source tables, original submitted-image checksums and rendering command. Render one figure in a separate process:

```text
python scripts/plot_figures.py Fig1 --output results/figures
```

Replace Fig1 with the indexed figure. The repository preparation checked source inputs and scripts; it did not regenerate figures. [Methods](docs/methods.md) and the [data dictionary](docs/data_dictionary.md) explain units, statistical families, missingness and inferential limits.

## Data availability

Included records are deidentified derived research outputs. Raw data and source-study workbooks remain available through the accessions and source articles listed in [data_access.md](docs/data_access.md), subject to their original terms. Full enrichment outputs are included; complete reference libraries are obtained from their named sources and checked against the frozen resource hashes. The existing LICENSE notice is preserved unchanged; no additional license is assigned to source-study data or third-party resources.

## Citation

Wei R and Liang G. *Patient-aware multidimensional evidence assessment of recurrent malignant transcriptional programs in esophageal squamous cell carcinoma*. Associated manuscript; publication details pending. Cite the repository version used alongside the manuscript. `CITATION.cff` provides the verified title and authors. No unconfirmed publication DOI, remote repository address or publication status is asserted in this snapshot.
