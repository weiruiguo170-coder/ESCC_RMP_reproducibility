# Data definitions

`datasets.tsv` lists scientific purpose, supplement mapping, observation unit, dimensions and logical SHA256 for each dataset. `data/metadata/catalog.json` additionally maps logical publication filenames to current physical paths. A partitioned table repeats its header in each part; the reader includes the first header once and concatenates all data rows in order. Every logical checksum is verified before analysis.

Programs are C01–C07, often prefixed RMP_. Frozen Top100 lists contain the ordered Top50 prefix. Empty numeric fields indicate missing, unavailable or ineligible measurements, never zero support. `NOT_EVALUABLE`, `LIMITED_SUPPORT`, `NOT_ESTABLISHED` and `SUPPORTED_WITHIN_SCOPE` are distinct states.

| Field family | Meaning |
|---|---|
| `rmp_id`, `RMP`, `size`, `signature_size` | Program identity and frozen member count |
| `cohort`, `patient_id`, `unit_id`, `PDO_ID` | Source-specific observation identity; no new donor identity is inferred |
| `aggregate_normalized_usage`, `mean_normalized_usage` | Within-source cell-sum and mean normalized cNMF usage |
| `RAW`, `EQUAL`, maximum contributions | Cell-weighted and equal-patient-group representation |
| `composite_similarity`, matching margins | S and differences from directional next-best component matches |
| `Q_effect`, `mean_patient_effect`, `relative_q` | Patient-mean or patient-level excess member correlation over the specified null |
| `method`, `background` | Separate depth/rank and null-background specification; RAW_RANK/HIST is the reference |
| `n_evaluable`, `n_original`, `n_patients_original` | Evaluable and enrolled denominators; missing patients remain in enrolled PC tests |
| `p_majority`, `holm_majority`, `state_majority` | Nominal, Holm7 and evaluability-aware patient-majority claims |
| `raw_paired_delta`, `Cohens_dz`, `protein_dz` | Raw patient-pair member delta and its standardized patient-first effect |
| `CI_low/high` | Interval for the row-specific native estimand; mean-delta and dz intervals are distinct |
| `dz_CI_low/high` | Bootstrap interval for standardized paired dz |
| `matched_null_P/FDR`, `protein_matched_null_q` | Matched-gene-set inference, separate from direct paired tests |
| `coverage`, `mapped_genes`, `fixed80` | Assay member availability and predefined admission; not proof of no effect in absent proteins |
| `DCI`, `P_DCI` | abs(sum gene-mean paired deltas)/sum(abs gene-mean paired deltas) |
| `rho`, `extreme`, `permutations`, `P_exact`, `q7` | Rank correlation, labeled null counts, exact tail probability and within-family BH7 |
| `D_pass`, `R_caution`, `Q_pass`, `P_pass` | Discovery, concentration, historical RNA and qualified protein conditions |
| `EDSC_group`, `frozen_B_category` | Final descriptive group and retained historical rule label, respectively; not an overall rank |
| `reference_*` in parameter-grid retention | Reference classifier paths, determinate counts and retention denominators |

RNA expression z, within-patient coordination Q, bulk percentile scores, raw-paired protein dz and gene-wise global-z are different estimands. S4 reference intervals and separately seeded S10 harmonized intervals remain separate. Historical `old_*`, discovery-tier and three-program PDO comparison fields are source annotations and do not replace current seven-program exact inference.

Current printed Tables S10F–G use some datasets whose original logical names began S5. `supplementary_tables.tsv` explicitly records that cross-reference. Analysis units are patients or pairs for clinical/tissue inference, patient–states for histology, genes for cross-cohort protein comparisons, PDOs for drug response, and independent worlds for simulation uncertainty.
