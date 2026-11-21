# Methods and parameters

`data/metadata/current_parameters.json` records the current settings and seeds. Module configurations in `data/metadata/config/` preserve original computational choices. The dataset catalog distinguishes submitted results from inputs used to reconstruct them.

## Independent discovery and source representation

cNMF 1.7.1 selected 2,000 variable genes independently per cohort. Candidate ranks used 20 pilot fits; selected K=10/8 used 100 formal fits, density threshold 0.5 and local-neighborhood fraction 0.30, Frobenius loss and at most 1,000 iterations. Completed fits are distinct from retained component spectra (994/1,000 and 800/800). Across 16,143 shared genes, S=[max(Spearman rho,0)+cosine similarity+Top50 intersection/50]/3. Joint gene-label permutation uses 10,000 upper-tail replicates, P=(1+exceedances)/10,001 and BH80. Seven accepted reciprocal matches also require >=2 supported component metrics and no technical flag. Bidirectional margins compare each winning partner with its strongest alternative.

RAW normalizes summed within-cell normalized usage; EQUAL normalizes mean usage within each patient group. Maximum share>=0.80 is a concentration warning. Inverse-Simpson effective group number is 1/sum(share²); Shannon summaries remain separately named. The six GSE188900 groups and 15 artificial single-pair aggregations form 16 sensitivity settings. Donor identifiers remain unresolved where the source record does not provide them; the analysis does not fabricate a donor crosswalk or change the six-group calculation.

GSE196756 retains the historical 489-cell set, repaired 992 ALL-normalization and 970 AUTO-normalization candidates, and 625 constrained candidates. ALL/AUTO denotes the normalization denominator; both repaired configurations use autosomal windows. CNV repair uses counts/10,000, log1p, 100-gene windows, step 10, clipping 3, dynamic threshold 1.5 and a full patient block. Candidate scores exceed matched-normal q99; further epithelial/PTPRC/outlier/chromosome filters define the constrained set. These selections do not establish malignant identity.

## External RNA and patient-count claims

Expression scores use patient pseudobulk percentile ranks and detection-matched random sets. They are distinct from within-patient correlation Q. Reference Q uses RAW_RANK/HIST, >=20 cells, at most 500 cells, gene detection in >=3 cells and >=5%, >=10 variable genes and >=20% of platform-available members. Patient excess correlation is relative to 1,000 matched sets; mean Q gives equal weight to evaluable patients. Cohort tests use separate BH7 families by cohort, size and measurement specification. Percentile intervals use 2,000 patient resamples. Original reference intervals and separately seeded harmonized intervals are not merged.

Patient p=(1+number of matched statistics>=observed−1e−12)/(B+1). Missing patients retain p=1 in the enrolled denominator. Partial conjunction is min(1,(N−u+1)p_(u)), with u=2 or u=8 of N=14 and separate Holm7 correction. First-step floors are 91/1,001 for PC2 and 49/1,001 for PC8. PC2 is resolution-limited; only reference C01/C02 pass PC8. GSE203067 remains descriptive with 2/4 evaluable patients. Nine measurement specifications per size retain separate depth/background results.

## Bulk, histology and clinical endpoints

Array reporters are uniquely mapped to HGNC, median-collapsed within specimen and scored by mean within-profile percentile rank. GSE53622/GSE53624 contribute 60/119 pairs. Ten prespecified tumor-only context comparisons per size use BH10; paired tissue shifts use BH7. Context sets remove the frozen Top50 union. Histology comprises 34 patient–state observations in 14 patients; failed/singular REML models use patient-fixed-effect regression with patient-cluster-robust errors. Slope P values use 10,000 within-patient permutations. Historical BH5/BH2 and BH7 sensitivity are kept distinct; state-pair median differences use 10,000 paired bootstraps and separate secondary families.

Top50 grade/TNM/N results form one 21-test BH family. Seven unavailable distant-metastasis tests remain blocked. The 179-patient, 106-event Cox series uses separate BH7 families for univariable and adjusted models. Adjusted models include age, sex and categorical stage with source-subseries-stratified baseline hazards. All non-significant results remain available.

## Proteins and PDOs

The primary 124-pair protein estimand is the mean available member tumor-minus-adjacent log2 difference within each patient, then dz=mean(patient delta)/sample_SD(patient delta). There are 5,000 patient-pair bootstraps and 10,000 abundance/missingness-matched sets; the matched test uses two-sided displacement from its null median with BH7 by size. Coverage>=0.60, an interval excluding zero and matched q<0.05 qualify the protein gate. DCI=abs(sum_g mean_gene_delta)/sum_g abs(mean_gene_delta). C07 Top50 has 24 positive and 16 negative measured proteins. Global gene-z results are a separately labeled historical estimand.

The second proteome uses 57 pairs; fixed80 admission requires measurements in >=80% of 120 specimens and nonzero variance. Its actual manuscript analyses comprise patient scores, coverage, gene effects, RNA–protein correlation and shared-gene comparisons. Across-gene correlations are descriptive. No unperformed second-cohort program-level dz family is implied.

All seven programs at each size are tested against author cisplatin AUC in ten PDOs using absolute two-sided Spearman and all 10! labeled assignments, then BH7. C07 Top50 rho=−0.8424242424, P=0.003705357142857 and q=0.0259375. Author and linear-trapezoid AUC ranks agree; ESCCO15's numerical difference is retained. Ten LOO, 45 C07 L2O per size, eight NRF2-WT models and canonical IFN controls remain nested sensitivity analyses. The Day-15 membership table contains a 13-model union, of which 12 have fixed-dose cisplatin measurements and nine overlap AUC. Dynamic programming recounts all 9!/12! labeled tails, including ties.

## Evidence rules and controlled worlds

The descriptive EDSC order is: missing required D/R/Q/P → INDETERMINATE; failed discovery, concentration caution or failed historical Q → limited/cautioned support; qualified P → joint RNA/protein contrast support; otherwise RNA-domain support. PDO, tissue context and narrative prominence do not alter the groups. Current mean-Q support and PC8 are separate claims.

Three mechanisms each contain 160 independent worlds (480 total). Within-world programs and paired depth conditions are dependent records. False majority calls and broad/weak-program power loss are both retained. Four strategies evaluate 56 program–claim cells per size; domain masking distinguishes own-claim loss from lost cross-claim qualifications. The exact parameter-grid generator reconstructs the archived 967,680 rows from frozen inputs and thresholds, verifies SHA256 2490b448a3b04eabb165d9a9cc275c29a2f79ff3ae029a838dcd9658c6106c79, and retains the scientific summaries without committing a 119-MiB cache. No independent external validation or universal program ranking is claimed.

## Validation scope

The portable command reconstructs observed discovery statistics, patient-level Q/partial conjunction, paired protein dz/DCI, exact PDO/Day-15 tests, complete clinical families and domain-mask summaries from included frozen records. Script syntax, hashes, dimensions and final figure input paths are checked. Raw cNMF/CNV fitting, full bootstrap refitting, initial matched-set generation, count-level simulation generation and figure rendering were not run during repository cleanup. Their code, inputs or official acquisition requirements are identified separately; unavailable execution is not labeled as a successful rerun.
