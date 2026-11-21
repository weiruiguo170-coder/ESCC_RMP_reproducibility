from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import ahocorasick
import numpy as np
import pandas as pd
SEED = 20260827
PROJECT_ROOT = INPUT_ROOT
OUT = PROJECT_ROOT / 'results' / 'bulk_context'
DISC = OUT / '01_dataset_discovery'
AUDIT = OUT / '00_audit'
PROC = OUT / '03_processed_spatial_data'
COV = OUT / '04_signature_coverage'
LOG = OUT / '13_logs'
GPL18109 = DISC / 'GPL18109_full_official.txt'
GPL19748 = DISC / 'GPL19748_full_official.txt'
FEATURE_EXTRACTION = DISC / 'GSM1297313_raw_probe_map.txt'
CDNA = DISC / 'Homo_sapiens.GRCh38.cdna.all.release112.fa.gz'
HGNC = DISC / 'hgnc_complete_set_20260827.txt'
RMP_IDS = [f'RMP_C{i:02d}' for i in range(1, 8)]
PRIMARY = {'RMP_C01', 'RMP_C02', 'RMP_C04', 'RMP_C05', 'RMP_C07'}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_soft_table(path: Path) -> pd.DataFrame:
    lines = []
    in_table = False
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            line = line.rstrip('\r\n')
            if line == '!platform_table_begin':
                in_table = True
                continue
            if line == '!platform_table_end':
                break
            if in_table:
                lines.append(line)
    if not lines:
        raise RuntimeError(f'No platform table in {path}')
    reader = csv.DictReader(lines, delimiter='\t')
    return pd.DataFrame(list(reader), dtype=str).fillna('')

def normalize_sequence(value: str) -> str:
    return re.sub('[^ACGTN]', '', (value or '').upper())

def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]

def parse_feature_extraction(path: Path, wanted: set[str]) -> pd.DataFrame:
    rows = []
    header = None
    with path.open('r', encoding='utf-8', errors='replace', newline='') as handle:
        reader = csv.reader(handle, delimiter='\t')
        for row in reader:
            if not row:
                continue
            if row[0] == 'FEATURES':
                header = row[1:]
                continue
            if header is None or row[0] != 'DATA':
                continue
            d = dict(zip(header, row[1:]))
            feature = d.get('FeatureNum', '')
            if feature not in wanted:
                continue
            rows.append({'feature_number': feature, 'fe_row': d.get('Row', ''), 'fe_col': d.get('Col', ''), 'fe_control_type': d.get('ControlType', ''), 'fe_probe_name': d.get('ProbeName', ''), 'fe_spot_id': d.get('SystematicName', '') or d.get('GeneName', ''), 'fe_sequence': normalize_sequence(d.get('Sequence', ''))})
    return pd.DataFrame(rows).drop_duplicates('feature_number', keep='first')

def build_official_crosswalk(gpl18109: pd.DataFrame, gpl19748: pd.DataFrame) -> pd.DataFrame:
    a = gpl18109.rename(columns={'ID': 'feature_number', 'COL': 'gpl18109_col', 'ROW': 'gpl18109_row', 'SPOT_ID': 'spot_id', 'Control Type': 'gpl18109_control', 'SEQUENCE': 'sequence'}).copy()
    a['sequence'] = a['sequence'].map(normalize_sequence)
    a['is_control'] = ~a['gpl18109_control'].str.upper().isin(['FALSE', '0'])
    b = gpl19748.rename(columns={'ID': 'agilent_probe_name', 'SPOT_ID': 'spot_id', 'ControlType': 'gpl19748_control', 'Sequence': 'sequence', 'Description': 'gpl19748_description'}).copy()
    b['sequence'] = b['sequence'].map(normalize_sequence)
    b_non = b[b['gpl19748_control'].str.upper().isin(['0', 'FALSE'])].copy()
    key_to_probe_spot: dict[tuple[str, str], set[str]] = defaultdict(set)
    key_to_probe_name: dict[tuple[str, str], set[str]] = defaultdict(set)
    seq_to_probe: dict[str, set[str]] = defaultdict(set)
    for r in b_non.itertuples(index=False):
        key_to_probe_spot[r.spot_id, r.sequence].add(r.agilent_probe_name)
        key_to_probe_name[r.agilent_probe_name, r.sequence].add(r.agilent_probe_name)
        if r.sequence:
            seq_to_probe[r.sequence].add(r.agilent_probe_name)
    rows = []
    for r in a.itertuples(index=False):
        exact_spot = key_to_probe_spot.get((r.spot_id, r.sequence), set()) if not r.is_control else set()
        exact_name = key_to_probe_name.get((r.spot_id, r.sequence), set()) if not r.is_control else set()
        exact = sorted(exact_spot | exact_name)
        seq_only = sorted(seq_to_probe.get(r.sequence, set())) if not r.is_control and r.sequence else []
        if r.is_control:
            (status, probe) = ('CONTROL', '')
        elif len(exact) == 1:
            (status, probe) = ('EXACT_IDENTITY_SEQUENCE', exact[0])
        elif len(exact) > 1:
            (status, probe) = ('AMBIGUOUS_OFFICIAL_TUPLE', '')
        elif len(seq_only) == 1:
            (status, probe) = ('SEQUENCE_ONLY_CANDIDATE', seq_only[0])
        elif len(seq_only) > 1:
            (status, probe) = ('AMBIGUOUS_SEQUENCE', '')
        else:
            (status, probe) = ('UNMATCHED_OFFICIAL', '')
        rows.append({'feature_number': r.feature_number, 'gpl18109_row': r.gpl18109_row, 'gpl18109_col': r.gpl18109_col, 'spot_id': r.spot_id, 'control_type': r.gpl18109_control, 'is_control': r.is_control, 'probe_sequence': r.sequence, 'agilent_probe_name': probe, 'official_crosswalk_status': status, 'official_identity_route': 'BOTH_EQUIVALENT' if len(exact_name) == 1 and len(exact_spot) == 1 else 'GPL18109_SPOT_ID_EQUALS_GPL19748_ID' if len(exact_name) == 1 else 'GPL18109_SPOT_ID_EQUALS_GPL19748_SPOT_ID' if len(exact_spot) == 1 else 'NONE_OR_AMBIGUOUS', 'official_exact_candidate_count': len(exact), 'sequence_only_candidate_count': len(seq_only)})
    return pd.DataFrame(rows)

def load_hgnc(path: Path) -> tuple[dict[str, str], set[str]]:
    table = pd.read_csv(resolve_data(path), sep='\t', dtype=str, low_memory=False).fillna('')
    approved = table[table['status'] == 'Approved'].copy()
    ensembl_to_symbol: dict[str, str] = {}
    conflicts = set()
    for r in approved.itertuples(index=False):
        if not r.ensembl_gene_id:
            continue
        for ens in str(r.ensembl_gene_id).split('|'):
            ens = ens.split('.')[0]
            if ens in ensembl_to_symbol and ensembl_to_symbol[ens] != r.symbol:
                conflicts.add(ens)
            else:
                ensembl_to_symbol[ens] = r.symbol.upper()
    for ens in conflicts:
        ensembl_to_symbol.pop(ens, None)
    return (ensembl_to_symbol, set(approved['symbol'].str.upper()))

def exact_sequence_to_gene(sequences: set[str], cdna_path: Path, ensembl_to_symbol: dict[str, str]):
    pattern_to_original: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for seq in sequences:
        if len(seq) < 25 or set(seq) - set('ACGT'):
            continue
        pattern_to_original[seq].append((seq, 'FORWARD'))
        rc = reverse_complement(seq)
        pattern_to_original[rc].append((seq, 'REVERSE_COMPLEMENT'))
    automaton = ahocorasick.Automaton()
    for pattern in pattern_to_original:
        automaton.add_word(pattern, pattern)
    automaton.make_automaton()
    sequence_hits: dict[str, set[str]] = defaultdict(set)
    orientation_hits: dict[str, set[str]] = defaultdict(set)

    def process_record(header: str, seq: str) -> None:
        if not header or not seq:
            return
        gm = re.search('\\bgene:(ENSG\\d+)', header)
        if not gm:
            return
        gene_id = gm.group(1)
        symbol = ensembl_to_symbol.get(gene_id)
        if not symbol:
            return
        for (_, pattern) in automaton.iter(seq):
            for (original, orientation) in pattern_to_original[pattern]:
                sequence_hits[original].add(symbol)
                orientation_hits[original].add(orientation)
    (header, chunks) = ('', [])
    transcript_count = 0
    with gzip.open(cdna_path, 'rt', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if line.startswith('>'):
                if header:
                    process_record(header, ''.join(chunks).upper())
                    transcript_count += 1
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header:
            process_record(header, ''.join(chunks).upper())
            transcript_count += 1
    return (sequence_hits, orientation_hits, transcript_count)

def coverage_table(genes: set[str]) -> pd.DataFrame:
    rows = []
    for rmp in RMP_IDS:
        for size in (50, 100):
            path = PROJECT_ROOT / 'signatures' / f'{rmp}_signature_top{size}.txt'
            frozen = [x.strip().upper() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
            detected = [g for g in frozen if g in genes]
            missing = [g for g in frozen if g not in genes]
            pct = 100.0 * len(detected) / len(frozen)
            label = 'GOOD' if pct >= 80 else 'ACCEPTABLE' if pct >= 70 else 'CAUTION' if pct >= 60 else 'LOW'
            rows.append({'cohort': 'GSE53625', 'RMP': rmp, 'RMP_class': 'PRIMARY' if rmp in PRIMARY else 'EXPLORATORY', 'signature_size': f'TOP{size}', 'total_frozen_genes': len(frozen), 'genes_detected': len(detected), 'genes_missing': len(missing), 'coverage_percent': pct, 'coverage_label': label, 'detected_genes': ';'.join(detected), 'missing_genes': ';'.join(missing)})
    return pd.DataFrame(rows)

def main() -> None:
    required = [GPL18109, GPL19748, FEATURE_EXTRACTION, CDNA, HGNC]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    g18109 = read_soft_table(GPL18109)
    g19748 = read_soft_table(GPL19748)
    crosswalk = build_official_crosswalk(g18109, g19748)
    fe = parse_feature_extraction(FEATURE_EXTRACTION, set(crosswalk['feature_number']))
    crosswalk = crosswalk.merge(fe, on='feature_number', how='left', validate='one_to_one')
    crosswalk['fe_feature_found'] = crosswalk['fe_probe_name'].notna()
    crosswalk['fe_spot_match'] = crosswalk['is_control'] | ((crosswalk['spot_id'] == crosswalk['fe_spot_id']) | (crosswalk['spot_id'] == crosswalk['fe_probe_name']))
    crosswalk['fe_sequence_match'] = crosswalk['is_control'] | (crosswalk['probe_sequence'] == crosswalk['fe_sequence'])
    crosswalk['fe_probe_name_match'] = crosswalk['is_control'] | (crosswalk['agilent_probe_name'] != '') & (crosswalk['agilent_probe_name'] == crosswalk['fe_probe_name'])
    crosswalk['feature_extraction_concordant'] = crosswalk['fe_feature_found'] & crosswalk['fe_spot_match'] & crosswalk['fe_sequence_match'] & crosswalk['fe_probe_name_match'] | crosswalk['is_control'] & crosswalk['fe_feature_found']
    (ensembl_to_symbol, approved_symbols) = load_hgnc(HGNC)
    noncontrol_sequences = set(crosswalk.loc[~crosswalk['is_control'], 'probe_sequence']) - {''}
    (sequence_hits, orientation_hits, transcript_count) = exact_sequence_to_gene(noncontrol_sequences, CDNA, ensembl_to_symbol)
    (gene_symbols, gene_ids, orientations, mapping_status) = ([], [], [], [])
    for r in crosswalk.itertuples(index=False):
        if r.is_control:
            gene_symbols.append('')
            gene_ids.append('')
            orientations.append('')
            mapping_status.append('CONTROL')
            continue
        hits = sorted(sequence_hits.get(r.probe_sequence, set()))
        orient = sorted(orientation_hits.get(r.probe_sequence, set()))
        official_ok = r.official_crosswalk_status == 'EXACT_IDENTITY_SEQUENCE' and bool(r.feature_extraction_concordant)
        if not official_ok:
            (status, symbol) = ('OFFICIAL_CROSSWALK_CONFLICT', '')
        elif len(hits) == 1 and hits[0] in approved_symbols:
            (status, symbol) = ('UNIQUE_ONE_TO_ONE', hits[0])
        elif len(hits) > 1:
            (status, symbol) = ('AMBIGUOUS_GENE', '')
        else:
            (status, symbol) = ('UNMAPPED_REPORTER', '')
        gene_symbols.append(symbol)
        gene_ids.append(';')
        orientations.append(';'.join(orient))
        mapping_status.append(status)
    crosswalk['hgnc_symbol'] = gene_symbols
    crosswalk['exact_match_orientation'] = orientations
    crosswalk['gene_mapping_status'] = mapping_status
    crosswalk_path = PROC / 'GPL18109_feature_to_HGNC_crosswalk.tsv'
    crosswalk.to_csv(crosswalk_path, sep='\t', index=False)
    nonctrl = crosswalk[~crosswalk['is_control']]
    unique = nonctrl[nonctrl['gene_mapping_status'] == 'UNIQUE_ONE_TO_ONE']
    ambiguous = nonctrl[nonctrl['gene_mapping_status'] == 'AMBIGUOUS_GENE']
    unmapped = nonctrl[nonctrl['gene_mapping_status'] == 'UNMAPPED_REPORTER']
    conflicts = nonctrl[nonctrl['gene_mapping_status'] == 'OFFICIAL_CROSSWALK_CONFLICT']
    conflict_rate = 100 * len(conflicts) / len(nonctrl)
    fe_concordance = 100 * nonctrl['feature_extraction_concordant'].mean()
    eligible = nonctrl.sort_values('feature_number', key=lambda s: pd.to_numeric(s, errors='coerce'))
    indices = np.linspace(0, len(eligible) - 1, 30, dtype=int)
    spot = eligible.iloc[indices][['feature_number', 'spot_id', 'agilent_probe_name', 'probe_sequence', 'official_crosswalk_status', 'fe_probe_name', 'fe_spot_id', 'fe_sequence', 'feature_extraction_concordant', 'hgnc_symbol', 'gene_mapping_status']]
    spot.to_csv(AUDIT / 'GPL18109_crosswalk_spot_checks.tsv', sep='\t', index=False)
    frozen_genes = set(unique['hgnc_symbol'])
    coverage = coverage_table(frozen_genes)
    coverage.to_csv(COV / 'GSE53625_signature_coverage_after_GPL18109_rescue.tsv', sep='\t', index=False)
    primary50 = coverage[(coverage['RMP_class'] == 'PRIMARY') & (coverage['signature_size'] == 'TOP50')]
    gate_pass = int((primary50['coverage_percent'] >= 70).sum()) >= 4 and bool((primary50['coverage_percent'] >= 60).all()) and (conflict_rate <= 0.5) and (fe_concordance >= 99.0)
    source_rows = []
    urls = {GPL18109.name: 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL18109&targ=self&form=text&view=full', GPL19748.name: 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL19748&targ=self&form=text&view=full', FEATURE_EXTRACTION.name: 'https://www.ebi.ac.uk/biostudies/files/E-GEOD-53625/GSM1297313_US10313827_253831410029_S02_GE2_107_Sep09_1_3b.txt', CDNA.name: 'https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz', HGNC.name: 'https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt'}
    for path in required:
        source_rows.append({'file': path.name, 'official_source': urls[path.name], 'access_date': '2026-08-27', 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    pd.DataFrame(source_rows).to_csv(AUDIT / 'GPL18109_annotation_source_manifest.tsv', sep='\t', index=False)
    counts = crosswalk['gene_mapping_status'].value_counts().to_dict()
    summary = {'status': 'PASS' if gate_pass else 'FAIL', 'total_reporters': int(len(crosswalk)), 'control_reporters': int(crosswalk['is_control'].sum()), 'non_control_reporters': int(len(nonctrl)), 'unique_one_to_one_reporters': int(len(unique)), 'unique_HGNC_genes': int(unique['hgnc_symbol'].nunique()), 'ambiguous_reporters': int(len(ambiguous)), 'unmapped_reporters': int(len(unmapped)), 'official_crosswalk_conflicts': int(len(conflicts)), 'official_crosswalk_conflict_rate_percent': conflict_rate, 'feature_extraction_concordance_percent': fe_concordance, 'ensembl_cdna_transcripts_scanned': int(transcript_count), 'mapping_strategy': 'GPL18109 exact Sequence plus SPOT_ID equal to GPL19748 ProbeName or GPL19748 SPOT_ID; validated by original Feature Extraction; exact bidirectional 60-mer match to Ensembl r112 cDNA; one approved HGNC symbol only', 'aggregation_rule': 'sample-wise median across all uniquely mapped reporters for each HGNC symbol', 'primary_top50_at_least_70': int((primary50['coverage_percent'] >= 70).sum()), 'primary_top50_below_60': int((primary50['coverage_percent'] < 60).sum()), 'mapping_gate': 'PASS' if gate_pass else 'FAIL', 'mapping_status_counts': counts}
    (AUDIT / 'GPL18109_ANNOTATION_RESCUE_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    report = f"# GPL18109 Platform Annotation Rescue Audit\n\nDate: 2026-08-27  \nSeed: `{SEED}`  \nMapping gate: **{('PASS' if gate_pass else 'FAIL')}**\n\n## Official sources and frozen files\n\n- GEO GPL18109 complete platform table: `{GPL18109.name}`\n- GEO GPL19748 complete platform table: `{GPL19748.name}`\n- Original same-platform Agilent Feature Extraction file: `{FEATURE_EXTRACTION.name}`\n- Ensembl GRCh38 release 112 cDNA transcriptome and matching project GTF\n- HGNC complete approved-symbol set downloaded 2026-08-27\n\nSHA256 values and URLs are recorded in `GPL18109_annotation_source_manifest.tsv`.\n\n## Deterministic mapping rule\n\nGPL18109 feature numbers were joined to GPL19748 Agilent ProbeNames by full uppercase `Sequence` plus an exact official identifier relation: GPL18109 `SPOT_ID` equals GPL19748 `ID/ProbeName` or GPL19748 `SPOT_ID`. Table row position was never used. The dual identity rule is required by the official tables because GPL18109 stores ProbeName for some reporter classes and systematic identity for others. The join was independently checked against FeatureNum, ProbeName, systematic name, full sequence, ControlType, and available row/column fields in the original Feature Extraction file. Full-length probe sequences were then matched exactly, in both orientations, to Ensembl release 112 cDNA transcripts. Only reporters whose exact transcript hits resolved to one current approved HGNC symbol were accepted.\n\nNo alias substitution, mismatch-tolerant rescue, frozen-gene-specific selection, or manual replacement was used.\n\n## Crosswalk counts\n\n- Total reporters: **{len(crosswalk):,}**\n- Non-control reporters: **{len(nonctrl):,}**\n- Controls: **{crosswalk['is_control'].sum():,}**\n- Unique one-to-one reporter mappings: **{len(unique):,}**\n- Unique approved HGNC genes: **{unique['hgnc_symbol'].nunique():,}**\n- Ambiguous gene mappings: **{len(ambiguous):,}**\n- Unmapped reporters: **{len(unmapped):,}**\n- Official crosswalk conflicts: **{len(conflicts):,}** ({conflict_rate:.4f}%)\n- Feature Extraction concordance: **{fe_concordance:.4f}%**\n\n## Multiple-probe aggregation rule\n\nThe pre-specified genome-wide aggregation is the sample-wise median of all uniquely mapped reporters for each approved HGNC symbol. No RMP-specific probe choice is permitted.\n\n## Coverage gate\n\nPrimary Top50 pass rule was fixed before rescue: at least four of five primary RMPs at >=70%, none below 60%, official conflict rate <=0.5%, and Feature Extraction concordance >=99%.\n\nPrimary Top50 programs at >=70%: **{int((primary50['coverage_percent'] >= 70).sum())}/5**  \nPrimary Top50 programs below 60%: **{int((primary50['coverage_percent'] < 60).sum())}/5**\n\n**GSE53625 ANNOTATION/MAPPING GATE: {('PASS' if gate_pass else 'FAIL')}**\n\nPatient-level RMP scoring is {('authorized to resume using the frozen rules' if gate_pass else 'not authorized; a cleaner gene-annotated ESCC RNA-seq fallback is required')}.\n"
    (AUDIT / 'GPL18109_ANNOTATION_RESCUE_AUDIT.md').write_text(report, encoding='utf-8')
    (LOG / 'gpl18109_annotation_rescue.log').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
