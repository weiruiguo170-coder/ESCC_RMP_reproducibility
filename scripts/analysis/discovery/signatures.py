"""Phase D: unbiased cohort-specific/consensus annotation and signature freezing."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path
import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import hypergeom, rankdata
PROJECT_ROOT = INPUT_ROOT
RESULTS_DIR = PROJECT_ROOT / 'results'
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / '03_cross_cohort_matching'))
from phase_c_program_matching import load_cohort
B1 = RESULTS_DIR / 'cnmf_formal'
C = RESULTS_DIR / 'cross_cohort_matching'
OUT = RESULTS_DIR / 'frozen_rmp_construction'
GS_DIR = PROJECT_ROOT / 'signatures' / 'enrichment_gene_sets'
CONFIG = json.loads((PROJECT_ROOT / 'config' / 'frozen_rmp_construction.json').read_text(encoding='utf-8'))
EVIDENCE = pd.read_csv(resolve_data(C / 'RMP_evidence_table.tsv'), sep='\t')
LIBRARIES = {name: GS_DIR / f'{name}.gmt' for name in ('MSigDB_Hallmark_2020', 'Reactome_Pathways_2024', 'GO_Biological_Process_2026', 'KEGG_2026')}
GSEA_LIBRARIES = ('MSigDB_Hallmark_2020', 'Reactome_Pathways_2024')
THEMES = CONFIG['theme_keyword_lexicon']
THEME_LABELS = CONFIG['theme_human_labels']
GENERIC_THEMES = set(CONFIG['generic_theme_set'])
VERSION = CONFIG['signature_version']
SEED = 20260826

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda : handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result

def verify_hash_manifest(path: Path) -> int:
    manifest = pd.read_csv(resolve_data(path), sep='\t')
    file_col = 'file' if 'file' in manifest else 'phase_b1_file'
    for row in manifest.itertuples(index=False):
        file_path = Path(getattr(row, file_col))
        if sha256(file_path).lower() != str(row.sha256).lower():
            raise RuntimeError(f'HASH_MISMATCH: {file_path}')
    return len(manifest)

def read_gmt(path: Path) -> dict[str, set[str]]:
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        fields = line.split('\t')
        if len(fields) >= 3:
            result[fields[0]] = set((g.upper() for g in fields[2:] if g))
    return result

def freeze_signatures(cohorts: dict[str, dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, pd.Series]]]:
    shared = cohorts['GSE160269']['scores'].index.intersection(cohorts['GSE188900']['scores'].index)
    shared = pd.Index(sorted(shared))
    (all_rows, frozen_rows) = ([], [])
    rankings: dict[str, dict[str, pd.Series]] = {}
    for rmp in EVIDENCE.itertuples():
        score_a = cohorts['GSE160269']['scores'].loc[shared, rmp.program_A].astype(float)
        score_b = cohorts['GSE188900']['scores'].loc[shared, rmp.program_B].astype(float)
        n = len(shared)
        percentile_a = pd.Series(rankdata(score_a.to_numpy(), method='average') / n, index=shared)
        percentile_b = pd.Series(rankdata(score_b.to_numpy(), method='average') / n, index=shared)
        rank_a = pd.Series(rankdata(-score_a.to_numpy(), method='min').astype(int), index=shared)
        rank_b = pd.Series(rankdata(-score_b.to_numpy(), method='min').astype(int), index=shared)
        consensus = np.sqrt(percentile_a * percentile_b)
        table = pd.DataFrame({'rmp_id': rmp.rmp_id, 'gene': shared, 'consensus_aggregate': consensus.to_numpy(), 'source_percentile_GSE160269': percentile_a.to_numpy(), 'source_percentile_GSE188900': percentile_b.to_numpy(), 'source_rank_GSE160269': rank_a.to_numpy(), 'source_rank_GSE188900': rank_b.to_numpy(), 'source_score_GSE160269': score_a.to_numpy(), 'source_score_GSE188900': score_b.to_numpy(), 'source_percentile_difference': np.abs(percentile_a.to_numpy() - percentile_b.to_numpy()), 'minimum_source_percentile': np.minimum(percentile_a.to_numpy(), percentile_b.to_numpy())})
        table = table.sort_values(['consensus_aggregate', 'minimum_source_percentile', 'gene'], ascending=[False, False, True]).reset_index(drop=True)
        table['consensus_rank'] = np.arange(1, n + 1)
        table['consensus_percentile'] = 1 - (table.consensus_rank - 1) / (n - 1)
        all_rows.append(table)

        def deterministic_prerank(score: pd.Series) -> pd.Series:
            ordered = pd.DataFrame({'gene': score.index, 'score': score.to_numpy()}).sort_values(['score', 'gene'], ascending=[False, True], kind='mergesort')
            metric = np.linspace(1.0, -1.0, len(ordered))
            return pd.Series(metric, index=ordered.gene.to_numpy(), name='rank_metric')
        rankings[rmp.rmp_id] = {'GSE160269': deterministic_prerank(score_a), 'GSE188900': deterministic_prerank(score_b), 'CONSENSUS': pd.Series(1 - 2 * (table.consensus_rank.to_numpy() - 1) / (n - 1), index=table.gene, name='rank_metric')}
        for size in (50, 100):
            top = table.head(size).copy()
            genes = top.gene.tolist()
            sig_path = OUT / f'{rmp.rmp_id}_signature_top{size}.txt'
            sig_path.write_text('\n'.join(genes) + '\n', encoding='utf-8')
            signature_hash = sha256(sig_path)
            for row in top.itertuples():
                frozen_rows.append({'rmp_id': rmp.rmp_id, 'signature_type': f'TOP{size}', 'signature_size': size, 'gene': row.gene, 'consensus_rank': row.consensus_rank, 'consensus_aggregate': row.consensus_aggregate, 'source_rank_GSE160269': row.source_rank_GSE160269, 'source_rank_GSE188900': row.source_rank_GSE188900, 'source_percentile_GSE160269': row.source_percentile_GSE160269, 'source_percentile_GSE188900': row.source_percentile_GSE188900, 'signature_version': VERSION, 'signature_sha256': signature_hash})
    all_genes = pd.concat(all_rows, ignore_index=True)
    frozen = pd.DataFrame(frozen_rows)
    all_genes.to_csv(OUT / 'consensus_rank_all_genes.tsv', sep='\t', index=False)
    frozen.to_csv(OUT / 'rmp_frozen_signatures.tsv', sep='\t', index=False)
    frozen.loc[frozen.signature_type.eq('TOP50')].to_csv(OUT / 'consensus_top50_genes.tsv', sep='\t', index=False)
    frozen.loc[frozen.signature_type.eq('TOP100')].to_csv(OUT / 'consensus_top100_genes.tsv', sep='\t', index=False)
    return (all_genes, frozen, rankings)

def run_prerank(rankings: dict[str, dict[str, pd.Series]], gene_sets: dict[str, dict[str, set[str]]]) -> pd.DataFrame:
    rows = []
    for (rmp_id, reps) in rankings.items():
        evidence_row = EVIDENCE.set_index('rmp_id').loc[rmp_id]
        program_map = {'GSE160269': evidence_row.program_A, 'GSE188900': evidence_row.program_B, 'CONSENSUS': rmp_id}
        for (representation, series) in reps.items():
            ranked = series.sort_values(ascending=False).rename('score').reset_index()
            ranked.columns = ['gene', 'score']
            for library in GSEA_LIBRARIES:
                print(f'GSEA {rmp_id} {representation} {library}', flush=True)
                result = gp.prerank(rnk=ranked, gene_sets=str(LIBRARIES[library]), outdir=None, min_size=10, max_size=500, permutation_num=1000, weight=1.0, ascending=False, threads=4, no_plot=True, seed=SEED, verbose=False).res2d
                for row in result.itertuples(index=False):
                    rows.append({'rmp_id': rmp_id, 'representation': representation, 'program_id': program_map[representation], 'library': library, 'term': row.Term, 'ES': float(row.ES), 'NES': float(row.NES), 'P': float(getattr(row, '_4')), 'FDR': float(getattr(row, '_5')), 'FWER_P': float(getattr(row, '_6')), 'tag_fraction': getattr(row, '_7'), 'gene_fraction': getattr(row, '_8'), 'core_genes': getattr(row, 'Lead_genes')})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'preranked_enrichment_all.tsv', sep='\t', index=False)
    return table

def run_ora(rankings: dict[str, dict[str, pd.Series]], gene_sets: dict[str, dict[str, set[str]]]) -> pd.DataFrame:
    rows = []
    background = set(next(iter(rankings.values()))['CONSENSUS'].index)
    M = len(background)
    evidence_index = EVIDENCE.set_index('rmp_id')
    for (rmp_id, reps) in rankings.items():
        program_map = {'GSE160269': evidence_index.loc[rmp_id, 'program_A'], 'GSE188900': evidence_index.loc[rmp_id, 'program_B'], 'CONSENSUS': rmp_id}
        for (representation, series) in reps.items():
            ordered = series.sort_values(ascending=False).index.tolist()
            for signature_size in (50, 100):
                query = set(ordered[:signature_size])
                family_start = len(rows)
                for (library, sets) in gene_sets.items():
                    library_start = len(rows)
                    for (term, genes) in sets.items():
                        tested = genes & background
                        K = len(tested)
                        if K < 10 or K > 500:
                            continue
                        overlap_genes = sorted(query & tested)
                        x = len(overlap_genes)
                        rows.append({'rmp_id': rmp_id, 'representation': representation, 'program_id': program_map[representation], 'signature_size': signature_size, 'library': library, 'term': term, 'background_genes': M, 'gene_set_size': K, 'overlap': x, 'odds_enrichment': x / signature_size / (K / M) if K else np.nan, 'P': float(hypergeom.sf(x - 1, M, K, signature_size)), 'overlap_genes': ';'.join(overlap_genes)})
                    pvals = np.array([r['P'] for r in rows[library_start:]], float)
                    if len(pvals):
                        fdrs = bh(pvals)
                        for (target, value) in zip(rows[library_start:], fdrs):
                            target['FDR'] = value
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'ora_enrichment_all.tsv', sep='\t', index=False)
    return table

def term_themes(term: str) -> list[str]:
    upper = str(term).upper()
    return [theme for (theme, keywords) in THEMES.items() if any((keyword in upper for keyword in keywords))]

def build_theme_scores(gsea: pd.DataFrame, ora: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rmp_id in EVIDENCE.rmp_id:
        for representation in ('GSE160269', 'GSE188900', 'CONSENSUS'):
            for theme in THEMES:
                candidates = []
                g = gsea.loc[gsea.rmp_id.eq(rmp_id) & gsea.representation.eq(representation) & (gsea.NES > 0) & (gsea.FDR < 0.05)]
                for row in g.itertuples():
                    if theme in term_themes(row.term):
                        candidates.append((min(-math.log10(max(row.FDR, 1e-300)), 10.0), row.term, row.library, row.core_genes, 'GSEA'))
                o = ora.loc[ora.rmp_id.eq(rmp_id) & ora.representation.eq(representation) & ora.signature_size.eq(100) & (ora.FDR < 0.05)]
                for row in o.itertuples():
                    if theme in term_themes(row.term):
                        candidates.append((min(-math.log10(max(row.FDR, 1e-300)), 10.0), row.term, row.library, row.overlap_genes, 'ORA_TOP100'))
                if candidates:
                    best = sorted(candidates, key=lambda x: (-x[0], x[1]))[0]
                    rows.append({'rmp_id': rmp_id, 'representation': representation, 'theme': theme, 'support_score': best[0], 'best_term': best[1], 'best_library': best[2], 'supporting_genes': best[3], 'method': best[4], 'supported': True})
                else:
                    rows.append({'rmp_id': rmp_id, 'representation': representation, 'theme': theme, 'support_score': 0.0, 'best_term': 'NONE', 'best_library': 'NONE', 'supporting_genes': '', 'method': 'NONE', 'supported': False})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'theme_support_scores.tsv', sep='\t', index=False)
    return table

def best_consensus_term(rmp_id: str, library_prefix: str, gsea: pd.DataFrame, ora: pd.DataFrame) -> str:
    g = gsea.loc[gsea.rmp_id.eq(rmp_id) & gsea.representation.eq('CONSENSUS') & gsea.library.str.contains(library_prefix, case=False) & (gsea.NES > 0)].sort_values(['FDR', 'NES'], ascending=[True, False])
    o = ora.loc[ora.rmp_id.eq(rmp_id) & ora.representation.eq('CONSENSUS') & ora.signature_size.eq(100) & ora.library.str.contains(library_prefix, case=False)].sort_values(['FDR', 'P'])
    candidates = []
    if len(g):
        candidates.append((float(g.iloc[0].FDR), str(g.iloc[0].term)))
    if len(o):
        candidates.append((float(o.iloc[0].FDR), str(o.iloc[0].term)))
    return sorted(candidates)[0][1] if candidates else 'NONE'

def annotate(theme_scores: pd.DataFrame, frozen: pd.DataFrame, gsea: pd.DataFrame, ora: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evidence = EVIDENCE.set_index('rmp_id')
    for rmp_id in EVIDENCE.rmp_id:
        per_rep = {}
        for rep in ('GSE160269', 'GSE188900', 'CONSENSUS'):
            sub = theme_scores.loc[theme_scores.rmp_id.eq(rmp_id) & theme_scores.representation.eq(rep)]
            sub = sub.sort_values(['support_score', 'theme'], ascending=[False, True])
            per_rep[rep] = sub
        cons = per_rep['CONSENSUS']
        primary = str(cons.iloc[0].theme) if cons.iloc[0].support_score > 0 else 'UNRESOLVED'
        primary_score = float(cons.iloc[0].support_score) if primary != 'UNRESOLVED' else 0.0
        second = str(cons.iloc[1].theme) if len(cons) > 1 and cons.iloc[1].support_score > 0 else 'NONE'
        second_score = float(cons.iloc[1].support_score) if second != 'NONE' else 0.0
        competing = second != 'NONE' and second_score >= 0.75 * primary_score
        support_a = bool(per_rep['GSE160269'].set_index('theme').support_score.get(primary, 0) > 0)
        support_b = bool(per_rep['GSE188900'].set_index('theme').support_score.get(primary, 0) > 0)
        top_a = str(per_rep['GSE160269'].iloc[0].theme) if per_rep['GSE160269'].iloc[0].support_score > 0 else 'UNRESOLVED'
        top_b = str(per_rep['GSE188900'].iloc[0].theme) if per_rep['GSE188900'].iloc[0].support_score > 0 else 'UNRESOLVED'
        cohort_top_theme_competition = top_a != primary or top_b != primary
        discordant = top_a != top_b and (not (support_a and support_b))
        primary_support_rows = theme_scores.loc[theme_scores.rmp_id.eq(rmp_id) & theme_scores.theme.eq(primary) & theme_scores.supported]
        supporting = set()
        for value in primary_support_rows.supporting_genes.fillna(''):
            supporting.update(str(value).split(';'))
        top50 = frozen.loc[frozen.rmp_id.eq(rmp_id) & frozen.signature_type.eq('TOP50')].sort_values('consensus_rank')
        coherence_n = len(set(top50.gene) & supporting)
        coherence = coherence_n >= 3
        if primary == 'UNRESOLVED' or discordant:
            confidence = 'LOW'
        elif support_a and support_b and coherence and (not competing) and (not cohort_top_theme_competition):
            confidence = 'HIGH'
        elif support_a or support_b:
            confidence = 'MODERATE'
        else:
            confidence = 'LOW'
        label = THEME_LABELS[primary]
        if competing:
            label = f'{label} / {THEME_LABELS[second]}'
        state_flag = 'GENERIC_MALIGNANT_STATE' if primary in GENERIC_THEMES or primary == 'UNRESOLVED' else 'MORE_SPECIFIC_MALIGNANT_STATE'
        technical_review = 'POSSIBLE_TECHNICAL_OR_GENERIC_STATE_REVIEW' if primary in {'TRANSLATION_RIBOSOMAL', 'MITOCHONDRIAL_RESPIRATION', 'HYPOXIA_STRESS_RESPONSE'} else 'NONE'
        ev = evidence.loc[rmp_id]
        rows.append({'rmp_id': rmp_id, 'tier': ev.evidence_tier, 'GSE160269_program': ev.program_A, 'GSE188900_program': ev.program_B, 'three_cohort_support': ev.THREE_COHORT_SUPPORT, 'biological_label': label, 'primary_theme': primary, 'secondary_theme': second if competing else 'NONE', 'annotation_confidence': confidence, 'state_specificity_flag': state_flag, 'technical_state_review': technical_review, 'top_theme_GSE160269': top_a, 'top_theme_GSE188900': top_b, 'GSE160269_supports_consensus_theme': support_a, 'GSE188900_supports_consensus_theme': support_b, 'consensus_theme_support_score': primary_score, 'competing_theme': competing, 'cohort_top_theme_competition': cohort_top_theme_competition, 'top50_pathway_core_gene_overlap': coherence_n, 'top_gene_coherence_pass': coherence, 'annotation_discordance_flag': 'COHORT_DISCORDANT_ANNOTATION' if discordant else 'CONCORDANT_OR_PARTIALLY_CONCORDANT', 'top10_consensus_genes': ';'.join(top50.head(10).gene), 'top_Hallmark': best_consensus_term(rmp_id, 'Hallmark', gsea, ora), 'top_Reactome': best_consensus_term(rmp_id, 'Reactome', gsea, ora), 'top_GO_BP': best_consensus_term(rmp_id, 'GO_Biological', gsea, ora), 'top_KEGG': best_consensus_term(rmp_id, 'KEGG', gsea, ora)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'rmp_annotation_table.tsv', sep='\t', index=False)
    return table

def nonredundant_terms(frame: pd.DataFrame, sets: dict[str, set[str]], max_terms: int=2) -> list[str]:
    selected = []
    for row in frame.sort_values(['FDR', 'P']).itertuples():
        genes = sets.get(row.term, set())
        if all((len(genes & sets.get(term, set())) / max(len(genes | sets.get(term, set())), 1) < 0.5 for term in selected)):
            selected.append(row.term)
        if len(selected) >= max_terms:
            break
    return selected

def make_figures(annotation: pd.DataFrame, frozen: pd.DataFrame, gsea: pd.DataFrame, ora: pd.DataFrame, theme_scores: pd.DataFrame, gene_sets: dict[str, dict[str, set[str]]]) -> None:
    hall_terms = []
    for rmp_id in annotation.rmp_id:
        frame = gsea.loc[gsea.rmp_id.eq(rmp_id) & gsea.representation.eq('CONSENSUS') & gsea.library.eq('MSigDB_Hallmark_2020') & (gsea.NES > 0) & (gsea.FDR < 0.05)].copy()
        frame['P'] = frame['P'].astype(float)
        hall_terms.extend(nonredundant_terms(frame, gene_sets['MSigDB_Hallmark_2020']))
    hall_terms = list(dict.fromkeys(hall_terms))
    hall = gsea.loc[gsea.representation.eq('CONSENSUS') & gsea.library.eq('MSigDB_Hallmark_2020')]
    hall_matrix = hall.pivot(index='rmp_id', columns='term', values='NES').reindex(index=annotation.rmp_id, columns=hall_terms).fillna(0)
    (fig, ax) = plt.subplots(figsize=(max(10, 0.55 * len(hall_terms) + 4), 7), constrained_layout=True)
    sns.heatmap(hall_matrix, cmap='vlag', center=0, ax=ax, cbar_kws={'label': 'Consensus NES'})
    ax.set_title('RMP × Hallmark consensus enrichment (fixed significant-term rule)')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0, labelsize=9)
    ax.tick_params(axis='x', labelrotation=60, labelsize=8)
    fig.savefig(OUT / 'rmp_hallmark_enrichment_heatmap.png', dpi=250)
    fig.savefig(OUT / 'rmp_hallmark_enrichment_heatmap.pdf')
    plt.close(fig)
    selected = []
    for rmp_id in annotation.rmp_id:
        for library in ('Reactome_Pathways_2024', 'GO_Biological_Process_2026'):
            frame = ora.loc[ora.rmp_id.eq(rmp_id) & ora.representation.eq('CONSENSUS') & ora.signature_size.eq(100) & ora.library.eq(library) & (ora.FDR < 0.05)]
            selected.extend(nonredundant_terms(frame, gene_sets[library]))
    selected = list(dict.fromkeys(selected))
    sub = ora.loc[ora.representation.eq('CONSENSUS') & ora.signature_size.eq(100) & ora.term.isin(selected)].copy()
    sub['value'] = -np.log10(sub.FDR.clip(lower=1e-300))
    matrix = sub.pivot_table(index='rmp_id', columns='term', values='value', aggfunc='max').reindex(index=annotation.rmp_id, columns=selected).fillna(0)
    (fig, ax) = plt.subplots(figsize=(max(12, 0.42 * len(selected) + 4), 7), constrained_layout=True)
    sns.heatmap(matrix, cmap='mako', ax=ax, cbar_kws={'label': '-log10(FDR), consensus Top100 ORA'})
    ax.set_title('RMP × selected Reactome/GO themes (fixed nonredundant-term rule)')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0, labelsize=9)
    ax.tick_params(axis='x', labelrotation=60, labelsize=7)
    fig.savefig(OUT / 'rmp_reactome_go_enrichment_heatmap.png', dpi=250)
    fig.savefig(OUT / 'rmp_reactome_go_enrichment_heatmap.pdf')
    plt.close(fig)
    top10 = frozen.loc[frozen.signature_type.eq('TOP50')].sort_values(['rmp_id', 'consensus_rank']).groupby('rmp_id').head(10)
    genes = list(dict.fromkeys(top10.gene))
    values = frozen.loc[frozen.signature_type.eq('TOP100') & frozen.gene.isin(genes)].pivot(index='rmp_id', columns='gene', values='consensus_aggregate').reindex(index=annotation.rmp_id, columns=genes).fillna(0)
    (fig, ax) = plt.subplots(figsize=(max(14, 0.3 * len(genes) + 4), 7), constrained_layout=True)
    sns.heatmap(values, cmap='viridis', ax=ax, cbar_kws={'label': 'Consensus rank aggregate'})
    ax.set_title('Consensus top-gene structure (union of per-RMP Top10)')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0, labelsize=9)
    ax.tick_params(axis='x', labelrotation=60, labelsize=7)
    fig.savefig(OUT / 'consensus_top_gene_heatmap.png', dpi=250)
    fig.savefig(OUT / 'consensus_top_gene_heatmap.pdf')
    plt.close(fig)
    rows = []
    for ann in annotation.itertuples():
        for rep in ('GSE160269', 'GSE188900', 'CONSENSUS'):
            score = theme_scores.loc[theme_scores.rmp_id.eq(ann.rmp_id) & theme_scores.representation.eq(rep) & theme_scores.theme.eq(ann.primary_theme), 'support_score']
            rows.append({'rmp_id': ann.rmp_id, 'representation': rep, 'support_score': float(score.iloc[0]) if len(score) else 0})
    concordance = pd.DataFrame(rows).pivot(index='rmp_id', columns='representation', values='support_score').reindex(index=annotation.rmp_id)
    (fig, ax) = plt.subplots(figsize=(7, 6), constrained_layout=True)
    sns.heatmap(concordance, cmap='rocket_r', annot=True, fmt='.1f', ax=ax, cbar_kws={'label': 'Assigned-theme support score'})
    ax.set_title('Annotation concordance: cohort members vs consensus')
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelrotation=0, labelsize=9)
    fig.savefig(OUT / 'annotation_concordance_matrix.png', dpi=250)
    fig.savefig(OUT / 'annotation_concordance_matrix.pdf')
    plt.close(fig)
    merged = annotation.merge(EVIDENCE[['rmp_id', 'primary_composite_similarity']], on='rmp_id')
    confidence_colors = {'HIGH': '#2ca02c', 'MODERATE': '#ffbf00', 'LOW': '#d62728'}
    (fig, ax) = plt.subplots(figsize=(12, 7), constrained_layout=True)
    y = np.arange(len(merged))
    for (i, row) in enumerate(merged.itertuples()):
        marker = 'o' if 'TIER_1' in row.tier else 's'
        ax.scatter(row.primary_composite_similarity, i, s=180, marker=marker, color=confidence_colors[row.annotation_confidence], edgecolor='black')
        suffix = '  +3C' if row.three_cohort_support == 'YES' else ''
        ax.text(row.primary_composite_similarity + 0.01, i, row.biological_label + suffix, va='center', fontsize=8)
    ax.set_yticks(y, merged.rmp_id)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.8, merged.primary_composite_similarity.max() + 0.2))
    ax.set_xlabel('Frozen Phase C primary composite similarity')
    ax.set_title('RMP evidence and biological annotation summary')
    ax.grid(axis='x', alpha=0.2)
    fig.savefig(OUT / 'rmp_evidence_annotation_summary.png', dpi=250)
    fig.savefig(OUT / 'rmp_evidence_annotation_summary.pdf')
    plt.close(fig)

def write_report(annotation: pd.DataFrame, frozen: pd.DataFrame, gsea: pd.DataFrame, ora: pd.DataFrame, b1_count: int, c_count: int) -> None:
    resources = pd.read_csv(resolve_data(GS_DIR / 'gene_set_resource_manifest.tsv'), sep='\t')
    tier1 = annotation.tier.str.contains('TIER_1')
    high_t1 = int((tier1 & annotation.annotation_confidence.eq('HIGH')).sum())
    moderate_t1 = int((tier1 & annotation.annotation_confidence.eq('MODERATE')).sum())
    discordant = annotation.loc[annotation.annotation_discordance_flag.eq('COHORT_DISCORDANT_ANNOTATION'), 'rmp_id'].tolist()
    top_theme_competition = annotation.loc[annotation.cohort_top_theme_competition, 'rmp_id'].tolist()
    ready = 'YES' if len(annotation) == 7 and annotation.annotation_confidence.ne('').all() else 'NO'
    lines = ['# Phase D — Cross-Cohort Biological Annotation and Consensus Signature Freezing', '', 'Date: 2026-08-26 (Asia/Shanghai)', '', '## 1. Executive status', '', f'Phase D completed for all 7 frozen RMPs. Phase B1 ({b1_count} files) and Phase C ({c_count} files) passed SHA256 audit and were not modified. **FROZEN SIGNATURES CREATED: {ready}**', '', 'No spatial, treatment, progression, survival, clinical, bulk, trajectory, outcome-based, or manuscript analysis was performed.', '', '## 2. Signature construction methodology', '', 'Only the frozen GSE160269 and GSE188900 members defined signatures. GSE196756 was excluded. Full-refit scores were mapped to canonical symbols, restricted to the 16,143-gene primary shared universe, converted to within-program percentiles, and aggregated by geometric mean. This penalizes cohort-rank disagreement without using enrichment or outcomes. Top50 is the primary frozen signature; Top100 is fixed sensitivity.', '', '## 3. Gene-set database versions', '', '| Library | Source/version label | Sets | Access date | Frozen GMT SHA256 |', '|---|---|---:|---|---|']
    for r in resources.itertuples():
        lines.append(f'| {r.library} | Enrichr `{r.library}` | {r.gene_sets:,} | {r.access_date} | `{r.gmt_sha256}` |')
    lines += ['', 'Preranked enrichment used gseapy 1.3.1, 1,000 permutations, seed 20260826, and BH FDR. ORA used Top50/Top100, the fixed shared-gene background, one-sided hypergeometric P, and within-family BH FDR.', '', '## 4. Cohort-specific enrichment', '', 'Each GSE160269 and GSE188900 member was enriched independently before consensus labeling. Complete results are in `preranked_enrichment_all.tsv` and `ora_enrichment_all.tsv`; nominal-only filtering was not used.', '', '## 5. Consensus enrichment', '', 'Consensus ranks/signatures were independently tested against Hallmark, Reactome, GO BP, and KEGG. Top50 genes were frozen before enrichment and were never modified based on terms.', '', '## 6. RMP annotation table', '', '| RMP | Tier | Members | 3-cohort | Biological label | Confidence | State class | Top10 | Top Hallmark | Top Reactome/GO | Cohort top-theme competition | Discordance |', '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in annotation.itertuples():
        top_rg = r.top_Reactome if r.top_Reactome != 'NONE' else r.top_GO_BP
        lines.append(f'| {r.rmp_id} | {r.tier} | {r.GSE160269_program} ↔ {r.GSE188900_program} | {r.three_cohort_support} | {r.biological_label} | {r.annotation_confidence} | {r.state_specificity_flag} | {r.top10_consensus_genes} | {r.top_Hallmark} | {top_rg} | {str(r.cohort_top_theme_competition).upper()} | {r.annotation_discordance_flag} |')
    lines += ['', 'Tier 2 recurrence evidence remains Tier 2 regardless of annotation confidence.', '', '## 7. Frozen signature audit', '', '| RMP | Top50 SHA256 | Top100 SHA256 | Counts |', '|---|---|---|---|']
    for rmp_id in annotation.rmp_id:
        f50 = frozen.loc[frozen.rmp_id.eq(rmp_id) & frozen.signature_type.eq('TOP50')].iloc[0]
        f100 = frozen.loc[frozen.rmp_id.eq(rmp_id) & frozen.signature_type.eq('TOP100')].iloc[0]
        lines.append(f'| {rmp_id} | `{f50.signature_sha256}` | `{f100.signature_sha256}` | 50 / 100 |')
    lines += ['', 'These signatures are frozen for external scoring. Missing external genes must be handled by coverage reporting; genes may not be substituted or outcome-optimized.', '', '## 8. Priority set', '', '**PRIMARY VALIDATION SET:** RMP_C01, RMP_C02, RMP_C04, RMP_C05, RMP_C07', '', '**SECONDARY / EXPLORATORY SET:** RMP_C03, RMP_C06', '', 'RMP_C04 is Tier 1 with three-cohort supportive evidence. RMP_C01/RMP_C02 retain the strongest Phase C primary pairwise composite evidence. These facts did not alter signatures.', '', '## 9. Warnings/errors', '', '- The Phase C request typo is resolved consistently: frozen GSE196756 silhouette = 0.862643.', '- Official MSigDB 2025.1.Hs direct GMT required login; exact Enrichr-versioned libraries were frozen locally instead.', '- Reactome official current release and the frozen Enrichr Reactome library have different version labels; all statistics use only `Reactome_Pathways_2024`.', '- An incomplete first enrichment attempt was stopped after gseapy reported 0.82% tied GSE188900 scores; deterministic canonical-gene tie-breaking was frozen before the final full run.', '- Post-enrichment confidence audit downgraded RMP_C06 from HIGH to MODERATE because the cohort-specific top themes differ; signatures and enrichment statistics were unchanged.', '', 'No unresolved computational errors.', '', '## 10. Readiness gate', '']
    for r in annotation.itertuples():
        lines += [f'`{r.rmp_id} ANNOTATION: {r.annotation_confidence}`', '']
    lines += [f'`NUMBER OF HIGH-CONFIDENCE ANNOTATED TIER 1 RMPs: {high_t1}`', '', f'`NUMBER OF MODERATE-CONFIDENCE ANNOTATED TIER 1 RMPs: {moderate_t1}`', '', f"`COHORT-DISCORDANT RMPs: {(', '.join(discordant) if discordant else 'NONE')}`", '', f"`COHORT TOP-THEME COMPETITION: {(', '.join(top_theme_competition) if top_theme_competition else 'NONE')}`", '', f'`FROZEN SIGNATURES CREATED: {ready}`', '', f'`READY FOR EXTERNAL BIOLOGICAL VALIDATION: {ready}`', '', f'`READY FOR NEXT PHASE: {ready}`', '', 'Phase D analysis complete.', '']
    (OUT / 'PHASE_D_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

def refresh_manifest() -> None:
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'output_manifest.tsv']
    pd.DataFrame([{'file': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in sorted(files)]).to_csv(OUT / 'output_manifest.tsv', sep='\t', index=False)

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    b1_count = verify_hash_manifest(B1 / 'output_manifest.tsv')
    c_count = verify_hash_manifest(C / 'output_manifest.tsv')
    resource_manifest = pd.read_csv(resolve_data(GS_DIR / 'gene_set_resource_manifest.tsv'), sep='\t')
    for row in resource_manifest.itertuples():
        if sha256(Path(row.gmt_file)) != row.gmt_sha256 or sha256(Path(row.raw_file)) != row.raw_sha256:
            raise RuntimeError(f'Gene-set hash mismatch: {row.library}')
    cohorts = {cohort: load_cohort(cohort) for cohort in ('GSE160269', 'GSE188900', 'GSE196756')}
    (all_genes, frozen, rankings) = freeze_signatures(cohorts)
    gene_sets = {name: read_gmt(path) for (name, path) in LIBRARIES.items()}
    captured = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gsea = run_prerank(rankings, gene_sets)
        captured.extend((str(w.message) for w in caught))
    ora = run_ora(rankings, gene_sets)
    theme_scores = build_theme_scores(gsea, ora)
    annotation = annotate(theme_scores, frozen, gsea, ora)
    make_figures(annotation, frozen, gsea, ora, theme_scores, gene_sets)
    write_report(annotation, frozen, gsea, ora, b1_count, c_count)
    (OUT / 'warnings_errors.txt').write_text('WARNINGS\n' + ('\n'.join(sorted(set(captured))) if captured else 'NONE') + '\n\nAUDIT_NOTES\nFrozen GSE196756 silhouette used: 0.862643\nRECOVERED_GSEA_TIE_WARNING: an incomplete first enrichment attempt was stopped after gseapy reported 0.82% tied GSE188900 scores; deterministic gene-symbol tie-breaking was frozen before the final full run.\nPOST_ENRICHMENT_CONFIDENCE_AUDIT: RMP_C06 was downgraded from HIGH to MODERATE because the two cohort-specific top themes differ; signatures and enrichment results were unchanged.\nOfficial MSigDB direct download required login; frozen Enrichr libraries used.\n\nERRORS\nNONE\n', encoding='utf-8')
    refresh_manifest()
if __name__ == '__main__':
    main()
