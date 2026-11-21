"""Frozen, patient-level RNA/protein analyses; technical failures remain visible."""
from pathlib import Path as _Path
import sys as _sys
_root = next((p for p in _Path(__file__).resolve().parents if (p / 'scripts/_paths.py').is_file()))
_sys.path.insert(0, str(_root / 'scripts'))
from _paths import ROOT, REPO_ROOT, INPUT_ROOT, OUTPUT_ROOT, data_path, resolve_data, read_table
from pathlib import Path
import json, hashlib, platform, importlib.metadata, datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = INPUT_ROOT / 'RMP_UPGRADE_PROJECT'
OUT = OUTPUT_ROOT/'second_proteome'
OUT.mkdir(exist_ok=True, parents=True)
DATA = INPUT_ROOT/'second_proteome'
SEED = 20260930
B = 5000
P = 10000
M = 1000
sig=[]
for i in range(1,8):
    for n in [50,100]:
        genes=data_path(f'RMP_C{i:02d}_signature_top{n}.txt').read_text().splitlines()
        assert len(genes)==len(set(genes))==n
        sig.append(dict(rmp=f'C{i:02d}',size=n,genes=genes))
rna_raw = pd.read_csv(resolve_data(DATA / 'rna.tsv'), sep='\t').drop(columns='Gene ID').set_index('Gene name').groupby(level=0).median()
protein_raw = pd.read_csv(resolve_data(DATA / 'protein.tsv'), sep='\t').drop(columns='Protein accession').set_index('Gene name').groupby(level=0).median()
assert (rna_raw >= 0).all().all() and (not rna_raw.isna().any().any())
assert not (protein_raw <= 0).any().any()
rna = rna_raw.rank(axis=0, method='average') / len(rna_raw)
plog = np.log2(protein_raw)
eligible = plog.notna().mean(axis=1).ge(0.8) & plog.std(axis=1, ddof=1).gt(0)
pz = plog.sub(plog.mean(axis=1), axis=0).div(plog.std(axis=1, ddof=1), axis=0)
pz = pz.loc[eligible]
complete = pz.index[pz.notna().all(axis=1)]
prank = plog.rank(axis=0, method='average').div(plog.notna().sum(axis=0), axis=1)
all_patients = sorted({s[:-1] for s in protein_raw.columns})
tumor_ids = [p for p in all_patients if p + 'T' in rna]
paired_ids = [p for p in tumor_ids if p + 'P' in rna]
normal_ids = [p for p in all_patients if p + 'P' in rna]
assert len(tumor_ids) == 57 and len(paired_ids) == 57 and (len(normal_ids) == 60)

def score(mat, genes, minimum_fraction=0):
    g = sorted(set(genes) & set(mat.index))
    if not g:
        return pd.Series(np.nan, index=mat.columns)
    sub = mat.loc[g]
    v = sub.mean(axis=0)
    return v.where(sub.notna().sum(axis=0) >= np.ceil(len(g) * minimum_fraction))

def rho(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    if v.sum() < 3:
        return np.nan
    a = rankdata(x[v])
    b = rankdata(y[v])
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt(a @ a * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan

def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    v = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(v[::-1])[::-1].clip(0, 1)
    out = np.empty(len(p))
    out[o] = q
    return out

def inference(x, y, ids, seed):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    x = x[v]
    y = y[v]
    ids = np.asarray(ids)[v]
    n = len(x)
    obs = rho(x, y)
    if n < 3:
        return ({'n': n, 'rho': np.nan}, [])
    rng = np.random.default_rng(seed)
    ind = rng.integers(0, n, (B, n))
    a = rankdata(x[ind], axis=1)
    b = rankdata(y[ind], axis=1)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        boots = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    ci = np.nanquantile(boots, [0.025, 0.975])
    ar = rankdata(x)
    br = rankdata(y)
    ar -= ar.mean()
    br -= br.mean()
    d = np.sqrt(ar @ ar * (br @ br))
    perm = np.stack([rng.permutation(br) for _ in range(P)]) @ ar / d
    pp = (1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (P + 1)
    loo = [rho(np.delete(x, i), np.delete(y, i)) for i in range(n)]
    row = {'n': n, 'rho': obs, 'ci_low': ci[0], 'ci_high': ci[1], 'permutation_p': pp, 'lopo_min': min(loo), 'lopo_max': max(loo), 'lopo_same_sign_fraction': np.mean(np.sign(loo) == np.sign(obs)), 'bootstrap_valid': int(np.isfinite(boots).sum())}
    return (row, [{'excluded_patient': p, 'rho': q} for (p, q) in zip(ids, loo)])
excluded = set((g for s in sig for g in s['genes']))
rg = set(rna.index)
pg = set(pz.index)
universe = rg | pg
ra = rna.to_numpy()
pa = pz.to_numpy()
rindex = {g: i for (i, g) in enumerate(rna.index)}
pindex = {g: i for (i, g) in enumerate(pz.index)}
rcindex = {c: i for (i, c) in enumerate(rna.columns)}
pcindex = {c: i for (i, c) in enumerate(pz.columns)}
feat = pd.DataFrame(index=sorted(universe))
feat['rna_abundance'] = np.log1p(rna_raw.mean(axis=1))
feat['protein_abundance'] = plog.mean(axis=1)
feat['protein_missing'] = plog.isna().mean(axis=1)
for c in list(feat):
    vals = feat[c].dropna()
    edges = np.unique(vals.quantile([1 / 3, 2 / 3]).to_numpy())
    feat[c + '_bin'] = np.searchsorted(edges, feat[c].fillna(-np.inf), side='left')

def stratum(g):
    a = feat.loc[g]
    if g in rg and g in pg:
        return ('both', int(a.rna_abundance_bin), int(a.protein_abundance_bin), int(a.protein_missing_bin))
    if g in rg:
        return ('rna_only', int(a.rna_abundance_bin))
    return ('protein_only', int(a.protein_abundance_bin), int(a.protein_missing_bin))
pools = {}
for g in sorted(universe - excluded):
    pools.setdefault(stratum(g), []).append(g)
null_design = []
null_draws = []

def matched_null(genes, x_obs, y_obs, ids, contrast, seed, rmp, size):
    target = {}
    for g in sorted(set(genes) & universe):
        target.setdefault(stratum(g), []).append(g)
    for (k, v) in target.items():
        null_design.append({'rmp': rmp, 'size': size, 'contrast': contrast, 'stratum': str(k), 'target_n': len(v), 'pool_n': len(pools.get(k, []))})
    if any((len(pools.get(k, [])) < len(v) for (k, v) in target.items())):
        return {'random_status': 'STRATUM_INSUFFICIENT', 'random_p': np.nan, 'random_valid': 0}
    rng = np.random.default_rng(seed)
    obs = rho(x_obs, y_obs)
    null = []
    ns = []
    rt = [rcindex[p + 'T'] for p in ids]
    pt = [pcindex[p + 'T'] for p in ids]
    rn = [rcindex[p + 'P'] for p in ids]
    pn = [pcindex[p + 'P'] for p in ids]
    ov = np.isfinite(x_obs) & np.isfinite(y_obs)

    def array_scores(gs, rcols, pcols):
        ri = [rindex[g] for g in sorted(gs) if g in rindex]
        pi = [pindex[g] for g in sorted(gs) if g in pindex]
        xs = ra[np.ix_(ri, rcols)].mean(axis=0) if ri else np.full(len(rcols), np.nan)
        sub = pa[np.ix_(pi, pcols)]
        ys = np.nanmean(sub, axis=0) if pi else np.full(len(pcols), np.nan)
        ys = np.where(np.isfinite(sub).sum(axis=0) >= np.ceil(len(pi) * 0.8), ys, np.nan)
        return (xs, ys)
    for j in range(M):
        gs = [g for (k, v) in target.items() for g in rng.choice(pools[k], size=len(v), replace=False)]
        (x, y) = array_scores(gs, rt, pt)
        if contrast == 'delta':
            (xn, yn) = array_scores(gs, rn, pn)
            x = x - xn
            y = y - yn
        x = x[ov]
        y = y[ov]
        n = int((np.isfinite(x) & np.isfinite(y)).sum())
        val = rho(x, y) if n >= 20 else np.nan
        null.append(val)
        ns.append(n)
        null_draws.append({'rmp': rmp, 'size': size, 'contrast': contrast, 'draw': j, 'n': n, 'rho': val})
    vals = np.asarray(null)
    valid = vals[np.isfinite(vals)]
    return {'random_status': 'OK' if len(valid) >= 950 else 'INSUFFICIENT_VALID_DRAWS', 'random_p': (1 + np.sum(np.abs(valid) >= abs(obs) - 1e-12)) / (1 + len(valid)) if len(valid) >= 950 else np.nan, 'random_valid': len(valid), 'random_rho_median': np.median(valid), 'random_rho_low': np.quantile(valid, 0.025), 'random_rho_high': np.quantile(valid, 0.975), 'random_n_min': min(ns), 'random_n_max': max(ns)}
coverage = []
rows = []
lopo = []
scores = []
sens = []
for (si, s) in enumerate(sig):
    name = s['rmp']
    size = s['size']
    genes = s['genes']
    mapped = sorted(set(genes) & set(plog.index))
    fixed = sorted(set(genes) & pg)
    cov = {'rmp': name, 'size': size, 'rna_mapped': len(set(genes) & rg), 'protein_mapped': len(mapped), 'protein_fixed_80pct': len(fixed), 'protein_complete_case': len(set(genes) & set(complete)), 'protein_fixed_fraction': len(fixed) / size, 'program_coverage_gate': len(fixed) >= 0.6 * size, 'mapped_protein_missing_fraction': float(plog.loc[mapped].isna().mean().mean()) if mapped else np.nan}
    coverage.append(cov)
    rs = score(rna, genes)
    ps = score(pz, genes, 0.8)
    for p in all_patients:
        for t in 'TP':
            scores.append({'rmp': name, 'size': size, 'patient_id': p, 'tissue': t, 'rna_score': rs.get(p + t, np.nan), 'protein_score': ps.get(p + t, np.nan), 'protein_genes_detected': int(pz.loc[fixed, p + t].notna().sum())})
    for (contrast, ids) in [('tumor', tumor_ids), ('delta', paired_ids), ('normal', normal_ids)]:
        x = rs[[p + ('P' if contrast == 'normal' else 'T') for p in ids]].to_numpy()
        y = ps[[p + ('P' if contrast == 'normal' else 'T') for p in ids]].to_numpy()
        if contrast == 'delta':
            x = x - rs[[p + 'P' for p in ids]].to_numpy()
            y = y - ps[[p + 'P' for p in ids]].to_numpy()
        (stats, loo) = inference(x, y, ids, SEED + si * 100 + {'tumor': 1, 'delta': 2, 'normal': 3}[contrast])
        stats.update({'rmp': name, 'size': size, 'contrast': contrast, 'coverage_gate': cov['program_coverage_gate'], 'status': 'ELIGIBLE' if cov['program_coverage_gate'] and stats['n'] >= 20 else 'UNINTERPRETABLE'})
        if contrast != 'normal':
            stats.update(matched_null(genes, x, y, ids, contrast, SEED + si * 100 + {'tumor': 11, 'delta': 12}[contrast], name, size))
        rows.append(stats)
        for z in loo:
            lopo.append({'rmp': name, 'size': size, 'contrast': contrast, **z})
    common = set(genes) & rg & pg
    for (method, rm, pm, g, frac) in [('complete_case', rna, pz, set(genes) & set(complete), 0), ('protein_percentile', rna, prank, genes, 0.8), ('direct_log2', rna, plog, genes, 0.8), ('shared_eligible_genes', rna, pz, common, 0.8)]:
        rscore = score(rm, genes if method != 'shared_eligible_genes' else g)
        pscore = score(pm, g, frac)
        for contrast in ['tumor', 'delta']:
            ids = tumor_ids if contrast == 'tumor' else paired_ids
            x = rscore[[p + 'T' for p in ids]].to_numpy()
            y = pscore[[p + 'T' for p in ids]].to_numpy()
            if contrast == 'delta':
                x = x - rscore[[p + 'P' for p in ids]].to_numpy()
                y = y - pscore[[p + 'P' for p in ids]].to_numpy()
            sens.append({'rmp': name, 'size': size, 'contrast': contrast, 'method': method, 'protein_gene_n': len(set(g) & set(pm.index)), 'n': int((np.isfinite(x) & np.isfinite(y)).sum()), 'rho': rho(x, y), 'interpretation': 'SENSITIVITY_ONLY; cannot rescue coverage gate'})
    print(name, size, 'fixed_proteins', len(fixed), 'of', size, flush=True)
res = pd.DataFrame(rows)
res['permutation_fdr_7'] = np.nan
res['random_fdr_7'] = np.nan
for size in [50, 100]:
    for contrast in ['tumor', 'delta']:
        ix = (res['size'] == size) & (res.contrast == contrast)
        eligible_row = res.loc[ix, 'status'].eq('ELIGIBLE').to_numpy()
        for (p, q) in [('permutation_p', 'permutation_fdr_7'), ('random_p', 'random_fdr_7')]:
            vals = res.loc[ix, p].to_numpy()
            use = np.where(eligible_row & np.isfinite(vals), vals, 1.0)
            qs = bh(use)
            qs[~eligible_row] = np.nan
            res.loc[ix, q] = qs
res.to_csv(OUT / 'concordance.tsv', sep='\t', index=False)
pd.DataFrame(coverage).to_csv(OUT / 'coverage.tsv', sep='\t', index=False)
pd.DataFrame(scores).to_csv(OUT / 'patient_scores.tsv', sep='\t', index=False)
pd.DataFrame(lopo).to_csv(OUT / 'lopo.tsv', sep='\t', index=False)
pd.DataFrame(sens).to_csv(OUT / 'sensitivity.tsv', sep='\t', index=False)
pd.DataFrame(null_design).drop_duplicates().to_csv(OUT / 'matched_null_design.tsv', sep='\t', index=False)
pd.DataFrame(null_draws).to_csv(OUT / 'matched_null_draws.tsv', sep='\t', index=False)
feat.to_csv(OUT / 'matched_null_gene_features.tsv', sep='\t')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})

def save(fig, name):
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / (name + '.' + ext), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
for (contrast, fn, title) in [('tumor', 'Figure_A1', 'Same-patient tumor RNA–protein concordance'), ('delta', 'Figure_A2', 'Same-patient tumor-minus-adjacent concordance')]:
    d = res[(res['size'] == 50) & (res.contrast == contrast)].copy()
    (fig, ax) = plt.subplots(figsize=(7.6, 4.8))
    for (i, (_, z)) in enumerate(d.iterrows()):
        color = '#235a77' if z.status == 'ELIGIBLE' else '#888888'
        ax.plot([z.ci_low, z.ci_high], [i, i], color=color, lw=1.8)
        ax.scatter(z.rho, i, s=36, c=color if z.status == 'ELIGIBLE' else 'white', edgecolors=color, zorder=3)
        ax.text(1.03, i, f"n={z.n}; {int(next((c['protein_fixed_80pct'] for c in coverage if c['rmp'] == z.rmp and c['size'] == 50)))}/50 proteins", va='center', fontsize=8)
    ax.axvline(0, color='#bbbbbb', lw=0.8)
    ax.set_xlim(-1.02, 1.7)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_yticks(range(len(d)), d.rmp)
    ax.invert_yaxis()
    ax.set_xlabel('Spearman rho and 95% patient bootstrap CI')
    ax.set_title(title + '\nTop50 primary; open symbols: protein coverage below prespecified gate', loc='left', pad=14)
    save(fig, fn)
sd = pd.DataFrame(sens)
(fig, axs) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for (ax, contrast) in zip(axs, ['tumor', 'delta']):
    dd = res[(res['size'] == 50) & (res.contrast == contrast)].set_index('rmp')
    vv = res[(res['size'] == 100) & (res.contrast == contrast)].set_index('rmp')
    ax.scatter(dd.rho, np.arange(7), label='Top50', color='#235a77')
    ax.scatter(vv.rho, np.arange(7) + 0.16, label='Top100', marker='s', color='#ac7b3d')
    ax.axvline(0, color='#bbbbbb', lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title(contrast)
    ax.set_xlabel('Spearman rho')
    ax.set_yticks(range(7), dd.index)
    ax.legend(frameon=False)
axs[0].invert_yaxis()
fig.suptitle('Sensitivity: frozen Top100 does not replace Top50')
fig.tight_layout()
save(fig, 'Figure_SA1_Top100')
(fig, axs) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for (ax, contrast) in zip(axs, ['tumor', 'delta']):
    dd = sd[(sd['size'] == 50) & (sd.contrast == contrast)]
    for (j, (method, sub)) in enumerate(dd.groupby('method')):
        ax.scatter(sub.rho, np.arange(7) + j * 0.1, label=method.replace('_', ' '), s=20)
    ax.set_yticks(np.arange(7) + 0.15, [f'C{x:02d}' for x in range(1, 8)])
    ax.axvline(0, color='#bbb', lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_title(contrast)
    ax.set_xlabel('Spearman rho')
axs[0].invert_yaxis()
axs[1].legend(frameon=False, fontsize=7, loc='lower right')
fig.suptitle('Scoring sensitivities; all frozen memberships unchanged')
fig.tight_layout()
save(fig, 'Figure_SA2_scoring')
print(res[(res['size'] == 50) & (res.contrast != 'normal')].to_string(index=False), flush=True)
