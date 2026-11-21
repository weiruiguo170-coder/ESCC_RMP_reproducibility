
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom

from common import (Audit, HERE, bh, composition, load_counts, mean_edges,
                    orthogonal_depth, poisson_moment_correlation, rank_expression, regress,
                    rng, save, sha, signatures, standardize)


SEED = 2026111825
WORLDS = 160
GENES = 12
DRAWS = 199
SCENARIOS = ["GAMMA_EXOGENOUS", "LOGNORMAL_EXOGENOUS", "LOGNORMAL_ENDOGENOUS"]
METHODS = ["RAW_RANK", "LINEAR_RANK", "CUBIC_RANK", "SCALED_PEARSON", "POISSON_MOMENT"]
PROGRAMS = ["NULL_1", "NULL_2", "NULL_3", "NULL_HIGH", "WEAK_BROAD", "BROAD", "FOCAL"]
PREVALENCE = np.array([0, 0, 0, 0, 1, .85, .30])
ACTIVE_RHO = np.array([0, 0, 0, 0, .10, .30, .80])
GAMMA_SHAPE = 3
LOG_SD = .65
DECISIONS = ["FROZEN_Q", "MATCH_ONLY", "MEAN_CI_ONLY", "SIGN_ONLY"]
CORE = len(PROGRAMS) * GENES
BG_START = CORE + 6 * GENES
TARGET = np.arange(CORE).reshape(len(PROGRAMS), GENES)


@dataclass
class Calibration:
    means: np.ndarray
    cells: np.ndarray
    depths: list
    median_lib: float


def calibrate(audit):
    x, names, patient = load_counts(audit, "GSE199654")
    members = {g for values in signatures(audit).values() for g in values}
    eligible = np.array([g in members for g in names])
    lib = np.asarray(x[:, ~eligible].sum(axis=1)).ravel().astype(float)
    med = np.median(lib)
    n = len(lib)
    rates = np.asarray(x[:, eligible].multiply((med / lib)[:, None]).sum(axis=0)).ravel() / n
    template = np.quantile(rates, np.linspace(.15, .90, GENES))
    assert (template > 0).all()
    # Ascending counts put the four unevaluable sample sizes first.
    order = sorted(set(patient), key=lambda p: (patient.eq(p).sum(), p))
    pools = [lib[patient.eq(p)] for p in order]
    counts = np.array([len(pool) for pool in pools])
    assert sum(counts < 20) == 4 and len(counts) == 14
    return Calibration(template, counts, pools, med)


def generate(random, cal, mechanism, pid, truth, depth_multiplier=1):
    """Independent patients, gene templates set before looking at any methods."""
    n = cal.cells[pid]
    exposure = random.choice(cal.depths[pid], size=n, replace=True)
    gaussian = mechanism != "GAMMA_EXOGENOUS"
    scales = np.ones(CORE + 12 * GENES)
    scales[TARGET[3]] = 3
    scales[BG_START:] = 3
    if gaussian:
        g = random.standard_normal(size=(n, len(scales)))
        shared = random.standard_normal(size=(n, len(PROGRAMS)))
        shape = np.log1p(truth * np.expm1(LOG_SD**2)) / LOG_SD**2
        g[:, :CORE] = (np.sqrt(1 - shape)[None, :, None] * g[:, :CORE].reshape(n, -1, GENES)
                       + np.sqrt(shape)[None, :, None] * shared[:, :, None]).reshape(n, CORE)
        latent = np.exp(LOG_SD * g - .5 * LOG_SD**2)
        if mechanism == "LOGNORMAL_ENDOGENOUS":
            exposure = exposure * np.exp(.65 * shared[:, 5] - .5 * .65**2)
    else:
        latent = random.gamma(GAMMA_SHAPE, 1 / GAMMA_SHAPE, size=(n, len(scales)))
        for program, rho in enumerate(truth):
            if rho:
                common = random.gamma(GAMMA_SHAPE * rho, 1 / GAMMA_SHAPE, size=(n, 1))
                individual = random.gamma(GAMMA_SHAPE * (1 - rho), 1 / GAMMA_SHAPE, size=(n, GENES))
                latent[:, TARGET[program]] = common + individual
    exposure = exposure * depth_multiplier
    means = np.tile(cal.means, len(scales) // GENES) * scales
    counts = random.poisson(exposure[:, None] / cal.median_lib * means * latent)
    return counts, exposure


def matched_sets(random):
    low = CORE + random.integers(0, 6, (DRAWS, GENES)) * GENES + np.arange(GENES)
    high = BG_START + random.integers(0, 6, (DRAWS, GENES)) * GENES + np.arange(GENES)
    # Other than the elevated-expression null, all sets share the low profile reference.
    return np.stack([high if program == 3 else low for program in range(len(PROGRAMS))])


def analyze_patient(counts, depth, backgrounds):
    rank, _ = rank_expression(counts, depth)
    rank_1 = standardize(regress(rank, orthogonal_depth(depth, 1)))
    rank_3 = standardize(regress(rank, orthogonal_depth(depth, 3)))
    normalized = standardize(counts / depth[:, None])
    simple = [composition(z) for z in [rank, rank_1, rank_3, normalized]]
    moment, _ = poisson_moment_correlation(counts, depth)
    values, nulls, eligibility, target_coverage, bg_coverage = [], [], [], [], []
    for corr in [*simple, moment]:
        estimate, valid_target = mean_edges(corr, TARGET)
        null, valid_null = mean_edges(corr, backgrounds)
        null = np.where(valid_null >= .8, null, np.nan)
        bg_ok = (valid_null >= .8).mean(axis=-1)
        ready = (valid_target >= .8) & (bg_ok >= .8)
        # A partially defined set contributes a mean only when predefined coverage is met.
        bg_mean = np.divide(np.nansum(null, axis=-1), np.isfinite(null).sum(axis=-1),
                            out=np.full(len(null), np.nan), where=np.isfinite(null).any(axis=-1))
        values.append(np.where(ready, estimate - bg_mean, np.nan))
        nulls.append(null - bg_mean[:, None])
        eligibility.append(ready)
        target_coverage.append(valid_target)
        bg_coverage.append(bg_ok)
    return map(np.array, (values, nulls, eligibility, target_coverage, bg_coverage))


def evaluate_cohort(q, null, ground, u, retain, labels):
    mean = np.full(len(PROGRAMS), np.nan)
    positive = mean.copy()
    low, high = mean.copy(), mean.copy()
    matched_p, sign_p = mean.copy(), mean.copy()
    n = np.zeros(len(PROGRAMS), dtype=int)
    matched_draw_n = n.copy()
    truth_in_evaluable = mean.copy()
    for i in range(len(PROGRAMS)):
        use = np.isfinite(q[:, i]) & retain
        vals = q[use, i]
        n[i] = len(vals)
        if not len(vals):
            continue
        mean[i] = vals.mean()
        truth_in_evaluable[i] = ground[use, i].mean()
        positives = (vals > 0).sum()
        positive[i] = positives
        if len(vals) < 5 or len(vals) < .5 * len(q):
            continue
        # The same uniform draws are reused; patient index resolution depends on availability.
        indices = (u[:, :len(vals)] * len(vals)).astype(int)
        bootstrap = vals[indices].mean(axis=1)
        low[i], high[i] = np.quantile(bootstrap, [.025, .975])
        force_complete = null[use, i]
        all_matched = np.all(np.isfinite(force_complete), axis=0)
        matched_draw_n[i] = all_matched.sum()
        if matched_draw_n[i] >= .8 * DRAWS:
            observed_null = force_complete[:, all_matched].mean(axis=0)
            matched_p[i] = (1 + (observed_null >= mean[i] - 1e-12).sum()) / (len(observed_null) + 1)
        sign_p[i] = binom.sf(positives - 1, len(vals), .5)
    q_match, q_sign = bh(matched_p), bh(sign_p)
    eval_mean = np.isfinite(low)
    decides = np.isfinite(q_match) & eval_mean
    possible_outcome = dict(
        FROZEN_Q=(decides, (q_match < .05) & (mean > 0) & (low > 0) & (positive >= .7 * n)),
        MATCH_ONLY=(decides, q_match < .05),
        MEAN_CI_ONLY=(eval_mean, low > 0),
        SIGN_ONLY=(np.isfinite(q_sign), q_sign < .05))
    rows = []
    for i, program in enumerate(PROGRAMS):
        total = len(q)
        analytic = PREVALENCE[i] * ACTIVE_RHO[i]
        truth_rate = (ground[:, i] > 0).mean()
        lo_bound = positive[i] / total
        hi_bound = (positive[i] + total - n[i]) / total
        flags = {f"{rule}_{name}": bool(vector[k][i]) for rule, vector in possible_outcome.items()
                 for k, name in enumerate(["evaluable", "supported"])}
        rows.append(dict(**labels, program=program, n_evaluable=n[i], finite_matched_draws=matched_draw_n[i],
                         estimate=mean[i], CI_low=low[i], CI_high=high[i], match_p=matched_p[i], match_q=q_match[i],
                         observed_positive=positive[i], sign_q=q_sign[i],
                         truth_active_rate=PREVALENCE[i], truth_population_mean=analytic,
                         truth_finite_mean=ground[:, i].mean(), truth_evaluable_mean=truth_in_evaluable[i],
                         truth_finite_active_rate=truth_rate, positive_bound_low=lo_bound,
                         positive_bound_high=hi_bound, bound_covers_latent=np.isfinite(lo_bound)
                         and lo_bound - 1e-10 <= truth_rate <= hi_bound + 1e-10,
                         CI_covers_population=np.isfinite(low[i]) and low[i] - 1e-10 <= analytic <= high[i] + 1e-10,
                         **flags))
    return rows


def simulate(calibration, worlds, seed, depth_multiplier):
    truth_rows, patient_rows, cohort_rows = [], [], []
    for j, scenario in enumerate(SCENARIOS):
        for i in range(worlds):
            label = dict(mechanism=scenario, world=i)
            world = rng(seed, j, i, 0)
            active = world.random((len(calibration.cells), len(PROGRAMS))) < PREVALENCE[None, :]
            truth = active * ACTIVE_RHO[None, :]
            shape = (len(calibration.cells), len(METHODS), len(PROGRAMS))
            qobs = np.full(shape, np.nan)
            all_null = np.full(shape + (DRAWS,), np.nan)
            truth_rows.extend(dict(**label, patient_slot=k, program=p, cells=int(calibration.cells[k]),
                                   latent_rho=truth[k, v]) for k in range(len(calibration.cells))
                              for v, p in enumerate(PROGRAMS))
            for k, cell_count in enumerate(calibration.cells):
                if cell_count < 20:
                    continue
                random = rng(seed, j, i, k + 1)
                counts, depth = generate(random, calibration, scenario, k, truth[k], depth_multiplier)
                assert len(depth) == cell_count and (depth > 0).all() and (counts >= 0).all()
                q, null, eligible, frac_edge, frac_bg = analyze_patient(counts, depth, matched_sets(random))
                qobs[k], all_null[k] = q, null
                for m, method in enumerate(METHODS):
                    patient_rows.extend(dict(**label, patient_slot=k, program=p, method=method,
                                             q_effect=q[m, v], coverage_edges=frac_edge[m, v],
                                             coverage_background_draws=frac_bg[m, v], eligible=eligible[m, v],
                                             cells=int(cell_count), library_median=np.median(depth))
                                        for v, p in enumerate(PROGRAMS))
            complete = calibration.cells >= 20
            assessed = np.flatnonzero(complete)
            priority = np.lexsort((world.random(len(assessed)), -truth[assessed, 5]))
            selected = complete.copy()
            selected[assessed[priority[:3]]] = False
            index = world.random((399, len(calibration.cells)))
            for m, method in enumerate(METHODS):
                for missingness, retain in [("STANDARD_SCOPE", complete), ("ACTIVE_FIRST_REMOVAL", selected)]:
                    cohort_rows.extend(evaluate_cohort(qobs[:, m], all_null[:, m], truth, index, retain,
                                                       dict(**label, method=method, missingness=missingness)))
            if (i + 1) % 40 == 0 or i + 1 == worlds:
                print(f"{scenario} / {i + 1} completed", flush=True)
    return pd.DataFrame(truth_rows), pd.DataFrame(patient_rows), pd.DataFrame(cohort_rows)


def summarize(cohort):
    rows, rules = [], []
    for keys, group in cohort.groupby(["mechanism", "program", "method", "missingness"]):
        mech, program, method, missing = keys
        label = dict(mechanism=mech, program=program, method=method, missingness=missing)
        n = len(group)
        finite = group[np.isfinite(group.CI_low)]
        paired = group[np.isfinite(group.estimate)]
        diff = paired.estimate - paired.truth_evaluable_mean
        informed = group.positive_bound_low.notna()
        rows.append(dict(**label, worlds=n, n_with_interval=len(finite),
                         mean_n_evaluable=group.n_evaluable.mean(),
                         mean_estimate=paired.estimate.mean(),
                         bias_vs_latent_Pearson=diff.mean(), RMSE_vs_latent_Pearson=np.sqrt((diff**2).mean()),
                         coverage_population_given_interval=finite.CI_covers_population.mean(),
                         coverage_population_all_worlds=group.CI_covers_population.mean(),
                         coverage_MCSE=np.sqrt(finite.CI_covers_population.mean() *
                                               (1 - finite.CI_covers_population.mean()) / max(1, len(finite))),
                         full_cohort_latent_positive_bound_coverage=group.loc[informed, "bound_covers_latent"].mean(),
                         positive_bound_evaluable=informed.mean(),
                         mean_observed_positive_fraction=(paired.observed_positive / paired.n_evaluable).mean(),
                         mean_latent_active_rate=paired.truth_finite_active_rate.mean(),
                         mean_observed_CI_width=(finite.CI_high - finite.CI_low).mean(),
                         scale="LATENT_PEARSON" if "PEARSON" in method or "MOMENT" in method else "DIFFERENT_RANK_SCALE"))
        for rule in DECISIONS:
            direct = group[f"{rule}_supported"]
            qualified = group[f"{rule}_evaluable"]
            rate = direct.mean()
            rules.append(dict(**label, rule=rule, worlds=n, positives=int(direct.sum()),
                              eligible=int(qualified.sum()), rate=rate, eligibility=qualified.mean(),
                              conditional_rate=direct[qualified].mean(),
                              mcse=np.sqrt(rate * (1 - rate) / n)))
    return pd.DataFrame(rows), pd.DataFrame(rules)


def main():
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--pilot", action="store_true")
    modes.add_argument("--high-depth", type=int, choices=[4, 16])
    args = p.parse_args()
    suffix = f"_depth{args.high_depth}x" if args.high_depth else ""
    audit = Audit("controlled_pilot" if args.pilot else "controlled" + suffix)
    if suffix:
        amendment = sha(HERE / "AMENDMENT_HIGH_DEPTH.md")
        assert amendment == (HERE / "AMENDMENT_HIGH_DEPTH.sha256").read_text().strip()
    calibration = calibrate(audit)
    seed = SEED + 1 if args.pilot else SEED
    worlds = 2 if args.pilot else 80 if args.high_depth else WORLDS
    truth, patient, cohort = simulate(calibration, worlds, seed, args.high_depth or 1)
    assert len(truth) == 3 * worlds * 14 * 7
    assert len(cohort) == 3 * worlds * 7 * 5 * 2
    performance, decisions = summarize(cohort)
    prep = dict(template_expected_count_at_reference=calibration.means.tolist(),
                patient_cell_counts=calibration.cells.tolist(), median_non_signature_UMI=float(calibration.median_lib),
                parent_cohort="GSE199654", negative_control_origin="SIMULATED_LATENT_RHO_ZERO")
    if args.pilot:
        save(pd.DataFrame([dict(program=r, truth_rows=len(g), mean_rho=g.latent_rho.mean())
                           for r, g in truth.groupby("program")]), "controlled_pilot_truth.tsv")
    else:
        save(truth, f"controlled_truth{suffix}.tsv")
        save(patient, f"controlled_patient_measures{suffix}.tsv")
        save(cohort, f"controlled_world_results{suffix}.tsv")
        save(performance, f"controlled_estimation_summary{suffix}.tsv")
        save(decisions, f"controlled_decision_summary{suffix}.tsv")
    audit.finish(worlds_per_scenario=worlds, total_worlds=3 * worlds, seed=seed,
                 patient_rows=len(patient), cohort_rows=len(cohort), calibration=prep,
                 depth_multiplier=args.high_depth or 1,
                 amendment=amendment if suffix else None)


if __name__ == "__main__":
    main()
