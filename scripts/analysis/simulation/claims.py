
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.proportion import proportion_confint

from core import ALPHA, Audit, HERE, R2, enough_resolution, sha, supported_effect, write


METHODS = ["RAW_RANK", "LINEAR_RANK", "CUBIC_RANK"]
KEY = ["mechanism", "world", "method", "program"]
P_BRANCH = ["bonferroni", "fisher"]
INTENTS = [("REPEATED", 2), ("MAJORITY", 8)]


def states_for_effect(frame):
    return [supported_effect(r.n_evaluable, 14, r.estimate, r.CI_low, r.match_q)
            for r in frame.itertuples()]


def join_p(audit, archived):
    patient = audit.result("simulated_patient_tails.tsv")
    output = []
    for scope, proc in [("STANDARD_SCOPE", patient),
                        ("ACTIVE_FIRST_REMOVAL", patient[~patient.active_case_removed])]:
        stats = None
        for b in [199, 999]:
            fields = proc.pivot(index=KEY, columns="patient_slot", values=f"patient_p{b}")
            if stats is None:
                stats = pd.DataFrame(index=fields.index)
            values = fields.to_numpy()
            n = np.isfinite(values).sum(axis=1)
            stats[f"n_{b}"] = n
            padded = np.pad(np.nan_to_num(values, nan=1.), ((0, 0), (0, 14 - values.shape[1])),
                            constant_values=1.)
            ordered = np.sort(padded, axis=1)
            for name, u in INTENTS:
                enough = n >= max(7, u)
                stats[f"{name}_bonferroni_{b}_p"] = np.where(
                    enough, np.minimum(1., (14 - u + 1) * ordered[:, u - 1]), np.nan)
                stats[f"{name}_fisher_{b}_p"] = np.where(
                    enough, chi2.sf(-2 * np.log(ordered[:, u - 1:]).sum(axis=1), 2 * (14 - u + 1)), np.nan)
        output.append(stats.reset_index().assign(missingness=scope))
    result = archived.merge(pd.concat(output, ignore_index=True), on=KEY + ["missingness"], validate="one_to_one")
    assert len(result) == 20160
    assert (result.n_199 == result.n_evaluable).all()
    sort_cols = ["mechanism", "world", "method", "missingness"]
    result = result.sort_values(sort_cols + ["program"]).reset_index(drop=True)
    assert result.groupby(sort_cols).size().eq(7).all()
    p_cols = [key for key in stats if key.endswith("_p")]
    for col in p_cols:
        nominal = result[col].to_numpy().reshape(-1, 7)
        mechanics = np.nan_to_num(nominal, nan=1.)
        order = np.argsort(mechanics, axis=1)
        sorted_p = np.take_along_axis(mechanics, order, axis=1)
        sorted_holm = np.maximum.accumulate(np.minimum(1., sorted_p * np.arange(7, 0, -1)), axis=1)
        adjusted = np.take_along_axis(sorted_holm, np.argsort(order, axis=1), axis=1)
        result[col.replace("_p", "_holm")] = np.where(np.isfinite(nominal), adjusted, np.nan).ravel()
    result["truth_finite_n_positive"] = (14 * result.truth_finite_active_rate).round().astype(int)
    result["effect_decision"] = states_for_effect(result)
    write(result, "controlled_Q_claims_by_world.tsv")
    return result


def pc_scores(result):
    key = ["mechanism", "method", "missingness", "program"]
    operating = []
    for ids, group in result.groupby(key):
        d = dict(zip(key, ids))
        old = group.FROZEN_Q_supported.where(group.FROZEN_Q_evaluable, None)
        prior = group.observed_positive.ge(.7 * group.n_evaluable).where(group.n_evaluable.ge(7), None)
        targets = dict(POSITIVE_MEAN=(group.truth_finite_mean > 0),
                       REPEATED=group.truth_finite_n_positive.ge(2),
                       MAJORITY=group.truth_finite_n_positive.ge(8))
        for target, actual in targets.items():
            tests = {"EDSC1_Q_GATE": old}  # Evaluated against other targets only to illustrate scope drift.
            if target == "POSITIVE_MEAN":
                tests["Q_EFFECT"] = group.effect_decision
            else:
                u = dict(INTENTS)[target]
                tests["OBSERVED_70_PERCENT"] = prior
                tests["OBSERVED_AT_LEAST_U"] = group.observed_positive.ge(u).where(group.n_evaluable.ge(max(7, u)), None)
                for B in [199, 999]:
                    for mode in P_BRANCH:
                        for adjustment in ["p", "holm"]:
                            selected = group[f"{target}_{mode}_{B}_{adjustment}"]
                            reachable = mode == "fisher" or enough_resolution(
                                14, u, B, 7 if adjustment == "holm" else 1)
                            tests[f"PC_{mode.upper()}_{B}_{adjustment.upper()}"] = (
                                selected.le(ALPHA).where(selected.notna() & reachable, None))
            for rule, called in tests.items():
                avail = called.notna()
                yes = called.eq(True)
                for truth_str, hits in [("TRUE", actual), ("FALSE", ~actual)]:
                    k = int(hits.sum())
                    n_hit = int((yes & hits).sum())
                    lo, hi = proportion_confint(n_hit, k, method="wilson") if k else (np.nan, np.nan)
                    operates = int((avail & hits).sum())
                    operating.append(dict(**d, target=target, rule=rule, truth=truth_str,
                                          world_n=k, determined_n=operates, support_n=n_hit,
                                          support_rate=n_hit / k if k else np.nan, wilson_lo=lo, wilson_hi=hi,
                                          evaluable_rate=operates / k if k else np.nan,
                                          conditional_support_rate=n_hit / operates if operates else np.nan))
    data = pd.DataFrame(operating)
    write(data, "controlled_claim_operating_characteristics.tsv")


def sensitivity_by_depth(audit, primary):
    secondary = audit.read(R2 / "results/controlled_world_results_depth4x.tsv")
    base = primary[primary.missingness.eq("STANDARD_SCOPE") & primary.world.lt(80)]
    base = base.set_index(["mechanism", "world", "program", "method"]).effect_decision.unstack().reset_index()
    base = base[base[METHODS].notna().all(axis=1)].copy()
    base["sensitive_at1x"] = base[METHODS].nunique(axis=1).gt(1)
    high = secondary[secondary.missingness.eq("STANDARD_SCOPE") & secondary.method.eq("RAW_RANK")].copy()
    high["RAW_4X"] = states_for_effect(high)
    pairs = base.merge(high[["mechanism", "world", "program", "RAW_4X"]],
                       on=["mechanism", "world", "program"], validate="one_to_one")
    pairs = pairs[pairs.RAW_4X.notna()].copy()
    pairs["raw_changed_at4x"] = pairs.RAW_4X.ne(pairs.RAW_RANK)
    write(pairs, "controlled_sensitivity_depth_pairs.tsv")
    rows = []
    rng = np.random.default_rng(2026111836)
    for scenario, frame in pairs.groupby("mechanism"):
        case = frame.assign(stable=~frame.sensitive_at1x,
                            s_change=frame.sensitive_at1x & frame.raw_changed_at4x,
                            t_change=~frame.sensitive_at1x & frame.raw_changed_at4x)
        clusters = case.groupby("world")[["sensitive_at1x", "stable", "s_change", "t_change"]].sum().to_numpy()
        aggregate = clusters.sum(axis=0)
        draws = clusters[rng.integers(0, len(clusters), size=(2000, len(clusters)))].sum(axis=1)
        effect = aggregate[2:] / aggregate[:2]
        effects_in_draws = draws[:, 2:] / draws[:, :2]
        d = effects_in_draws[:, 0] - effects_in_draws[:, 1]
        lo, hi = np.quantile(d, [.025, .975])
        rows.append(dict(mechanism=scenario, independent_worlds=len(clusters), program_world_records=len(case),
                         sensitive_n=int(aggregate[0]), stable_n=int(aggregate[1]),
                         raw_switch_given_sensitive=effect[0], raw_switch_given_stable=effect[1],
                         risk_difference=effect[0] - effect[1], bootstrap_CI_low=lo, bootstrap_CI_high=hi))
    write(rows, "controlled_sensitivity_depth_predictiveness.tsv")


def resolution():
    rows = []
    for B in [199, 999, 1000]:
        for target, u in INTENTS:
            floor = (14 - u + 1) / (B + 1)
            rows.append(dict(N=14, u=u, target=target, matched_draws=B, nominal_p_floor=floor,
                             holm_first_floor=7 * floor, attainable_nominal=floor <= ALPHA,
                             attainable_holm=7 * floor <= ALPHA))
    write(rows, "partial_conjunction_resolution.tsv")


def main():
    audit = Audit("controlled")
    amendment = sha(HERE / "AMENDMENT_PC_RESOLUTION.md")
    assert amendment == (HERE / "AMENDMENT_PC_RESOLUTION.sha256").read_text().strip()
    correction = sha(HERE / "AMENDMENT_EVALUABILITY.md")
    assert correction == (HERE / "AMENDMENT_EVALUABILITY.sha256").read_text().strip()
    archived = audit.read(R2 / "results/controlled_world_results.tsv")
    archived = archived[archived.method.isin(METHODS)]
    joined = join_p(audit, archived)
    pc_scores(joined)
    sensitivity_by_depth(audit, joined)
    resolution()
    audit.finish(reused_independent_generated_worlds=480, controlled_cases_with_methods=len(joined),
                 amendment=amendment, correction=correction)


if __name__ == "__main__":
    main()
