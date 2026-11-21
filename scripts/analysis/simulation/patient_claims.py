
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from core import ALPHA, Audit, R1, R2, S8, V2, code, enough_resolution, holm, pc_test, supported_effect, tail_p, write


KEY = ["cohort", "signature_key", "method", "background"]
FAMILY = ["cohort", "size", "method", "background"]
BASE = ["RAW_RANK", "HIST"]


def effect_state(r):
    return code(supported_effect(r.evaluable, r.full_n, r.effect, r.CI_low, r.q))


def patients_and_claims(audit):
    qpath = R2 / "results"
    patient = audit.read(qpath / "real_q_each_patient.tsv")
    draws = audit.read(qpath / "real_q_background_draws.tsv")
    summary = audit.read(qpath / "real_q_cohort_specifications.tsv")
    pat_key = KEY + ["patient_id"]
    packed = {key: g.statistic.to_numpy() for key, g in draws.groupby(pat_key)}
    patient["patient_p"] = [tail_p(row.target_corr, packed[tuple(getattr(row, k) for k in pat_key)])
                            for row in patient.itertuples(index=False)]
    assert patient.patient_p.notna().all()
    write(patient, "real_patient_measurements.tsv")
    indexed = patient.set_index(KEY).sort_index()
    stat = []
    for r in summary.itertuples():
        key = tuple(getattr(r, col) for col in KEY)
        matches = indexed.loc[key]
        # MultiIndex lookup retains all patients for a single measured condition.
        p = matches.patient_p.to_numpy()
        N = int(r.full_n)
        stats = dict(**{key: getattr(r, key) for key in KEY}, RMP=r.rmp, size=r.size,
                     n_evaluable=r.evaluable, n_original=N, Q_effect=r.effect, B=int(matches.draws.max()),
                     CI_low=r.CI_low, CI_high=r.CI_high, matched_q=r.q, observed_positive=r.positive,
                     n_positive_estimated=int((matches.relative_q > 0).sum()),
                     n_nominal_patient_p_le05=int((p <= .05).sum()), old_Q_gate=r.q_gate,
                     n_majority_required=N // 2 + 1, effect_state=effect_state(r),
                     p_repeated=pc_test(p, N, 2), p_majority=pc_test(p, N, N // 2 + 1),
                     p_fisher_repeated=pc_test(p, N, 2, "fisher"),
                     p_fisher_majority=pc_test(p, N, N // 2 + 1, "fisher"))
        stat.append(stats)
    records = pd.DataFrame(stat)
    for _, group in records.groupby(FAMILY):
        for intent in ["repeated", "majority", "fisher_repeated", "fisher_majority"]:
            records.loc[group.index, f"holm_{intent}"] = holm(group[f"p_{intent}"].to_numpy())
    for intent in ["repeated", "majority"]:
        eligibility = [("INSUFFICIENT_CASES" if not np.isfinite(getattr(r, f"holm_{intent}"))
                        else "DRAW_RESOLUTION" if not enough_resolution(r.n_original,
                              2 if intent == "repeated" else r.n_majority_required, r.B, 7)
                        else "EVALUABLE") for r in records.itertuples()]
        records[f"status_{intent}_reason"] = eligibility
        records[f"state_{intent}"] = [code(p <= ALPHA if e == "EVALUABLE" else None)
                                       for p, e in zip(records[f"holm_{intent}"], eligibility)]
    write(records, "real_Q_claims_by_spec.tsv")
    return records, patient, packed


def envelopes(records):
    report = []
    for (cohort, key), g in records.groupby(["cohort", "signature_key"]):
        x = g.set_index(["method", "background"])
        ref = x.loc[tuple(BASE)]
        raw = x.xs("RAW_RANK")
        legacy_ref = x.xs("HIST", level="background")
        record = dict(cohort=cohort, signature_key=key, RMP=ref.RMP, size=ref["size"],
                      n_evaluable=ref.n_evaluable, n_original=ref.n_original,
                      Q_effect=ref.Q_effect, CI_low=ref.CI_low, CI_high=ref.CI_high,
                      mean_q=ref.matched_q, fraction_positive=ref.observed_positive,
                      n_positive_estimated=ref.n_positive_estimated,
                      n_nominal_patient_p_le05=ref.n_nominal_patient_p_le05,
                      q_effect_min=g.Q_effect.min(), q_effect_max=g.Q_effect.max(),
                      cohort_raw_mean_detect=x.loc["RAW_RANK", "NN_MEAN_DETECT"].Q_effect,
                      cohort_raw_slope_match=x.loc["RAW_RANK", "NN_ADD_DEPTH_SLOPE"].Q_effect,
                      cohort_linear_hist=x.loc["LINEAR_RANK", "HIST"].Q_effect,
                      raw_background_changes=raw.effect_state.nunique() > 1,
                      history_adjustment_changes=legacy_ref.effect_state.nunique() > 1,
                      ref_holm_repeated=ref.holm_repeated, ref_holm_majority=ref.holm_majority)
        for state in ["effect_state", "state_repeated", "state_majority"]:
            record["reference_" + state] = ref[state]
            record["supported_specs_" + state] = int(g[state].eq("SUPPORTED_WITHIN_SCOPE").sum())
            record["assessed_specs_" + state] = int(g[state].ne("NOT_EVALUABLE").sum())
            seen = set(g[state]) - {"NOT_EVALUABLE"}
            record["method_" + state] = ("UNAVAILABLE" if not seen else
                                          "METHOD_SENSITIVE" if len(seen) > 1 else
                                          "CONSISTENT_ACROSS_ASSESSED")
        report.append(record)
    wide = pd.DataFrame(report)
    write(wide, "real_Q_envelopes.tsv")
    return wide


def compare_structure(audit, records):
    bridge = audit.read(R2 / "results/real_frozen_EDSC_compatibility.tsv")
    subset = records[records.cohort.eq("GSE199654")].merge(
        bridge, on=["signature_key", "method", "background"], validate="one_to_one")
    ref = subset[subset.method.eq(BASE[0]) & subset.background.eq(BASE[1])][
        ["signature_key", "effect_state", "state_repeated", "state_majority", "Q_effect",
         "CI_low", "CI_high", "observed_Q"]]
    subset = subset.merge(ref.add_suffix("_ref").rename(columns={"signature_key_ref": "signature_key"}),
                          on="signature_key", validate="many_to_one")
    subset["group_switch"] = subset.frozen_orig_group.ne(subset.observed_group)
    subset["old_domain_Q_switch"] = subset.observed_Q_ref.ne(subset.observed_Q)
    for label in ["effect_state", "state_repeated", "state_majority"]:
        subset[label + "_switch"] = subset[label].ne(subset[label + "_ref"])
        subset[label + "_comparable_switch"] = (subset[label + "_switch"] &
            subset[label].ne("NOT_EVALUABLE") & subset[label + "_ref"].ne("NOT_EVALUABLE"))
    subset["continuous_Q_delta"] = subset.Q_effect - subset.Q_effect_ref
    subset["numeric_changes_tolerance_1e8"] = subset.continuous_Q_delta.abs() > 1e-8
    subset["sign_switch"] = np.sign(subset.Q_effect) != np.sign(subset.Q_effect_ref)
    subset["legacy_Q_hidden_by_early_R"] = (subset.old_domain_Q_switch &
                                            subset.freeze_R & ~subset.group_switch)
    subset["minimum_patient_changed_without_effect_gate_change"] = (
        (subset.state_repeated_comparable_switch | subset.state_majority_comparable_switch) &
        ~subset.effect_state_switch)
    subset["is_reference"] = subset.method.eq(BASE[0]) & subset.background.eq(BASE[1])
    assert len(subset) == 126 and subset.old_domain_Q_switch.sum() == 32 and subset.group_switch.sum() == 0
    subset = subset.sort_values(["signature_key", "method", "background"])
    write(subset, "same_input_paired_comparison.tsv")
    columns = ["group_switch", "old_domain_Q_switch", "effect_state_switch", "state_repeated_switch",
               "state_majority_switch", "state_majority_comparable_switch", "state_repeated_comparable_switch",
               "sign_switch", "legacy_Q_hidden_by_early_R",
               "numeric_changes_tolerance_1e8", "minimum_patient_changed_without_effect_gate_change"]
    explain = {
        "group_switch": "Frozen descriptive group vector",
        "old_domain_Q_switch": "Historical Q-domain gate; inspect domain-specific outputs",
        "effect_state_switch": "Mean-Q support-state change",
        "state_repeated_switch": "Repeated-patient claim-state change, including evaluability",
        "state_majority_switch": "Patient-majority claim-state change, including evaluability",
        "state_majority_comparable_switch": "Patient-majority support change among jointly evaluable records",
        "state_repeated_comparable_switch": "Repeated-patient support change among jointly evaluable records",
        "sign_switch": "Change in the sign of mean Q",
        "legacy_Q_hidden_by_early_R": "Historical Q-gate changes hidden by the earlier R gate",
        "numeric_changes_tolerance_1e8": "Observable numerical change in continuous baselines",
        "minimum_patient_changed_without_effect_gate_change": "Patient-count claim change without a mean-effect gate change",
    }
    write([dict(metric=c, observed=int(subset[c].sum()), all_cells=len(subset),
                nonreference_cells=int((~subset.is_reference).sum()),
                distinct_program_sizes=int(subset.loc[subset[c], "signature_key"].nunique()),
                interpretation=explain[c]) for c in columns], "structural_change_counts.tsv")
    return subset


def leave_patient(patient, packed):
    cases = patient[patient.cohort.eq("GSE199654")]
    rows = []
    for removed_i, removed in enumerate(sorted(set(cases.patient_id))):
        for (sig, method, bg), g in cases[cases.patient_id.ne(removed)].groupby(
                ["signature_key", "method", "background"]):
            data = np.column_stack([packed[("GSE199654", sig, method, bg, p)] for p in g.patient_id])
            null = (data - data.mean(axis=0)).mean(axis=1)
            e = g.relative_q.to_numpy()
            gen = np.random.default_rng(np.random.SeedSequence([2026111835, removed_i]))
            boot = e[gen.integers(0, len(e), (2000, len(e)))].mean(axis=1)
            lo, hi = np.quantile(boot, [.025, .975])
            rows.append(dict(signature_key=sig, size=int(sig.split("TOP")[1]), method=method, background=bg,
                             removed_patient=removed, N=14, n=len(e), estimate=e.mean(), CI_low=lo, CI_high=hi,
                             matched_p=tail_p(e.mean(), null), p_repeated=pc_test(g.patient_p, 14, 2),
                             p_majority=pc_test(g.patient_p, 14, 8)))
    out = pd.DataFrame(rows)
    for _, g in out.groupby(["size", "method", "background", "removed_patient"]):
        out.loc[g.index, "q"] = multipletests(g.matched_p, method="fdr_bh")[1]
        for claim in ["repeated", "majority"]:
            out.loc[g.index, claim + "_holm"] = holm(g["p_" + claim])
    out["effect_state"] = [code(supported_effect(r.n, r.N, r.estimate, r.CI_low, r.q)) for r in out.itertuples()]
    assert len(out) == 1260
    write(out, "real_Q_leave_one_patient.tsv")


def load_profile(audit):
    r1 = audit.read(R1 / "results/claim_specific_reporting_table.tsv")
    disc = audit.read(R1 / "results/baseline_gate_margins.tsv")
    r1 = r1.merge(disc[disc.cohort.eq("external_1")][["RMP", "size", "D_metrics", "P_N_pairs"]],
                  on=["RMP", "size"], validate="one_to_one")
    pdo = audit.read(S8)
    pdo = pdo[pdo.subset.eq("All10_author_AUC") & pdo.endpoint.eq("author_AUC") &
              pdo.program_or_control.str.startswith("RMP")].copy()
    pdo["size"] = pdo.signature.str[3:].astype(int)
    pdo = pdo.rename(columns={"program_or_control": "RMP", "rho": "PDO_spearman",
                              "FDR_seven_exact": "PDO_q", "P_exact_two_sided": "PDO_p", "n": "PDO_n"})
    keep = ["RMP", "size", "PDO_spearman", "PDO_q", "PDO_p", "PDO_n"]
    r1 = r1.merge(pdo[keep], on=["RMP", "size"], validate="one_to_one")
    alias = audit.read(V2 / "Reproducibility_Update_V2/Replay/A_donor_scenarios.tsv")
    alias = alias.groupby("RMP")[["RAW_max", "EQUAL_max", "P_units"]].agg(["min", "max"])
    alias.columns = ["alias_scenarios_" + "_".join(c) for c in alias.columns]
    r1 = r1.merge(alias.reset_index(), on="RMP", validate="many_to_one")
    qcontext = ["RMP", "size", "Q_STANDARD", "Q_LOOSE", "Q_STRICT", "Q_AFTER_DOUBLET_LIKE_REMOVAL",
                "Q_depth_leave_set_out", "Q_depth_leave_set_out_effect", "Q_whole_cohort_positive_lower",
                "Q_whole_cohort_positive_upper", "Q_GSE203067_qualification", "original_role", "frozen_EDSC"]
    write(r1[qcontext], "prior_Q_conditions_and_historical_roles.tsv")
    return r1


def claims_for(r, q):
    def available(*xs):
        return all(pd.notna(x) for x in xs)
    d = code(None if not available(r.D_match_q, r.D_metrics) else
             r.D_match_q < .05 and r.D_metrics >= 2 and r.D_match_verified)
    if not available(r.GSE160269_RAW, r.GSE160269_EQUAL, r.GSE188900_alias_RAW, r.GSE188900_alias_EQUAL):
        r_state = "NOT_EVALUABLE"
    else:
        r_state = ("CONCENTRATION_FLAG_WITHIN_OBSERVED_UNITS" if
                   max(r.GSE160269_RAW, r.GSE160269_EQUAL, r.GSE188900_alias_RAW, r.GSE188900_alias_EQUAL) >= .8
                   else "NO_FLAG_WITHIN_OBSERVED_UNITS")
    p = code(None if not available(r.P_coverage, r.P_dz_CI_low, r.P_dz_CI_high, r.P_matched_q,
                                  r.P_N_pairs) or r.P_N_pairs <= 0 else
             r.P_coverage >= .60 and r.P_matched_q < .05 and
             (r.P_dz_CI_low > 0 or r.P_dz_CI_high < 0))
    pdo = ("NOT_EVALUABLE" if not available(r.PDO_q, r.PDO_spearman, r.PDO_n) else
           "EXPLORATORY_SUPPORT" if r.PDO_q < .05 else "NOT_ESTABLISHED")
    return dict(DISCOVERY_MATCH=d, DISCOVERY_CONTRIBUTION=r_state, DONOR_REPRESENTATION="NOT_EVALUABLE",
                Q_EFFECT=q.get("reference_effect_state", "NOT_EVALUABLE"),
                Q_REPEATED=q.get("reference_state_repeated", "NOT_EVALUABLE"),
                Q_MAJORITY=q.get("reference_state_majority", "NOT_EVALUABLE"),
                LATENT_BIOLOGICAL_SHARING="NOT_ESTABLISHED", PAIRED_TISSUE_PROTEIN=p,
                PDO_ASSOCIATION=pdo, INDEPENDENT_DRUG_PREDICTION="NOT_EVALUABLE")


def make_profiles(audit, wide):
    baseline = load_profile(audit)
    import importlib.util
    model = audit.path("E1_FINAL_RESOLUTION/inputs/frozen_classifier.py")
    import json
    protocol = json.loads(audit.path("E1_FINAL_RESOLUTION/E1_RULES_CONFIG.json").read_text(encoding="utf-8"))
    records = json.loads(audit.path("E1_FINAL_RESOLUTION/inputs/anonymous_records.json").read_text(encoding="utf-8"))
    id_map = audit.read("E1_FINAL_RESOLUTION/inputs/IDENTITY_KEY.tsv").set_index("anonymous_id").RMP
    package = importlib.util.spec_from_file_location("edsc1_audit", model)
    module = importlib.util.module_from_spec(package)
    package.loader.exec_module(module)
    d = pd.DataFrame([dict(RMP=id_map[r["anonymous_id"]], size=r["size"],
                           D_match_verified=module.discovery(r, protocol["thresholds"]),
                           replay_group=module.classify(r, "B", protocol)["category"]) for r in records])
    baseline = baseline.merge(d, on=["RMP", "size"], validate="one_to_one")
    assert baseline.replay_group.eq(baseline.frozen_EDSC).all()
    ext = wide[wide.cohort.eq("GSE199654")].set_index(["RMP", "size"])
    result = []
    from types import SimpleNamespace
    for row in baseline.itertuples():
        q = ext.loc[row.RMP, row.size].to_dict()
        decision = claims_for(row, q)
        twins = [
            (SimpleNamespace(**(row._asdict() | {"GSE188900_alias_RAW": 1.})), q,
             {"DISCOVERY_CONTRIBUTION"}, "R_OR_FLAG_ONLY"),
            (row, {}, {"Q_EFFECT", "Q_REPEATED", "Q_MAJORITY"}, "Q_MISSING_SCOPED"),
            (SimpleNamespace(**(row._asdict() | {"P_dz_CI_low": np.nan})), q,
             {"PAIRED_TISSUE_PROTEIN"}, "PROTEIN_MISSING_ONLY"),
            (SimpleNamespace(**(row._asdict() | {"PDO_q": .00001})), q,
             {"PDO_ASSOCIATION"}, "EXCELLENT_PDO_NO_PREDICTION_UPGRADE"),
        ]
        for changed_input, alternative_q, allowed, scenario in twins:
            test = claims_for(changed_input, alternative_q)
            diff = {k for k in test if test[k] != decision[k]}
            assert diff <= allowed
            result.append(dict(RMP=row.RMP, size=row.size, scenario=scenario,
                               changed=";".join(sorted(diff)), allowed=";".join(sorted(allowed))))
    write(result, "structural_dependency_checks.tsv")
    statuses = [dict(RMP=r.RMP, size=r.size, **claims_for(r, ext.loc[r.RMP, r.size].to_dict()))
                for r in baseline.itertuples()]
    status = pd.DataFrame(statuses)
    write(status.melt(id_vars=["RMP", "size"], var_name="scientific_claim", value_name="reference_state"),
          "claim_decisions_long.tsv")
    profile = baseline.merge(ext.reset_index(), on=["RMP", "size"], validate="one_to_one",
                             suffixes=("_prior", ""))
    profile = profile.merge(status, on=["RMP", "size"], validate="one_to_one")
    profile = profile.sort_values(["RMP", "size"])
    write(profile, "seven_RMP_profiles.tsv")


def main():
    audit = Audit("real")
    records, patients, packed = patients_and_claims(audit)
    scope = envelopes(records)
    pairs = compare_structure(audit, records)
    leave_patient(patients, packed)
    make_profiles(audit, scope)
    audit.finish(real_spec_cohort_rows=len(records), real_patient_rows=len(patients),
                 real_program_size_pairs=len(pairs), removed_patient_specs=1260)


if __name__ == "__main__":
    main()
