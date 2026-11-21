
from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table

import sys

import numpy as np

from core import Audit, ROOT, R2, write


def rebuild():
    audit = Audit("patient_tails")
    for name in ["controlled_q.py", "common.py"]:
        audit.path(R2 / name)
    sys.path.insert(0, str(ROOT / R2))
    import controlled_q as sim
    from common import composition, mean_edges, orthogonal_depth, rank_expression, regress, rng, standardize

    # Register the transitive data inputs without creating files in the historical project.
    for row in audit.read(R2 / "results/inputs_controlled.tsv").itertuples():
        audit.path(row.path)
    compat = type("SourceAdapter", (), {"source": lambda _, name: audit.path(name)})()
    calibration = sim.calibrate(compat)
    archived = audit.read(R2 / "results/controlled_patient_measures.tsv")
    archived = archived[archived.method.isin(sim.METHODS[:3])].set_index(
        ["mechanism", "world", "patient_slot", "method", "program"]).sort_index()
    known = audit.read(R2 / "results/controlled_truth.tsv").set_index(
        ["mechanism", "world", "patient_slot", "program"]).latent_rho.sort_index()
    historical_seed = 2026111825
    extra_seed = 2026111834
    output = []
    for s, scenario in enumerate(sim.SCENARIOS):
        for wid in range(160):
            random = rng(historical_seed, s, wid, 0)
            active = random.random((14, len(sim.PROGRAMS))) < sim.PREVALENCE[None, :]
            truth = active * sim.ACTIVE_RHO[None, :]
            np.testing.assert_array_equal(truth.ravel(), known.loc[scenario, wid].unstack().loc[
                :, sim.PROGRAMS].to_numpy().ravel())
            slots = np.flatnonzero(calibration.cells >= 20)
            priority = np.lexsort((random.random(len(slots)), -truth[slots, 5]))
            removed = set(slots[priority[:3]])
            for pid in slots:
                gen = rng(historical_seed, s, wid, int(pid) + 1)
                counts, depth = sim.generate(gen, calibration, scenario, pid, truth[pid])
                old = sim.matched_sets(gen)
                extra = rng(extra_seed, s, wid, int(pid))
                low = sim.CORE + extra.integers(0, 6, (800, sim.GENES)) * sim.GENES + np.arange(sim.GENES)
                high = sim.BG_START + extra.integers(0, 6, (800, sim.GENES)) * sim.GENES + np.arange(sim.GENES)
                extended = np.concatenate([old, np.stack([high if i == 3 else low
                                           for i in range(len(sim.PROGRAMS))])], axis=1)
                rank, _ = rank_expression(counts, depth)
                z = [rank, standardize(regress(rank, orthogonal_depth(depth, 1))),
                     standardize(regress(rank, orthogonal_depth(depth, 3)))]
                for mid, value in enumerate(z):
                    method = sim.METHODS[mid]
                    matrix = composition(value)
                    obs, obs_cov = mean_edges(matrix, sim.TARGET)
                    bg, bg_cov = mean_edges(matrix, extended)
                    bg = np.where(bg_cov >= .8, bg, np.nan)
                    finite = np.isfinite(bg)
                    old_bg_mean = np.divide(np.nansum(bg[:, :199], axis=1), finite[:, :199].sum(axis=1),
                                            out=np.full(len(obs), np.nan), where=finite[:, :199].any(axis=1))
                    available = (obs_cov >= .8) & (finite[:, :199].mean(axis=1) >= .8)
                    new_available = available & (finite.mean(axis=1) >= .8)
                    ref = archived.loc[scenario, wid, pid, method].loc[sim.PROGRAMS]
                    q = np.where(available, obs - old_bg_mean, np.nan)
                    np.testing.assert_allclose(q, ref.q_effect.to_numpy(), rtol=0, atol=2e-11)
                    np.testing.assert_array_equal(available, ref.eligible.to_numpy())
                    exceed = bg >= (obs[:, None] - 1e-12)
                    p_old = np.where(available, (1 + exceed[:, :199].sum(axis=1)) /
                                     (1 + finite[:, :199].sum(axis=1)), np.nan)
                    p_cur = np.where(new_available, (1 + exceed.sum(axis=1)) / (1 + finite.sum(axis=1)), np.nan)
                    output.extend(dict(mechanism=scenario, world=wid, patient_slot=int(pid),
                                       method=method, program=program, count_cells=int(calibration.cells[pid]),
                                       active_case_removed=pid in removed, latent_rho=truth[pid, i],
                                       relative_q=q[i], patient_p199=p_old[i], patient_p999=p_cur[i],
                                       finite_draws199=int(finite[i, :199].sum()), finite_draws999=int(finite[i].sum()))
                                  for i, program in enumerate(sim.PROGRAMS))
            if (wid + 1) % 40 == 0:
                print(f"{scenario} {wid + 1}/160", flush=True)
    assert len(output) == 100800
    write(output, "simulated_patient_tails.tsv")
    audit.finish(patient_estimates=len(output), worlds=480, prefix_draws=199, extended_draws=999,
                 historical_seed=historical_seed, additional_draws_seed=extra_seed)


if __name__ == "__main__":
    rebuild()
