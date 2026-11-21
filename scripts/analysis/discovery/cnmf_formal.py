#!/usr/bin/env python
"""Run one locked Phase B1 primary-K cNMF analysis and no downstream biology."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from cnmf.cnmf import cNMF, load_df_from_npz
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score


PROJECT_ROOT = INPUT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results"
OUT = RESULTS_DIR / "cnmf_formal"
MANIFEST = RESULTS_DIR / "cnmf_pilot" / "input_manifest.tsv"
SEED = 20260826
N_ITER = 100
NUM_HVG = 2000
WORKERS = 4
LOCAL_NEIGHBORHOOD_SIZE = 0.30
DENSITY_CANDIDATES = (0.5, 1.0, 2.0)
MIN_RETAINED_FRACTION = 0.80
MIN_CLUSTER_SIZE = 5
K_BY_COHORT = {"GSE160269": 10, "GSE188900": 8, "GSE196756": 5}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_for(cohort_dir: Path) -> cNMF:
    return cNMF(output_dir=str(cohort_dir), name="formal")


def audit_input(cohort: str, input_path: Path) -> str:
    manifest = pd.read_csv(resolve_data(MANIFEST), sep="\t").set_index("cohort")
    expected = str(manifest.loc[cohort, "sha256"]).lower()
    observed = sha256(input_path).lower()
    if observed != expected:
        raise RuntimeError(f"HASH_MISMATCH {cohort}: expected={expected}; observed={observed}")
    return observed


def choose_density(model: cNMF, k: int) -> tuple[float, pd.DataFrame, dict[str, float]]:
    merged = load_df_from_npz(model.paths["merged_spectra"] % k)
    l2 = merged.div(np.sqrt((merged**2).sum(axis=1)), axis=0)
    n_neighbors = int(LOCAL_NEIGHBORHOOD_SIZE * merged.shape[0] / k)
    distances = pairwise_distances(l2.values, metric="euclidean")
    nearest = np.argpartition(distances, n_neighbors + 1, axis=1)[:, : n_neighbors + 1]
    local_density = distances[np.arange(distances.shape[0])[:, None], nearest].sum(axis=1) / n_neighbors
    density_df = pd.DataFrame({"local_density": local_density}, index=l2.index)
    rows: list[dict[str, float]] = []
    selected = None
    selected_metrics: dict[str, float] = {}
    for threshold in DENSITY_CANDIDATES:
        keep = local_density < threshold
        retained = int(keep.sum())
        row: dict[str, float] = {
            "density_threshold": threshold,
            "components_total": int(keep.size),
            "components_retained": retained,
            "components_removed": int((~keep).sum()),
            "retained_fraction": float(keep.mean()),
            "min_cluster_size": np.nan,
            "silhouette": np.nan,
            "eligible_by_locked_rule": False,
        }
        if retained > k:
            labels = KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(l2.values[keep])
            minimum = int(np.bincount(labels, minlength=k).min())
            row["min_cluster_size"] = minimum
            row["silhouette"] = float(silhouette_score(l2.values[keep], labels, metric="euclidean"))
            eligible = keep.mean() >= MIN_RETAINED_FRACTION and minimum >= MIN_CLUSTER_SIZE
            row["eligible_by_locked_rule"] = eligible
            if selected is None and eligible:
                selected = threshold
                selected_metrics = row.copy()
        rows.append(row)
    if selected is None:
        raise RuntimeError("No density threshold satisfies the locked technical retention rule")
    from cnmf.cnmf import save_df_to_npz
    save_df_to_npz(density_df, model.paths["local_density_cache"] % k)
    return float(selected), pd.DataFrame(rows), selected_metrics


def reconstruction_error(model: cNMF, k: int, density_threshold: float) -> float:
    merged = load_df_from_npz(model.paths["merged_spectra"] % k)
    density = load_df_from_npz(model.paths["local_density_cache"] % k).iloc[:, 0]
    l2 = merged.div(np.sqrt((merged**2).sum(axis=1)), axis=0)
    filtered = l2.loc[density < density_threshold]
    labels = pd.Series(KMeans(n_clusters=k, n_init=10, random_state=1).fit_predict(filtered) + 1,
                       index=filtered.index)
    spectra = filtered.groupby(labels).median()
    spectra = spectra.div(spectra.sum(axis=1), axis=0)
    norm_counts = ad.read_h5ad(model.paths["normalized_counts"])
    usages = model.refit_usage(norm_counts.X, spectra)
    x = norm_counts.X
    x_norm2 = float(x.multiply(x).sum()) if hasattr(x, "multiply") else float(np.square(x).sum())
    wt_w = usages.T @ usages
    hh_t = spectra.values @ spectra.values.T
    reconstruction_norm2 = float((wt_w * hh_t).sum())
    xh_t = x @ spectra.values.T
    cross = float((usages * np.asarray(xh_t)).sum())
    return x_norm2 + reconstruction_norm2 - 2.0 * cross


def run_worker(args: argparse.Namespace) -> None:
    model_for(args.cohort_dir).factorize(
        worker_i=args.worker_index, total_workers=args.workers, skip_completed_runs=False
    )


def run_formal(args: argparse.Namespace) -> None:
    cohort = args.cohort
    k = K_BY_COHORT[cohort]
    cohort_dir = OUT / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)
    model = model_for(cohort_dir)
    ledger = Path(model.paths["nmf_replicate_parameters"])
    if ledger.exists():
        raise FileExistsError(f"Refusing to overwrite existing formal cNMF ledger: {ledger}")
    observed_hash = audit_input(cohort, args.input)

    config = {
        "phase": "B1", "cohort": cohort, "input": str(args.input.resolve()),
        "input_sha256": observed_hash, "cnmf_version": "1.7.1", "primary_k": k,
        "replicates": N_ITER, "seed": SEED, "hvg_rule": "cNMF independent top 2000 HVGs per cohort",
        "num_highvar_genes": NUM_HVG, "matrix": ".X raw nonnegative integer UMI counts",
        "beta_loss": "frobenius", "init": "random", "max_nmf_iter": 1000,
        "workers": args.workers, "threads_per_worker": 1, "densify": False,
        "consensus_local_neighborhood_size": LOCAL_NEIGHBORHOOD_SIZE,
        "density_candidates_in_order": list(DENSITY_CANDIDATES),
        "density_selection_rule": (
            "first candidate retaining >=80% replicate components and >=5 components in every KMeans cluster; "
            "candidates 0.5 (cNMF default), 1.0, 2.0; no top-gene inspection"
        ),
        "formal_instability_stop_rule": "silhouette <0.50",
        "dominance_rules_locked_before_results": {
            "NO_DOMINANCE": "largest patient total-usage fraction <0.50",
            "MODERATE_PATIENT_DOMINANCE": "0.50 <= largest patient fraction <0.80",
            "STRONG_PATIENT_DOMINANCE": "largest patient fraction >=0.80",
        },
        "usage_non_negligible_rule": "normalized cell usage >=0.01",
        "appreciable_patient_rule": "patient mean normalized usage >=0.01",
        "technical_flag_rules": {
            "HIGH_MT_TOP50": ">=50% of top-50 score-ranked genes start MT-",
            "HIGH_RIBOSOMAL_TOP50": ">=50% of top-50 score-ranked genes start RPL or RPS",
            "DISSOCIATION_STRESS_TOP50_REVIEW": ">=30% of top-50 genes in locked stress set",
            "VERY_SPARSE_USAGE_REVIEW": "<1% cells have normalized usage >=0.01",
            "PRIVATE_PATIENT_REVIEW": "<=1 patient has mean normalized usage >=0.01",
            "EXTREME_SAMPLE_DEPENDENCE_REVIEW": "largest sample contributes >=80% total usage",
        },
    }
    (cohort_dir / "exact_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model.prepare(
        counts_fn=str(args.input), components=[k], n_iter=N_ITER, densify=False, seed=SEED,
        beta_loss="frobenius", num_highvar_genes=NUM_HVG, init="random", max_NMF_iter=1000,
    )
    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    processes = []
    for worker_index in range(args.workers):
        stdout_path = cohort_dir / f"factorize_worker_{worker_index}.stdout.log"
        stderr_path = cohort_dir / f"factorize_worker_{worker_index}.stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--cohort-dir", str(cohort_dir),
                   "--worker-index", str(worker_index), "--workers", str(args.workers)]
        process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, env=environment)
        processes.append((process, stdout_handle, stderr_handle, stdout_path, stderr_path))
    failures = []
    for process, stdout_handle, stderr_handle, stdout_path, stderr_path in processes:
        code = process.wait()
        stdout_handle.close(); stderr_handle.close()
        if code != 0:
            failures.append(f"exit={code}: {stdout_path}, {stderr_path}")
    if failures:
        raise RuntimeError("; ".join(failures))

    model.combine(components=[k], skip_missing_files=False)
    selected, candidate_metrics, selected_metrics = choose_density(model, k)
    candidate_metrics.to_csv(cohort_dir / "density_threshold_audit.tsv", sep="\t", index=False)
    selected_metrics["prediction_error"] = reconstruction_error(model, k, selected)
    pd.DataFrame([selected_metrics]).to_csv(cohort_dir / "formal_cnmf_metrics.tsv", sep="\t", index=False)
    if float(selected_metrics["silhouette"]) < 0.50:
        (cohort_dir / "FORMAL_STOP_UNSTABLE.txt").write_text(
            f"Consensus silhouette {selected_metrics['silhouette']:.6f} is below locked 0.50 stop threshold.\n",
            encoding="utf-8")
        raise RuntimeError(f"{cohort}: formal consensus significantly unstable; consensus stopped")
    model.consensus(k=k, density_threshold=selected, local_neighborhood_size=LOCAL_NEIGHBORHOOD_SIZE,
                    show_clustering=True, build_ref=False, close_clustergram_fig=True,
                    refit_usage=True, normalize_tpm_spectra=False)
    config["selected_density_threshold"] = selected
    config["selection_metrics"] = selected_metrics
    (cohort_dir / "exact_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cohort", choices=K_BY_COHORT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cohort-dir", type=Path)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--workers", type=int, default=WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        run_worker(parsed)
    else:
        if parsed.cohort is None or parsed.input is None:
            raise SystemExit("--cohort and --input are required")
        run_formal(parsed)
