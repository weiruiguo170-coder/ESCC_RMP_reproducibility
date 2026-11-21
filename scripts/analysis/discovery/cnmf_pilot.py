#!/usr/bin/env python
"""Run one independent, lightweight cNMF K-selection pilot (Phase B0 only)."""

from __future__ import annotations

from pathlib import Path as _Path
import sys as _sys
_root=next(p for p in _Path(__file__).resolve().parents if (p/'scripts/_paths.py').is_file())
_sys.path.insert(0,str(_root/'scripts'))
from _paths import ROOT,REPO_ROOT,INPUT_ROOT,OUTPUT_ROOT,data_path,resolve_data,read_table


import argparse
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cnmf.cnmf import cNMF, load_df_from_npz


SEED = 20260826
N_ITER = 20
NUM_HVG = 2000
WORKERS = 4


def get_cnmf(cohort_dir: Path) -> cNMF:
    return cNMF(output_dir=str(cohort_dir), name="pilot")


def worker(args: argparse.Namespace) -> None:
    model = get_cnmf(args.cohort_dir)
    model.factorize(worker_i=args.worker_index, total_workers=args.workers, skip_completed_runs=False)


def count_completed(model: cNMF, k: int, run_params: pd.DataFrame) -> int:
    rows = run_params.loc[run_params["n_components"].astype(int).eq(k)]
    return sum(Path(model.paths["iter_spectra"] % (k, int(row["iter"]))).exists() for _, row in rows.iterrows())


def orchestrate(args: argparse.Namespace) -> None:
    args.cohort_dir.mkdir(parents=True, exist_ok=True)
    model = get_cnmf(args.cohort_dir)
    ledger = Path(model.paths["nmf_replicate_parameters"])
    if ledger.exists():
        raise FileExistsError(f"Refusing to overwrite existing cNMF pilot ledger: {ledger}")

    model.prepare(
        counts_fn=str(args.input),
        components=args.ks,
        n_iter=N_ITER,
        densify=False,
        seed=SEED,
        beta_loss="frobenius",
        num_highvar_genes=NUM_HVG,
        init="random",
        max_NMF_iter=1000,
    )

    environment = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    processes: list[tuple[subprocess.Popen, object, object, Path, Path]] = []
    for worker_index in range(args.workers):
        stdout_path = args.cohort_dir / f"factorize_worker_{worker_index}.stdout.log"
        stderr_path = args.cohort_dir / f"factorize_worker_{worker_index}.stderr.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--cohort-dir", str(args.cohort_dir), "--worker-index", str(worker_index),
            "--workers", str(args.workers),
        ]
        process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle, env=environment)
        processes.append((process, stdout_handle, stderr_handle, stdout_path, stderr_path))

    failures: list[str] = []
    for process, stdout_handle, stderr_handle, stdout_path, stderr_path in processes:
        code = process.wait()
        stdout_handle.close()
        stderr_handle.close()
        if code != 0:
            failures.append(f"PID {process.pid} exit={code}; logs={stdout_path},{stderr_path}")
    if failures:
        raise RuntimeError("; ".join(failures))

    model.combine(components=args.ks, skip_missing_files=False)
    model.k_selection_plot(close_fig=True)

    stats = load_df_from_npz(model.paths["k_selection_stats"]).copy()
    run_params = load_df_from_npz(model.paths["nmf_replicate_parameters"])
    stats["replicates_requested"] = N_ITER
    stats["replicates_completed"] = [count_completed(model, int(k), run_params) for k in stats["k"]]
    stats["run_status"] = ["PASS" if value == N_ITER else "INCOMPLETE" for value in stats["replicates_completed"]]
    stats["density_filter_applied_for_rank_stats"] = False
    stats["local_neighborhood_size"] = 0.3
    stats["seed"] = SEED
    stats["hvg_requested"] = NUM_HVG
    genes = [line for line in Path(model.paths["nmf_genes_list"]).read_text(encoding="utf-8").splitlines() if line]
    stats["hvg_used"] = len(genes)
    stats["beta_loss"] = "frobenius"
    stats["init"] = "random"
    stats["max_nmf_iter"] = 1000
    warnings = []
    errors = []
    for _, _, _, stdout_path, stderr_path in processes:
        combined = stdout_path.read_text(encoding="utf-8", errors="replace") + "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")
        warnings.extend(line.strip() for line in combined.splitlines() if "warning" in line.lower())
        errors.extend(line.strip() for line in combined.splitlines() if "error" in line.lower() or "traceback" in line.lower())
    stats["warning_count"] = len(warnings)
    stats["error_count"] = len(errors)
    stats.to_csv(args.cohort_dir / "k_selection_metrics.tsv", sep="\t", index=False)
    (args.cohort_dir / "pilot_warnings_errors.txt").write_text(
        "WARNINGS\n" + ("\n".join(warnings) if warnings else "NONE") +
        "\n\nERRORS\n" + ("\n".join(errors) if errors else "NONE") + "\n",
        encoding="utf-8",
    )

    fig, left = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    right = left.twinx()
    left.plot(stats["k"], stats["silhouette"], "o-", color="#1f77b4", label="Consensus stability")
    right.plot(stats["k"], stats["prediction_error"], "o-", color="#d62728", label="Reconstruction error")
    left.set_xlabel("K")
    left.set_ylabel("Consensus stability (silhouette)", color="#1f77b4")
    right.set_ylabel("Reconstruction error (Frobenius SSE)", color="#d62728")
    left.set_title(f"{args.cohort}: independent cNMF rank-selection pilot")
    left.grid(alpha=0.25)
    left.set_xticks(stats["k"])
    fig.savefig(args.cohort_dir / "k_selection_plot.png", dpi=250)
    fig.savefig(args.cohort_dir / "k_selection_plot.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--cohort")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--ks", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--worker-index", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker:
        worker(parsed)
    else:
        if not parsed.cohort or not parsed.input or not parsed.ks:
            raise SystemExit("--cohort, --input, and --ks are required")
        orchestrate(parsed)
