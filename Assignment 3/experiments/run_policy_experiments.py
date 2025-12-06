"""
Policy intervention experiments (Part 3).

Defines a few targeted/timed interventions and runs them against a
baseline scenario. Results are saved as CSVs for later plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from experiments.sim_utils import SimulationConfig, run_simulation
from experiments.run_baseline_sweeps import _network_kwargs_from_args


def _policies(args: argparse.Namespace) -> List[Tuple[str, Dict | None]]:
    """Return a list of (label, policy_config) pairs."""
    return [
        ("baseline", None),
        (
            "early_hubs_seed",
            {
                "type": "targeted_seed",
                "params": {"fraction": 0.05, "metric": "degree", "start": 0, "duration": 10, "lock": True, "infrastructure_boost": 0.05},
            },
        ),
        (
            "mid_hubs_seed",
            {
                "type": "targeted_seed",
                "params": {"fraction": 0.05, "metric": "degree", "start": 25, "duration": 10, "lock": True, "infrastructure_boost": 0.0},
            },
        ),
        (
            "infra_shock",
            {
                "type": "infrastructure",
                "params": {"start": 10, "boost": 0.15, "once": True},
            },
        ),
        (
            "time_limited_subsidy",
            {
                "type": "subsidy",
                "params": {"start": 5, "end": 40, "delta_a0": 0.3, "delta_beta_I": 0.0},
            },
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description="Targeted/timed policy experiments.")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--reps", type=int, default=20, help="Runs per policy.")
    parser.add_argument("--seed-base", type=int, default=777)
    parser.add_argument("--network-type", type=str, default="small_world")
    parser.add_argument("--n-nodes", type=int, default=300)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--rewiring-p", type=float, default=0.05)
    parser.add_argument("--ratio", type=float, default=1.8)
    parser.add_argument("--beta-I", dest="beta_I", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--g-i", type=float, default=0.08)
    parser.add_argument("--i0", type=float, default=0.02)
    parser.add_argument("--x0", type=float, default=0.08)
    parser.add_argument("--init-method", type=str, default="random")
    parser.add_argument("--strategy", type=str, default="imitate")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--collect-series", action="store_true", help="Store full trajectories.")
    parser.add_argument("--runs-output", type=str, default="results/policy_runs.csv")
    parser.add_argument("--series-output", type=str, default="results/policy_timeseries.csv")
    args = parser.parse_args()

    network_kwargs = _network_kwargs_from_args(args)
    policies = _policies(args)
    run_rows: List[Dict] = []
    series_rows: List[Dict] = []

    run_id = 0
    for label, policy in policies:
        for rep in range(args.reps):
            seed = args.seed_base + run_id
            config = SimulationConfig(
                T=args.horizon,
                n_nodes=args.n_nodes,
                network_type=args.network_type,
                network_kwargs=network_kwargs,
                ratio=args.ratio,
                beta_I=args.beta_I,
                b=args.b,
                g_I=args.g_i,
                I0=args.i0,
                X0_frac=args.x0,
                init_method=args.init_method,
                strategy_choice_func=args.strategy,
                tau=args.tau,
                collect_series=args.collect_series,
            )
            row, ts = run_simulation(
                config,
                seed=seed,
                policy_config=policy,
                run_id=run_id,
                convergence_tol=1e-3,
                patience=30,
                record_series=args.collect_series,
            )
            row["policy_label"] = label
            run_rows.append(row)
            if ts:
                for entry in ts:
                    entry["policy_label"] = label
                series_rows.extend(ts)
            run_id += 1

    runs_path = Path(args.runs_output)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(runs_path, index=False)
    print(f"Saved run summaries to {runs_path}")

    if args.collect_series:
        series_path = Path(args.series_output)
        series_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(series_rows).to_csv(series_path, index=False)
        print(f"Saved trajectories to {series_path}")


if __name__ == "__main__":
    main()
