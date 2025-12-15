"""
Generate illustrative time-series trajectories for phase plots.

Designed to show bistability: same parameters, slightly different
initial conditions, different long-run outcomes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
import pandas as pd

from sim_utils import SimulationConfig, run_simulation
from run_baseline_sweeps import _network_kwargs_from_args


def _phase_cases(args: argparse.Namespace, net_type: str) -> List[Tuple[str, SimulationConfig]]:
    common = dict(
        T=args.horizon,
        n_nodes=args.n_nodes,
        network_type=net_type,
        network_kwargs=_network_kwargs_from_args(args),
        ratio=args.ratio,
        beta_I=args.beta_I,
        b=args.b,
        g_I=args.g_i,
        init_method=args.init_method,
        strategy_choice_func=args.strategy,
        tau=args.tau,
        collect_series=True,
    )

    return [
        ("low_seed",  SimulationConfig(I0=0.02, X0_frac=0.05, **common)),
        ("mid_seed",  SimulationConfig(I0=0.03, X0_frac=0.08, **common)),
        ("high_seed", SimulationConfig(I0=0.04, X0_frac=0.12, **common)),
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate phase trajectories (bistable regime).")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=999)

    parser.add_argument(
        "--network-types",
        nargs="+",
        default=["grid", "small_world", "erdos_renyi", "barabasi_albert"],
    )

    parser.add_argument("--n-nodes", type=int, default=300)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--rewiring-p", type=float, default=0.05)

    parser.add_argument("--ratio", type=float, default=1.4)
    parser.add_argument("--beta-I", dest="beta_I", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--g-i", type=float, default=0.08)

    parser.add_argument("--init-method", type=str, default="random")
    parser.add_argument("--strategy", type=str, default="imitate")
    parser.add_argument("--tau", type=float, default=1.0)

    parser.add_argument("--runs-output", type=str, default="results/phase_runs.csv")
    parser.add_argument("--series-output", type=str, default="results/phase_timeseries.csv")
    args = parser.parse_args()

    run_rows: List[Dict] = []
    series_rows: List[Dict] = []

    run_id = 0
    for net_type in args.network_types:
        args.network_type = net_type
        cases = _phase_cases(args, net_type)

        for case_name, config in cases:
            for rep in range(args.reps):
                seed = args.seed_base + run_id
                row, ts = run_simulation(
                    config,
                    seed=seed,
                    policy_config=None,
                    run_id=run_id,
                    convergence_tol=1e-3,
                    patience=30,
                    record_series=True,
                )
                row.update({
                    "network_label": net_type,
                    "scenario": case_name,
                })
                run_rows.append(row)

                if ts:
                    for entry in ts:
                        entry.update({
                            "network_label": net_type,
                            "scenario": case_name,
                        })
                    series_rows.extend(ts)

                run_id += 1

    Path(args.runs_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.series_output).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(run_rows).to_csv(args.runs_output, index=False)
    pd.DataFrame(series_rows).to_csv(args.series_output, index=False)

    print(f"Saved phase runs to {args.runs_output}")
    print(f"Saved phase timeseries to {args.series_output}")


if __name__ == "__main__":
    main()
