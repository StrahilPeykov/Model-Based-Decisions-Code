"""
Parameter sweeps for Part 1 (baseline dynamics).

Runs a grid over initial adoption (X0) and initial infrastructure (I0),
optionally over multiple beta_I values, and stores one row per run in
`results/baseline_sweep.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sim_utils import SimulationConfig, run_simulation


def _network_kwargs_from_args(args: argparse.Namespace) -> Dict:
    if args.network_type == "erdos_renyi":
        return {"p": args.p}
    if args.network_type == "barabasi_albert":
        return {"m": args.m}
    if args.network_type == "small_world":
        return {"k": args.k, "rewiring_p": args.rewiring_p}
    if args.network_type == "grid":
        return {}
    return {}


def _first_crossing_time(
    series: Optional[List[Dict[str, float]]],
    threshold: float,
    key: str = "X",
) -> Optional[int]:
    if series is None:
        return None
    for t, state in enumerate(series):
        x = state.get(key)
        if x is not None and x >= threshold:
            return t
    return None

def main():
    parser = argparse.ArgumentParser(description="Baseline sweep over X0, I0, beta_I.")
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--network-type", type=str, default="erdos_renyi")
    parser.add_argument("--n-nodes", type=int, default=300)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--rewiring-p", type=float, default=0.05)
    parser.add_argument("--ratio", type=float, default=1.3)
    parser.add_argument("--beta-list", type=float, nargs="+", default=[0.6, 0.8, 1.0])
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--g-i", type=float, default=0.08)
    parser.add_argument("--x0-min", type=float, default=0.04)
    parser.add_argument("--x0-max", type=float, default=0.12)
    parser.add_argument("--x0-points", type=int, default=7)
    parser.add_argument("--i0-min", type=float, default=0.02)
    parser.add_argument("--i0-max", type=float, default=0.06)
    parser.add_argument("--i0-points", type=int, default=7)
    parser.add_argument("--init-method", type=str, default="random")
    parser.add_argument("--strategy", type=str, default="logit")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="results/baseline_sweep.csv")
    args = parser.parse_args()

    x0_values = np.linspace(args.x0_min, args.x0_max, args.x0_points)
    i0_values = np.linspace(args.i0_min, args.i0_max, args.i0_points)
    beta_values = args.beta_list

    rows: List[Dict] = []
    run_id = 0
    network_kwargs = _network_kwargs_from_args(args)

    for beta_I in beta_values:
        for X0 in x0_values:
            for I0 in i0_values:
                for rep in range(args.reps):
                    seed = args.seed_base + run_id
                    config = SimulationConfig(
                        T=args.horizon,
                        n_nodes=args.n_nodes,
                        network_type=args.network_type,
                        network_kwargs=network_kwargs,
                        ratio=args.ratio,
                        beta_I=beta_I,
                        b=args.b,
                        g_I=args.g_i,
                        I0=I0,
                        X0_frac=float(X0),
                        init_method=args.init_method,
                        strategy_choice_func=args.strategy,
                        tau=args.tau,
                        collect_series=True,
                    )

                    row, series = run_simulation(
                        config,
                        seed=seed,
                        policy_config=None,
                        run_id=run_id,
                        convergence_tol=1e-3,
                        patience=25,
                    )

                    x_final = float(row.get("X_final", np.nan))

                    row.update({
                        "run_id": run_id,
                        "X0": float(X0),
                        "I0_grid": float(I0),
                        "beta_I_grid": float(beta_I),
                        "high_adoption": int(np.isfinite(x_final) and x_final >= 0.8),
                        "low_adoption": int(np.isfinite(x_final) and x_final <= 0.2),
                        "t_high": _first_crossing_time(series, 0.8),
                        "t_half": _first_crossing_time(series, 0.5),
                    })

                    rows.append(row)
                    run_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # summary table
    summary = (
        df.groupby(["beta_I_grid", "X0", "I0_grid"], dropna=False)
        .agg(
            n=("run_id", "count"),
            X_mean=("X_final", "mean"),
            X_std=("X_final", "std"),
            p_high=("high_adoption", "mean"),
            p_low=("low_adoption", "mean"),
            t_high_mean=("t_high", "mean"),
        )
        .reset_index()
    )

    # True bistability = two basins observed
    summary["bistable"] = (
        (summary["p_high"] >= 0.1) &
        (summary["p_low"] >= 0.1)
    ).astype(int)

    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
