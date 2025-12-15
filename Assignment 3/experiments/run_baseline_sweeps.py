"""
Parameter sweeps for Part 1 (baseline dynamics).

Runs a grid over initial adoption (X0) and initial infrastructure (I0),
optionally over multiple beta_I values, and stores one row per run in
`results/baseline_sweep.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import argparse
from pathlib import Path
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

def _first_crossing_time(series: Optional[List[float]], threshold: float) -> Optional[int]:
    if series is None:
        return None
    for t, x in enumerate(series):
        if x >= threshold:
            return t
    return None

def main():
    parser = argparse.ArgumentParser(description="Baseline sweep over X0, I0, beta_I.")
    parser.add_argument("--horizon", type=int, default=200, help="Time steps per run.")
    parser.add_argument("--reps", type=int, default=5, help="Trials per grid cell.")
    parser.add_argument("--seed-base", type=int, default=123, help="Base seed for runs.")
    parser.add_argument("--network-type", type=str, default="erdos_renyi", help="Network type (erdos_renyi|barabasi_albert|small_world|grid).")
    parser.add_argument("--n-nodes", type=int, default=300, help="Number of agents.")
    parser.add_argument("--p", type=float, default=0.05, help="Edge prob (ER).")
    parser.add_argument("--m", type=int, default=2, help="Edges per new node (BA).")
    parser.add_argument("--k", type=int, default=6, help="Mean degree (small-world).")
    parser.add_argument("--rewiring-p", type=float, default=0.05, help="Rewiring prob (small-world).")
    parser.add_argument("--ratio", type=float, default=2.2, help="Initial payoff ratio a_I/b.")
    parser.add_argument("--beta-list", type=float, nargs="+", default=[1.5, 2.0, 2.5], help="beta_I values to sweep.")
    parser.add_argument("--b", type=float, default=1.0, help="Defection payoff.")
    parser.add_argument("--g-i", type=float, default=0.08, help="Infrastructure adjustment rate.")
    parser.add_argument("--x0-min", type=float, default=0.0)
    parser.add_argument("--x0-max", type=float, default=0.6)
    parser.add_argument("--x0-points", type=int, default=7)
    parser.add_argument("--i0-min", type=float, default=0.0)
    parser.add_argument("--i0-max", type=float, default=0.3)
    parser.add_argument("--i0-points", type=int, default=7)
    parser.add_argument("--init-method", type=str, default="random", help="random|degree_high|degree_low|centrality")
    parser.add_argument("--strategy", type=str, default="imitate", help="imitate|logit")
    parser.add_argument("--tau", type=float, default=1.0, help="Logit temperature.")
    parser.add_argument("--output", type=str, default="results/baseline_sweep.csv", help="Path to save CSV.")
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
                    row.update({"X0": float(X0), "I0_grid": float(I0), "beta_I_grid": float(beta_I)})

                    if "high_adoption" not in row:
                        x_final = float(row.get("X_final", np.nan))
                        row["high_adoption"] = int(np.isfinite(x_final) and x_final > 0.8)

                    if "t_high" not in row:
                        row["t_high"] = _first_crossing_time(series, 0.8)

                    if "t_half" not in row:
                        row["t_half"] = _first_crossing_time(series, 0.5)

                    rows.append(row)
                    run_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")

    # Summary table: probability of high adoption + speed proxy + bistability flag
    required_cols = {"run_id", "X_final", "high_adoption", "beta_I_grid", "X0", "I0_grid"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for summary computation: {sorted(missing)}")

    group_cols = ["beta_I_grid", "X0", "I0_grid"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("run_id", "count"),
            X_mean=("X_final", "mean"),
            X_std=("X_final", "std"),
            p_high=("high_adoption", "mean"),
            t_high_mean=("t_high", "mean"),
        )
        .reset_index()
    )
    summary["bistable"] = ((summary["p_high"] >= 0.2) & (summary["p_high"] <= 0.8)).astype(int)

    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)

if __name__ == "__main__":
    main()
