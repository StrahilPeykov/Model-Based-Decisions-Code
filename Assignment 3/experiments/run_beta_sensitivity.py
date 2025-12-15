"""
Sweep over beta_I for fixed initial conditions.

Generates one row per run (multiple seeds per beta_I) to assess
sensitivity of tipping behaviour to infrastructure feedback strength.
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
from typing import Dict, List

import numpy as np
import pandas as pd

from .sim_utils import SimulationConfig, run_simulation


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


def main():
    parser = argparse.ArgumentParser(description="Beta_I sensitivity sweep.")
    parser.add_argument("--horizon", type=int, default=250, help="Time steps per run.")
    parser.add_argument("--reps", type=int, default=10, help="Trials per beta_I value.")
    parser.add_argument("--seed-base", type=int, default=321, help="Base RNG seed.")
    parser.add_argument("--network-type", type=str, default="erdos_renyi", help="Network type.")
    parser.add_argument("--n-nodes", type=int, default=300, help="Number of agents.")
    parser.add_argument("--p", type=float, default=0.05, help="Edge prob (ER).")
    parser.add_argument("--m", type=int, default=2, help="Edges per new node (BA).")
    parser.add_argument("--k", type=int, default=6, help="Mean degree (small-world).")
    parser.add_argument("--rewiring-p", type=float, default=0.05, help="Rewiring prob (small-world).")
    parser.add_argument("--ratio", type=float, default=1.4, help="Initial payoff ratio a_I/b.")
    parser.add_argument("--beta-list", type=float, nargs="+", default=[0.4, 0.6, 0.8, 1.0, 1.2], help="beta_I values.")
    parser.add_argument("--b", type=float, default=1.0, help="Defection payoff.")
    parser.add_argument("--g-i", type=float, default=0.08, help="Infrastructure rate.")
    parser.add_argument("--i0", type=float, default=0.03, help="Initial infrastructure.")
    parser.add_argument("--x0", type=float, default=0.08, help="Initial EV share.")
    parser.add_argument("--init-method", type=str, default="random", help="random|degree_high|degree_low|centrality")
    parser.add_argument("--strategy", type=str, default="imitate", help="imitate|logit")
    parser.add_argument("--tau", type=float, default=1.0, help="Logit temperature.")
    parser.add_argument("--output", type=str, default="results/beta_sensitivity.csv", help="Path to save CSV.")
    args = parser.parse_args()

    network_kwargs = _network_kwargs_from_args(args)
    rows: List[Dict] = []
    run_id = 0

    for beta_I in args.beta_list:
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
                I0=args.i0,
                X0_frac=args.x0,
                init_method=args.init_method,
                strategy_choice_func=args.strategy,
                tau=args.tau,
                collect_series=False,
            )
            row, _ = run_simulation(
                config,
                seed=seed,
                policy_config=None,
                run_id=run_id,
                convergence_tol=1e-3,
                patience=25,
            )
            row.update({"beta_I_grid": beta_I})
            rows.append(row)
            run_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
