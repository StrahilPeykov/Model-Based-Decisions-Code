"""
Network structure experiments (Part 2).

Compares adoption dynamics across multiple network topologies by running
many trials per type and saving a tidy CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from experiments.sim_utils import SimulationConfig, run_simulation


def _kwargs_for_type(net_type: str, args: argparse.Namespace) -> Dict:
    if net_type == "erdos_renyi":
        return {"p": args.p}
    if net_type == "barabasi_albert":
        return {"m": args.m}
    if net_type == "small_world":
        return {"k": args.k, "rewiring_p": args.rewiring_p}
    if net_type == "grid":
        return {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Compare network types.")
    parser.add_argument("--network-types", nargs="+", default=["grid", "small_world", "erdos_renyi", "barabasi_albert"])
    parser.add_argument("--runs-per-type", type=int, default=20, help="Number of runs per network type.")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--seed-base", type=int, default=555)
    parser.add_argument("--n-nodes", type=int, default=300)
    parser.add_argument("--p", type=float, default=0.05, help="Edge prob (ER).")
    parser.add_argument("--m", type=int, default=2, help="Edges per new node (BA).")
    parser.add_argument("--k", type=int, default=6, help="Mean degree (small-world).")
    parser.add_argument("--rewiring-p", type=float, default=0.05, help="Rewiring prob (small-world).")
    parser.add_argument("--ratio", type=float, default=2.0)
    parser.add_argument("--beta-I", dest="beta_I", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--g-i", type=float, default=0.08)
    parser.add_argument("--i0", type=float, default=0.05)
    parser.add_argument("--x0", type=float, default=0.15)
    parser.add_argument("--init-method", type=str, default="random")
    parser.add_argument("--strategy", type=str, default="imitate")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="results/network_comparison.csv")
    args = parser.parse_args()

    rows: List[Dict] = []
    run_id = 0
    for net_type in args.network_types:
        kwargs = _kwargs_for_type(net_type, args)
        for rep in range(args.runs_per_type):
            seed = args.seed_base + run_id
            config = SimulationConfig(
                T=args.horizon,
                n_nodes=args.n_nodes,
                network_type=net_type,
                network_kwargs=kwargs,
                ratio=args.ratio,
                beta_I=args.beta_I,
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
                patience=30,
            )
            row.update({"network_label": net_type})
            rows.append(row)
            run_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
