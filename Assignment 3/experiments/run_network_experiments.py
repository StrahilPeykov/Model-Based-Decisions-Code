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

from .sim_utils import SimulationConfig, run_simulation


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
    parser.add_argument(
        "--network-types",
        nargs="+",
        default=["grid", "small_world", "erdos_renyi", "barabasi_albert"],
    )
    parser.add_argument("--runs-per-type", type=int, default=20, help="Runs per network type.")
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

    # tipping regime support
    parser.add_argument("--baseline-summary", type=str, default=None)
    parser.add_argument("--pick-bistable", type=int, default=0)

    args = parser.parse_args()

    scenarios: List[Dict] = []
    if args.baseline_summary is None:
        scenarios.append({"scenario_id": 0, "beta_I": args.beta_I, "X0": args.x0, "I0": args.i0})
    else:
        if args.pick_bistable <= 0:
            raise ValueError("--pick-bistable must be > 0 when --baseline-summary is provided.")

        baseline_df = pd.read_csv(args.baseline_summary)
        required = {"beta_I_grid", "X0", "I0_grid"}
        missing = required.difference(baseline_df.columns)
        if missing:
            raise ValueError(f"Baseline summary missing required columns: {sorted(missing)}")

        # choose bistable rows
        if "bistable" in baseline_df.columns:
            bistable_rows = baseline_df[baseline_df["bistable"] == 1].copy()
        elif "p_high" in baseline_df.columns:
            bistable_rows = baseline_df[(baseline_df["p_high"] >= 0.2) & (baseline_df["p_high"] <= 0.8)].copy()
        else:
            raise ValueError("Baseline summary must have 'bistable' or 'p_high'.")

        if bistable_rows.empty:
            raise ValueError("No bistable/tipping rows found in baseline summary.")

        # ranking: prefer high variance if available; otherwise closest to 0.5 probability
        if "X_std" in bistable_rows.columns:
            bistable_rows = bistable_rows.sort_values("X_std", ascending=False)
        else:
            if "p_high" not in bistable_rows.columns:
                raise ValueError("Need either 'X_std' or 'p_high' to rank bistable scenarios.")
            bistable_rows = bistable_rows.assign(p_distance=(bistable_rows["p_high"] - 0.5).abs()).sort_values(
                "p_distance", ascending=True
            )

        selected_rows = bistable_rows.head(args.pick_bistable)

        for sid, (orig_idx, row) in enumerate(selected_rows.iterrows()):
            scenarios.append(
                {
                    "scenario_id": sid,
                    "baseline_cell_id": int(orig_idx),
                    "beta_I": float(row["beta_I_grid"]),
                    "X0": float(row["X0"]),
                    "I0": float(row["I0_grid"]),
                }
            )

        print(f"Loaded {len(scenarios)} bistable scenarios from {args.baseline_summary}")

    rows: List[Dict] = []
    run_id = 0
    for scenario in scenarios:
        for net_type in args.network_types:
            kwargs = _kwargs_for_type(net_type, args)
            for _ in range(args.runs_per_type):
                seed = args.seed_base + run_id
                config = SimulationConfig(
                    T=args.horizon,
                    n_nodes=args.n_nodes,
                    network_type=net_type,
                    network_kwargs=kwargs,
                    ratio=args.ratio,
                    beta_I=float(scenario["beta_I"]),
                    b=args.b,
                    g_I=args.g_i,
                    I0=float(scenario["I0"]),
                    X0_frac=float(scenario["X0"]),
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
                row.update(
                    {
                        "scenario_id": scenario["scenario_id"],
                        "baseline_cell_id": scenario.get("baseline_cell_id"),
                        "network_label": net_type,
                        "scenario_beta_I": float(scenario["beta_I"]),
                        "scenario_X0": float(scenario["X0"]),
                        "scenario_I0": float(scenario["I0"]),
                    }
                )
                rows.append(row)
                run_id += 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(rows)} rows to {out_path}")

    group_cols = ["scenario_id", "network_label"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("run_id", "count"),
            p_high=("high_adoption", "mean"),
            t_high_mean=("t_high", "mean"),
            X_mean=("X_final", "mean"),
            X_std=("X_final", "std"),
            cluster_max_mean=("cluster_max_final", "mean"),
        )
        .reset_index()
    )
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
