"""
Policy intervention experiments (Part 3).

Defines a few targeted/timed interventions and runs them against a
baseline scenario or multiple bistable/tipping scenarios. 
Results are saved as CSVs for later plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from experiments.sim_utils import SimulationConfig, run_simulation
from experiments.run_baseline_sweeps import _network_kwargs_from_args

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

def _load_scenarios(args: argparse.Namespace) -> List[Dict]:
    scenarios: List[Dict] = []

    if args.baseline_summary is None:
        scenarios.append({"scenario_id": 0, "beta_I": args.beta_I, "X0": args.x0, "I0": args.i0})
        return scenarios

    if args.pick_bistable <= 0:
        raise ValueError("--pick-bistable must be > 0 when --baseline-summary is provided.")

    df = pd.read_csv(args.baseline_summary)
    required = {"beta_I_grid", "X0", "I0_grid"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline summary missing required columns: {sorted(missing)}")

    if "bistable" in df.columns:
        cand = df[df["bistable"] == 1].copy()
    elif "p_high" in df.columns:
        cand = df[(df["p_high"] >= 0.2) & (df["p_high"] <= 0.8)].copy()
    else:
        raise ValueError("Baseline summary must have 'bistable' or 'p_high'.")

    if cand.empty:
        raise ValueError("No bistable/tipping rows found in baseline summary.")

    if "X_std" in cand.columns:
        cand = cand.sort_values("X_std", ascending=False)
    else:
        if "p_high" not in cand.columns:
            raise ValueError("Need either 'X_std' or 'p_high' to rank scenarios.")
        cand = cand.assign(p_distance=(cand["p_high"]-0.5).abs()).sort_values("p_distance", ascending=True)

    selected = cand.head(args.pick_bistable)
    for sid, (orig_idx, row) in enumerate(selected.iterrows()):
        scenarios.append(
            {
                "scenario_id": sid,
                "baseline_cell_id": int(orig_idx),
                "beta_I": float(row["beta_I_grid"]),
                "X0": float(row["X0"]),
                "I0": float(row["I0_grid"]),
            }
        )

    print(f"Loaded {len(scenarios)} scenarios from {args.baseline_summary}")
    return scenarios

def main():
    parser = argparse.ArgumentParser(description="Targeted/timed policy experiments.")
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--reps", type=int, default=20, help="Runs per policy.")
    parser.add_argument("--seed-base", type=int, default=777)
    parser.add_argument("--network-types", nargs="+", default=["small_world"], help="Network types (grid|small_world|erdos_renyi|barabasi_albert).",
    )
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
    parser.add_argument("--baseline-summary", type=str, default=None)
    parser.add_argument("--pick-bistable", type=int, default=0)
    args = parser.parse_args()

    scenarios = _load_scenarios(args)
    policies = _policies(args)
    run_rows: List[Dict] = []
    series_rows: List[Dict] = []

    run_id = 0
    for scenario in scenarios:
        for net_type in args.network_types:
            network_kwargs = _kwargs_for_type(net_type, args)
            for label, policy in policies:
                for rep in range(args.reps):
                    seed = args.seed_base + run_id
                    config = SimulationConfig(
                        T=args.horizon,
                        n_nodes=args.n_nodes,
                        network_type=net_type,
                        network_kwargs=network_kwargs,
                        ratio=args.ratio,
                        beta_I=float(scenario["beta_I"]),
                        b=args.b,
                        g_I=args.g_i,
                        I0=float(scenario["I0"]),
                        X0_frac=float(scenario["X0"]),
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
                    row.update(
                        {
                            "scenario_id": scenario["scenario_id"],
                            "baseline_cell_id": scenario.get("baseline_cell_id"),
                            "network_label": net_type,
                            "policy_label": label,
                            "scenario_beta_I": float(scenario["beta_I"]),
                            "scenario_X0": float(scenario["X0"]),
                            "scenario_I0": float(scenario["I0"]),
                        }
                    )
                    run_rows.append(row)
                    if ts:
                        for entry in ts:
                            entry.update(
                                {
                                    "scenario_id": scenario["scenario_id"],
                                    "baseline_cell_id": scenario.get("baseline_cell_id"),
                                    "network_label": net_type,
                                    "policy_label": label,
                                }
                            )
                        series_rows.extend(ts)
                    run_id += 1

    runs_path = Path(args.runs_output)
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(run_rows)
    df.to_csv(runs_path, index=False)
    print(f"Saved run summaries to {runs_path}")

    #summary
    group_cols = ["scenario_id", "network_label", "policy_label"]
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
    summary_path = runs_path.with_name(runs_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    if args.collect_series:
        series_path = Path(args.series_output)
        series_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(series_rows).to_csv(series_path, index=False)
        print(f"Saved trajectories to {series_path}")


if __name__ == "__main__":
    main()
