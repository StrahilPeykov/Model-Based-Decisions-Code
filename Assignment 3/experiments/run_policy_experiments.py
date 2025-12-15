"""
Policy intervention experiments (Part 3).

Defines a few targeted/timed interventions and runs them against a
baseline scenario or multiple bistable/tipping scenarios. 

Sweeps over policy intensities and timings
Runs across multiple network structures
Adds (policy) cost proxy and parameter columns
Writes CSVs
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
from typing import Dict, List, Tuple

import pandas as pd

from sim_utils import SimulationConfig, run_simulation
from run_baseline_sweeps import _network_kwargs_from_args

def _kwargs_for_type(net_type: str, args: argparse.Namespace) -> Dict:
    n = args.n_nodes
    if net_type == "erdos_renyi":
        return {"p": 6.0 / n}      
    if net_type == "barabasi_albert":
        return {"m": 3}            
    if net_type == "small_world":
        return {"k": 6, "rewiring_p": args.rewiring_p}
    if net_type == "grid":
        return {}                  
    return {}


def _policy_cost(meta: Dict, n_nodes: int) -> float:
    """Cheap comparable cost proxy"""
    ptype = meta.get("ptype")
    if ptype == "targeted_seed":
        frac = float(meta["fraction"])
        dur = int(meta["duration"])
        k = int(round(frac*n_nodes))
        return float(k*dur)
    if ptype == "subsidy":
        dur = int(meta["duration"])
        da0 = float(meta["delta_a0"])
        return float(dur*da0)
    if ptype == "infrastructure":
        return float(meta["boost"])
    return 0.0

def _policies(args: argparse.Namespace) -> List[Tuple[str, Dict | None, Dict]]:
    """Return a list of (label, policy_config) pairs."""
    policies: List[Tuple[str, Dict | None, Dict]] = [("baseline", None, {"ptype": "baseline"})]

    for frac in args.seed_fracs:
        for start in args.seed_starts:
            for dur in args.seed_durations:
                label = f"seed:{args.seed_metric}:f{frac}:s{start}:d{dur}:lock{int(args.seed_lock)}"
                cfg = {
                    "type": "targeted_seed",
                    "params": {
                        "fraction": float(frac),
                        "metric": args.seed_metric,
                        "start": int(start),
                        "duration": int(dur),
                        "lock": bool(args.seed_lock),
                        "infrastructure_boost": 0.0,
                    },
                }
                meta = {
                    "ptype": "targeted_seed",
                    "metric": args.seed_metric,
                    "fraction": float(frac),
                    "start": int(start),
                    "duration": int(dur),
                    "lock": bool(args.seed_lock),
                }
                policies.append((label, cfg, meta))

    for da0 in args.subsidy_delta_a0:
        for start in args.subsidy_starts:
            for dur in args.subsidy_durations:
                end = int(start) + int(dur)
                label = f"subsidy:da0{da0}:s{start}:d{dur}"
                cfg = {
                    "type": "subsidy",
                    "params": {
                        "start": int(start),
                        "end": end,
                        "delta_a0": float(da0),
                        "delta_beta_I": 0.0,
                    },
                }
                meta = {"ptype": "subsidy", "delta_a0": float(da0), "start": int(start), "duration": int(dur)}
                policies.append((label, cfg, meta))

    for boost in args.infra_boosts:
        for start in args.infra_starts:
            label = f"infra:boost{boost}:s{start}"
            cfg = {"type": "infrastructure", "params": {"start": int(start), "boost": float(boost), "once": True}}
            meta = {"ptype": "infrastructure", "boost": float(boost), "start": int(start), "duration": 1}
            policies.append((label, cfg, meta))

    return policies


def _load_scenarios(args: argparse.Namespace) -> List[Dict]:
    scenarios: List[Dict] = []

    if args.baseline_summary is None:
        scenarios.append(
            {
                "scenario_id": 0,
                "scenario_type": "manual",
                "beta_I": args.beta_I,
                "X0": args.x0,
                "I0": args.i0,
            }
        )
        return scenarios
    
    df = pd.read_csv(args.baseline_summary)

    required = {"beta_I_grid", "X0", "I0_grid", "p_high"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Baseline summary missing required columns: {sorted(missing)}")

    bistable = df[(df["p_high"] >= 0.2) & (df["p_high"] <= 0.8)].copy()

    if bistable.empty:
        raise ValueError(
            "No bistable scenarios found. "
            "Re-run baseline sweeps with lower ratio / beta_I."
        )

    if args.pick_bistable > len(bistable):
        raise ValueError(
            f"Requested {args.pick_bistable} bistable scenarios, "
            f"but only {len(bistable)} available."
        )
    
    if "X_std" in bistable.columns:
        bistable = bistable.sort_values("X_std", ascending=False)
    else:
        bistable = bistable.assign(
            p_distance=(bistable["p_high"] - 0.5).abs()
        ).sort_values("p_distance")

    bistable = bistable.head(args.pick_bistable)

    for sid, (_, row) in enumerate(bistable.iterrows()):
        scenarios.append(
            {
                "scenario_id": sid,
                "scenario_type": "bistable",
                "baseline_cell_id": int(row.name),
                "beta_I": float(row["beta_I_grid"]),
                "X0": float(row["X0"]),
                "I0": float(row["I0_grid"]),
            }
        )

    next_id = len(scenarios)

    if args.pick_hard > 0:
        hard = df[df["p_high"] < 0.2].copy()

        if hard.empty:
            print("WARNING: No hard regimes found; skipping hard controls.")
        else:
            hard = hard.sort_values("p_high").head(args.pick_hard)

            for _, row in hard.iterrows():
                scenarios.append(
                    {
                        "scenario_id": next_id,
                        "scenario_type": "hard",
                        "baseline_cell_id": int(row.name),
                        "beta_I": float(row["beta_I_grid"]),
                        "X0": float(row["X0"]),
                        "I0": float(row["I0_grid"]),
                    }
                )
                next_id += 1

    print(
        f"Loaded {len(scenarios)} scenarios "
        f"({sum(s['scenario_type']=='bistable' for s in scenarios)} bistable, "
        f"{sum(s['scenario_type']=='hard' for s in scenarios)} hard)"
    )

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
    parser.add_argument("--ratio", type=float, default=1.4)
    parser.add_argument("--beta-I", dest="beta_I", type=float, default=1.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--g-i", type=float, default=0.08)
    parser.add_argument("--i0", type=float, default=0.03)
    parser.add_argument("--x0", type=float, default=0.08)
    parser.add_argument("--init-method", type=str, default="random")
    parser.add_argument("--strategy", type=str, default="imitate")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--collect-series", action="store_true", help="Store full trajectories.")
    parser.add_argument("--runs-output", type=str, default="results/policy_runs.csv")
    parser.add_argument("--series-output", type=str, default="results/policy_timeseries.csv")
    parser.add_argument("--baseline-summary", type=str, default=None)
    parser.add_argument("--pick-bistable", type=int, default=0)
    parser.add_argument("--seed-fracs", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--seed-starts", type=int, nargs="+", default=[0, 25, 50])
    parser.add_argument("--seed-durations", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--seed-metric", type=str, default="degree")
    parser.add_argument("--seed-lock", action="store_true")
    parser.add_argument("--subsidy-delta-a0", type=float, nargs="+", default=[0.1, 0.3, 0.6])
    parser.add_argument("--subsidy-starts", type=int, nargs="+", default=[0, 10, 25])
    parser.add_argument("--subsidy-durations", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--infra-boosts", type=float, nargs="+", default=[0.05, 0.15, 0.30])
    parser.add_argument("--infra-starts", type=int, nargs="+", default=[0, 10, 25])
    parser.add_argument("--pick-hard", type=int, default=0, help="Pick hard regimes from baseline summary.")

    args = parser.parse_args()

    scenarios = _load_scenarios(args)
    policies = _policies(args)
    run_rows: List[Dict] = []
    series_rows: List[Dict] = []

    run_id = 0
    for scenario in scenarios:
        for net_type in args.network_types:
            network_kwargs = _kwargs_for_type(net_type, args)
            for label, policy, meta in policies:
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
                    row["policy_label"] = label
                    row["policy_type"] = meta.get("ptype")
                    row["policy_cost"] = _policy_cost(meta, args.n_nodes)
                    for k in ("metric", "fraction", "start", "duration", "delta_a0", "boost", "lock"):
                        if k in meta:
                            row[f"policy_{k}"] = meta[k]
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
            cost_mean=("policy_cost", "mean"),
        )
        .reset_index()
    )
    summary["p_high_per_cost"] = summary["p_high"] / summary["cost_mean"].replace(0, pd.NA)
    summary["ineffective"] = summary["p_high"] < 0.2
    summary["inefficient"] = summary["p_high_per_cost"] < summary["p_high_per_cost"].median()


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
