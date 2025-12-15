"""
Reusable simulation helpers for Assignment 3 experiments.

This module keeps new experiment code separate from the provided
`ev_core.py`. It adds:
- a network-flexible model wrapper that accepts pre-built graphs,
- a single `run_simulation` function that returns tidy results,
- policy utilities for targeted/timed interventions.
"""

from __future__ import annotations

import sys
from pathlib import Path

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from mesa import Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector

from ev_core import EVAgent
from ev_experiments import policy_subsidy_factory, policy_infrastructure_boost_factory
from experiments.networks import build_network, rank_nodes_by_centrality, select_top_fraction


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))

def _adopter_cluster_stats(G: nx.Graph, adopter_nodes: List[int]) -> Dict[str, float]:
    """Connected-component statistics on the adopter-induced subgraph."""
    if not adopter_nodes:
        return {"clusters_n": 0, "cluster_max": 0, "cluster_mean": 0.0}

    H = G.subgraph(adopter_nodes)
    comps = list(nx.connected_components(H))
    sizes = [len(c) for c in comps]
    return {
        "clusters_n": int(len(sizes)),
        "cluster_max": int(max(sizes)) if sizes else 0,
        "cluster_mean": float(sum(sizes) / len(sizes)) if sizes else 0.0,
    }

@dataclass
class SimulationConfig:
    """Minimal configuration for a single simulation run."""

    T: int = 200
    n_nodes: int = 200
    network_type: str = "erdos_renyi"
    network_kwargs: Dict = field(default_factory=dict)
    a0: float = 2.0
    ratio: Optional[float] = None  # if provided, overrides a0 via ratio*b - beta_I*I0
    beta_I: float = 2.0
    b: float = 1.0
    g_I: float = 0.05
    I0: float = 0.05
    X0_frac: float = 0.10
    init_method: str = "random"  # random | degree_high | degree_low | centrality
    init_centrality: str = "degree"
    strategy_choice_func: str = "imitate"
    tau: float = 1.0
    collect_series: bool = False


class EVStagHuntNetworkModel(Model):
    """Variant of the EV model that accepts a pre-built graph."""

    def __init__(
        self,
        graph: nx.Graph,
        *,
        a0: float,
        beta_I: float,
        b: float,
        g_I: float,
        I0: float,
        collect: bool = False,
        strategy_choice_func: str = "imitate",
        tau: float = 1.0,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed)
        self.G = graph.copy()
        self.grid = NetworkGrid(self.G)
        self.schedule = SimultaneousActivation(self)

        self.a0 = a0
        self.beta_I = beta_I
        self.b = b
        self.g_I = g_I
        self.infrastructure = I0
        self.step_count = 0
        self.strategy_choice_func = strategy_choice_func
        self.tau = tau

        for n in self.G.nodes:
            self.G.nodes[n]["agent"] = []

        # create agents (default all D)
        uid = 0
        for node in self.G.nodes:
            agent = EVAgent(uid, self, init_strategy="D")
            uid += 1
            self.schedule.add(agent)
            self.grid.place_agent(agent, node)

        self.datacollector = None
        if collect:
            self.datacollector = DataCollector(
                model_reporters={
                    "X": self.get_adoption_fraction,
                    "I": lambda m: m.infrastructure,
                },
                agent_reporters={"strategy": "strategy", "payoff": "payoff"},
            )

    def get_adoption_fraction(self) -> float:
        agents = self.schedule.agents
        if not agents:
            return 0.0
        return float(sum(1 for a in agents if a.strategy == "C") / len(agents))

    def step(self):
        self.schedule.step()
        X = self.get_adoption_fraction()
        I = self.infrastructure
        dI = self.g_I * (X - I)
        self.infrastructure = _clip01(I + dI)
        if self.datacollector is not None:
            self.datacollector.collect(self)
        self.step_count += 1


def _agents_on_nodes(model: EVStagHuntNetworkModel, nodes: List[int]) -> List:
    agents: List = []
    for node in nodes:
        agents.extend(model.grid.get_cell_list_contents([node]))
    return agents


def apply_initial_adopters(
    model: EVStagHuntNetworkModel,
    X0_frac: float,
    *,
    method: str = "random",
    seed: Optional[int] = None,
    centrality_metric: str = "degree",
) -> List[int]:
    """Set initial EV adopters on the model and return node ids that were seeded."""
    agents = model.schedule.agents
    for a in agents:
        a.strategy = "D"
        a.next_strategy = "D"

    n = len(agents)
    k = int(round(X0_frac * n))
    if k <= 0:
        return []

    if method == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=k, replace=False)
        seeded_nodes = [agents[i].pos for i in idx]
    elif method in {"degree_high", "degree_low"}:
        deg = dict(model.G.degree())
        ordered = sorted(deg.keys(), key=lambda u: deg[u], reverse=(method == "degree_high"))
        seeded_nodes = ordered[:k]
    elif method == "centrality":
        ordered = rank_nodes_by_centrality(model.G, metric=centrality_metric)
        seeded_nodes = ordered[:k]
    else:
        raise ValueError(f"Unknown init_method '{method}'")

    for agent in _agents_on_nodes(model, seeded_nodes):
        agent.strategy = "C"
        agent.next_strategy = "C"
    return seeded_nodes


# -----------------------------
# Policy helpers
# -----------------------------

def policy_targeted_seeding(
    target_nodes: List[int],
    *,
    start: int = 0,
    duration: Optional[int] = 1,
    lock: bool = True,
    infrastructure_boost: float = 0.0,
) -> Callable:
    """Force target nodes to adopt EV during a time window; optional infra boost."""
    end = None if duration is None else start + duration

    def policy(model, step):
        active = step >= start and (end is None or step < end)
        if not active:
            return
        if lock or step == start:
            for agent in _agents_on_nodes(model, target_nodes):
                agent.strategy = "C"
                agent.next_strategy = "C"
        if infrastructure_boost > 0.0 and step == start:
            model.infrastructure = _clip01(model.infrastructure + infrastructure_boost)

    return policy


def build_policy(policy_config: Optional[Dict], graph: nx.Graph) -> Tuple[Optional[Callable], str]:
    """Create a policy callable from a config dict."""
    if not policy_config:
        return None, "none"

    ptype = policy_config.get("type")
    if ptype == "subsidy":
        params = policy_config.get("params", {})
        return policy_subsidy_factory(**params), f"subsidy:{params}"
    if ptype == "infrastructure":
        params = policy_config.get("params", {})
        return policy_infrastructure_boost_factory(**params), f"infra:{params}"
    if ptype == "targeted_seed":
        params = policy_config.get("params", {})
        metric = params.get("metric", "degree")
        frac = params.get("fraction", 0.05)
        k = params.get("k", None)
        ordered = rank_nodes_by_centrality(graph, metric=metric)
        target_nodes = select_top_fraction(ordered, frac=frac, k=k)
        policy = policy_targeted_seeding(
            target_nodes,
            start=params.get("start", 0),
            duration=params.get("duration", 5),
            lock=params.get("lock", True),
            infrastructure_boost=params.get("infrastructure_boost", 0.0),
        )
        label = f"targeted_seed:{metric}:{len(target_nodes)}"
        return policy, label

    raise ValueError(f"Unsupported policy type '{ptype}'")


def run_simulation(
    config: SimulationConfig,
    *,
    seed: Optional[int] = None,
    policy_config: Optional[Dict] = None,
    run_id: Optional[int] = None,
    convergence_tol: float = 1e-3,
    patience: int = 30,
    record_series: bool | None = None,
) -> Tuple[Dict, Optional[List[Dict[str, float]]]]:
    """Run one simulation and return (summary_row, timeseries_rows?)."""
    record_series = config.collect_series if record_series is None else record_series
    network_kwargs = config.network_kwargs or {}

    G = build_network(config.network_type, config.n_nodes, seed=seed, **network_kwargs)
    a0 = config.a0
    if config.ratio is not None:
        a0 = float(config.ratio) * float(config.b) - float(config.beta_I) * float(config.I0)

    model = EVStagHuntNetworkModel(
        G,
        a0=a0,
        beta_I=config.beta_I,
        b=config.b,
        g_I=config.g_I,
        I0=config.I0,
        collect=record_series,
        strategy_choice_func=config.strategy_choice_func,
        tau=config.tau,
        seed=seed,
    )
    seeded_nodes = apply_initial_adopters(
        model,
        config.X0_frac,
        method=config.init_method,
        seed=seed,
        centrality_metric=config.init_centrality,
    )

    policy, policy_label = build_policy(policy_config, G)

    timeseries: List[Dict[str, float]] = []
    stable_steps = 0
    prev_X = None
    prev_I = None
    converged = False
    steps_run = 0
    t_high: Optional[int] = None
    t_half: Optional[int] = None

    for t in range(config.T):
        if policy is not None:
            policy(model, t)
        model.step()
        X = model.get_adoption_fraction()
        I = model.infrastructure

        if t_half is None and X >= 0.5:
            t_half = t
        if t_high is None and X >= 0.8:
            t_high = t

        if record_series:
            timeseries.append({"run_id": run_id, "time": t, "X": X, "I": I, "policy": policy_label})

        if prev_X is not None and prev_I is not None:
            if abs(X - prev_X) < convergence_tol and abs(I - prev_I) < convergence_tol:
                stable_steps += 1
            else:
                stable_steps = 0
        prev_X, prev_I = X, I
        steps_run = t + 1

        if stable_steps >= patience or X in (0.0, 1.0):
            converged = True
            break

    X_final = float(model.get_adoption_fraction())
    
    # cluster stats at final time
    adopter_nodes = [a.pos for a in model.schedule.agents if a.strategy == "C"]
    cstats = _adopter_cluster_stats(G, adopter_nodes)

    result_row = {
        "run_id": run_id,
        "seed": seed,
        "network_type": config.network_type,
        "network_kwargs": json.dumps(network_kwargs, sort_keys=True),
        "n_nodes": config.n_nodes,
        "a0": a0,
        "ratio": config.ratio,
        "beta_I": config.beta_I,
        "b": config.b,
        "g_I": config.g_I,
        "I0": config.I0,
        "X0_frac": config.X0_frac,
        "init_method": config.init_method,
        "strategy_choice_func": config.strategy_choice_func,
        "tau": config.tau,
        "policy": policy_label,
        "seeded_nodes": len(seeded_nodes),
        "steps_run": steps_run,
        "converged": converged,
        "X_final": X_final,
        "I_final": float(model.infrastructure),
        "high_adoption": int(X_final > 0.8),
        "t_half": t_half,
        "t_high": t_high,
        "clusters_n_final": cstats["clusters_n"],
        "cluster_max_final": cstats["cluster_max"],
        "cluster_mean_final": cstats["cluster_mean"],
    }

    return result_row, timeseries if record_series else None
