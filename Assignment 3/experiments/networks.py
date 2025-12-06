"""
Network builders and helpers for Assignment 3 experiments.

Provides a small registry of graph generators so experiment scripts can
request network types by name (e.g. "erdos_renyi", "barabasi_albert",
"small_world", "grid"). 
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence

import networkx as nx


def _grid_graph(n_nodes: int, seed: int | None = None) -> nx.Graph:
    """Build a roughly square 2D grid and relabel nodes to 0..n-1.

    Seed is accepted for API symmetry but ignored (grid is deterministic).
    """
    side = max(1, int(round(math.sqrt(n_nodes))))
    G = nx.grid_2d_graph(side, side, periodic=False)
    # Trim extra nodes if the grid is larger than requested
    nodes = list(G.nodes)[:n_nodes]
    G = G.subgraph(nodes).copy()
    mapping = {node: i for i, node in enumerate(G.nodes)}
    return nx.relabel_nodes(G, mapping)


def _small_world_graph(n_nodes: int, k: int = 6, rewiring_p: float = 0.05, seed: int | None = None) -> nx.Graph:
    """Watts–Strogatz small-world network."""
    k = max(2, min(k, n_nodes - 1))
    return nx.watts_strogatz_graph(n_nodes, k, rewiring_p, seed=seed)


def _erdos_renyi_graph(n_nodes: int, p: float = 0.05, seed: int | None = None) -> nx.Graph:
    """Erdős–Rényi random graph."""
    return nx.erdos_renyi_graph(n_nodes, p, seed=seed)


def _barabasi_albert_graph(n_nodes: int, m: int = 2, seed: int | None = None) -> nx.Graph:
    """Barabási–Albert scale-free graph."""
    m = max(1, min(m, n_nodes - 1))
    return nx.barabasi_albert_graph(n_nodes, m, seed=seed)


NETWORK_BUILDERS: Dict[str, Callable[..., nx.Graph]] = {
    "grid": _grid_graph,
    "small_world": _small_world_graph,
    "erdos_renyi": _erdos_renyi_graph,
    "barabasi_albert": _barabasi_albert_graph,
}


def build_network(network_type: str, n_nodes: int, seed: int | None = None, **kwargs) -> nx.Graph:
    """Create a networkx graph for the requested type.

    Parameters
    ----------
    network_type : str
        One of the keys in NETWORK_BUILDERS.
    n_nodes : int
        Number of nodes to generate (rounded up for grids).
    seed : int | None
        Optional RNG seed for stochastic generators.
    kwargs :
        Extra parameters forwarded to the builder (e.g., p for ER, m for BA).
    """
    network_type = network_type.lower()
    if network_type not in NETWORK_BUILDERS:
        supported = ", ".join(sorted(NETWORK_BUILDERS.keys()))
        raise ValueError(f"Unknown network_type '{network_type}'. Supported: {supported}")

    builder = NETWORK_BUILDERS[network_type]
    return builder(n_nodes, seed=seed, **kwargs)


def rank_nodes_by_centrality(G: nx.Graph, metric: str = "degree") -> List[int]:
    """Return node ids sorted from most to least central by the chosen metric."""
    metric = metric.lower()
    if metric == "degree":
        scores = dict(G.degree())
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif metric == "eigenvector":
        try:
            scores = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            scores = nx.betweenness_centrality(G)
    else:
        raise ValueError(f"Unsupported centrality metric '{metric}'")

    return [node for node, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def select_top_fraction(nodes_ordered: Sequence[int], frac: float | None = None, k: int | None = None) -> List[int]:
    """Select the top-k or top-frac nodes from an ordered list."""
    if frac is None and k is None:
        raise ValueError("Provide either frac or k to select nodes.")
    if frac is not None:
        k = max(1, int(round(frac * len(nodes_ordered))))
    assert k is not None
    k = min(k, len(nodes_ordered))
    return list(nodes_ordered[:k])
