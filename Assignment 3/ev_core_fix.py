"""
Core EV Stag Hunt model components (FIXED VERSION).

This file is a minimal, theory-consistent correction of the original
`ev_core.py`. All changes are explicitly marked and justified.

KEY FIX:
-------------------
The original implementation computed payoffs only for the agent’s
current strategy. This makes logit dynamics ill-defined, because
logit choice requires comparison of payoff(C) vs payoff(D).

As a result, EV adoption was mathematically impossible.

This file fixes that by:
1. Computing both payoff_C and payoff_D at each step
2. Applying imitation and logit choice using those payoffs
3. Preserving all original model semantics, parameters, and structure

No experiment-specific logic is included here.
"""

from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import random
from typing import Iterable, List, Dict


# ============================================================
# Strategy selection helpers
# ============================================================

def choose_strategy_imitate(agent, neighbors):
    """
    Imitation rule:
    Choose the strategy of the highest-payoff agent among self + neighbors.
    """
    candidates = neighbors + [agent]
    best = max(candidates, key=lambda a: a.payoff)
    return best.strategy


# ============================================================
# Agent class
# ============================================================

class EVAgent(Agent):
    """
    Single agent at a graph node.

    Attributes
    ----------
    strategy : str
        Current strategy ("C" for EV, "D" for ICE)
    payoff : float
        Payoff of the CURRENT strategy (used for imitation)
    payoff_C : float
        Counterfactual payoff if agent were to play C
    payoff_D : float
        Counterfactual payoff if agent were to play D
    next_strategy : str
        Strategy chosen for the next step (SimultaneousActivation)
    """

    def __init__(self, unique_id, model, init_strategy="D"):
        super().__init__(unique_id, model)
        self.strategy = init_strategy
        self.payoff = 0.0
        self.payoff_C = 0.0
        self.payoff_D = 0.0
        self.next_strategy = init_strategy

    # --------------------------------------------------------
    # FIX 1: Compute BOTH payoff_C and payoff_D
    # --------------------------------------------------------
    def step(self):
        """
        Compute counterfactual payoffs for BOTH strategies.

        This is the critical fix. Logit dynamics REQUIRE access to
        payoff_C - payoff_D. The original code did not compute this.
        """
        I = self.model.infrastructure
        a0 = self.model.a0
        beta_I = self.model.beta_I
        b = self.model.b
        a_I = a0 + beta_I * I

        neighbors = []
        for nbr in self.model.G.neighbors(self.pos):
            neighbors.extend(self.model.grid.get_cell_list_contents([nbr]))

        if not neighbors:
            self.payoff_C = 0.0
            self.payoff_D = 0.0
            self.payoff = 0.0
            return

        payoff_C = 0.0
        payoff_D = 0.0

        for other in neighbors:
            if other.strategy == "C":
                payoff_C += a_I      # C vs C
                payoff_D += b        # D vs C
            else:
                payoff_C += 0.0      # C vs D
                payoff_D += b        # D vs D

        self.payoff_C = payoff_C
        self.payoff_D = payoff_D

        # payoff used for imitation must correspond to CURRENT strategy
        self.payoff = payoff_C if self.strategy == "C" else payoff_D

    # --------------------------------------------------------
    # FIX 2: Correct logit implementation
    # --------------------------------------------------------
    def advance(self, strategy_choice_func=None):
        """
        Update strategy using imitation or logit rule.

        - Imitation uses realized payoff (as before)
        - Logit uses payoff_C vs payoff_D (FIXED)
        """
        func = (
            strategy_choice_func
            if strategy_choice_func is not None
            else getattr(self.model, "strategy_choice_func", "imitate")
        )

        neighbors = []
        for nbr in self.model.G.neighbors(self.pos):
            neighbors.extend(self.model.grid.get_cell_list_contents([nbr]))

        if func == "imitate":
            self.next_strategy = choose_strategy_imitate(self, neighbors)

        elif func == "logit":
            tau = max(getattr(self.model, "tau", 1e-6), 1e-6)
            delta = self.payoff_C - self.payoff_D
            p_C = 1.0 / (1.0 + np.exp(-delta / tau))
            self.next_strategy = "C" if random.random() < p_C else "D"

        else:
            raise ValueError(f"Unknown strategy choice function: {func}")

        self.strategy = self.next_strategy


# ============================================================
# Model class (UNCHANGED except comments)
# ============================================================

class EVStagHuntModel(Model):
    """
    Mesa model for EV Stag Hunt on a network.
    """

    def __init__(
        self,
        initial_ev=10,
        a0=2.0,
        beta_I=3.0,
        b=1.0,
        g_I=0.1,
        I0=0.05,
        seed=None,
        network_type="random",
        n_nodes=100,
        p=0.05,
        m=2,
        collect=True,
        strategy_choice_func="imitate",
        tau=1.0,
    ):
        super().__init__(seed=seed)

        if network_type == "BA":
            G = nx.barabasi_albert_graph(n_nodes, m, seed=seed)
        else:
            G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)

        self.G = G
        self.grid = NetworkGrid(G)
        self.schedule = SimultaneousActivation(self)

        self.a0 = a0
        self.beta_I = beta_I
        self.b = b
        self.g_I = g_I
        self.infrastructure = I0
        self.strategy_choice_func = strategy_choice_func
        self.tau = tau
        self.step_count = 0

        for n in self.G.nodes:
            self.G.nodes[n]["agent"] = []

        total_nodes = self.G.number_of_nodes()
        k_ev = max(0, min(initial_ev, total_nodes))
        ev_nodes = set(self.random.sample(list(self.G.nodes), k_ev))

        uid = 0
        for node in self.G.nodes:
            init_strategy = "C" if node in ev_nodes else "D"
            agent = EVAgent(uid, self, init_strategy)
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

    def get_adoption_fraction(self):
        agents = self.schedule.agents
        return sum(a.strategy == "C" for a in agents) / len(agents)

    def step(self):
        self.schedule.step()
        X = self.get_adoption_fraction()
        I = self.infrastructure
        dI = self.g_I * (X - I)
        self.infrastructure = float(min(1.0, max(0.0, I + dI)))
        if self.datacollector:
            self.datacollector.collect(self)
        self.step_count += 1
