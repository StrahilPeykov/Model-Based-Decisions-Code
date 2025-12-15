"""
Generate plots from experiment results.

Produces figures:
- Figure 1: Baseline phase diagram
- Figure 2: Network structure effects
- Figure 3: Policy effectiveness & efficiency
- Figure 4: Policy timing & cost
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Paths
# ----------------------------
RESULTS = Path("results")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True, parents=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# ----------------------------
# Load data (EXACT files you produce)
# ----------------------------
baseline_summary = pd.read_csv(RESULTS / "baseline_sweep_summary.csv")
baseline_raw = pd.read_csv(RESULTS / "baseline_sweep.csv")

network_runs = pd.read_csv(RESULTS / "network_comparison.csv")
network_summary = pd.read_csv(RESULTS / "network_comparison_summary.csv")

policy_runs = pd.read_csv(RESULTS / "policy_runs.csv")
policy_summary = pd.read_csv(RESULTS / "policy_runs_summary.csv")

# ----------------------------
# FIGURE 1 — Baseline phase diagram
# ----------------------------
beta_I = 2.0
df = baseline_summary[baseline_summary["beta_I_grid"] == beta_I]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)

# A: p_high
piv = df.pivot(index="X0", columns="I0_grid", values="p_high")
im = axes[0, 0].imshow(piv.values, origin="lower", vmin=0, vmax=1, cmap="viridis")
axes[0, 0].set_title("P(high adoption)")
axes[0, 0].set_xlabel("I₀")
axes[0, 0].set_ylabel("X₀")
plt.colorbar(im, ax=axes[0, 0])

# B: bistable
if "bistable" in df.columns:
    piv = df.pivot(index="X0", columns="I0_grid", values="bistable")
    im = axes[0, 1].imshow(piv.values, origin="lower", cmap="binary")
    axes[0, 1].set_title("Bistable region")
    plt.colorbar(im, ax=axes[0, 1])

# C: time to tipping
if "t_high_mean" in df.columns:
    piv = df.pivot(index="X0", columns="I0_grid", values="t_high_mean")
    im = axes[1, 0].imshow(piv.values, origin="lower", cmap="plasma")
    axes[1, 0].set_title("Mean time to X ≥ 0.8")
    plt.colorbar(im, ax=axes[1, 0])

# D: variability
if "X_std" in df.columns:
    piv = df.pivot(index="X0", columns="I0_grid", values="X_std")
    im = axes[1, 1].imshow(piv.values, origin="lower", cmap="inferno")
    axes[1, 1].set_title("Std. dev. of final adoption")
    plt.colorbar(im, ax=axes[1, 1])

fig.savefig(FIGDIR / "fig1_baseline_phase.png", dpi=300)
plt.close(fig)

# ----------------------------
# FIGURE 2 — Network structure effects
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

# A: p_high
sns.barplot(
    data=network_summary,
    x="network_label",
    y="p_high",
    ax=axes[0],
)
axes[0].set_ylim(0, 1)
axes[0].set_title("A: Adoption probability")

# B: speed
if "t_high_mean" in network_summary.columns:
    sns.barplot(
        data=network_summary,
        x="network_label",
        y="t_high_mean",
        ax=axes[1],
    )
    axes[1].set_title("B: Time to tipping")

# C: clustering
sns.barplot(
    data=network_summary,
    x="network_label",
    y="cluster_max_mean",
    ax=axes[2],
)
axes[2].set_title("C: Cluster size")

fig.savefig(FIGDIR / "fig2_network_effects.png", dpi=300)
plt.close(fig)

# ----------------------------
# FIGURE 3 — Policy effectiveness & efficiency
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# A: effectiveness
sns.barplot(
    data=policy_summary,
    x="policy_label",
    y="p_high",
    hue="network_label",
    ax=axes[0],
)
axes[0].set_ylim(0, 1)
axes[0].set_title("Policy effectiveness")

# B: efficiency
if "p_high_per_cost" in policy_summary.columns:
    sns.barplot(
        data=policy_summary,
        x="policy_label",
        y="p_high_per_cost",
        hue="network_label",
        ax=axes[1],
    )
    axes[1].set_title("Policy efficiency")

for ax in axes:
    ax.tick_params(axis="x", rotation=30)

fig.savefig(FIGDIR / "fig3_policy_effectiveness.png", dpi=300)
plt.close(fig)

# ----------------------------
# FIGURE 4 — Policy timing & cost
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

# Timing
if "policy_start" in policy_runs.columns:
    sns.boxplot(
        data=policy_runs,
        x="policy_type",
        y="policy_start",
        ax=axes[0],
    )
    axes[0].set_title("Intervention timing")

# Cost
sns.boxplot(
    data=policy_runs,
    x="policy_type",
    y="policy_cost",
    ax=axes[1],
)
axes[1].set_title("Policy cost (proxy)")

fig.savefig(FIGDIR / "fig4_policy_cost_timing.png", dpi=300)
plt.close(fig)

print("All figures generated successfully.")
