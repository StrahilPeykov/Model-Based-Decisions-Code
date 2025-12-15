import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(
    style="whitegrid",
    context="paper",
    font_scale=1.2,
)

Path("figures").mkdir(exist_ok=True)

base = pd.read_csv("results/baseline_sweep_summary.csv")

heat = base.pivot_table(
    index="X0",
    columns="I0_grid",
    values="p_high"
)

plt.figure(figsize=(6, 5))
ax = sns.heatmap(
    heat,
    cmap="viridis",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "P(high EV adoption)"},
)

ax.set_xlabel("Initial infrastructure $I_0$")
ax.set_ylabel("Initial EV adoption $X_0$")
ax.set_title("Baseline bistability and tipping regimes")

plt.tight_layout()
plt.savefig("figures/fig1_baseline_bistability.png", dpi=300)
plt.close()

net = pd.read_csv("results/network_comparison_summary.csv")

plt.figure(figsize=(5.5, 4))
ax = sns.barplot(
    data=net,
    x="network_label",
    y="p_high",
    errorbar="sd",
)

ax.set_xlabel("Network topology")
ax.set_ylabel("P(high EV adoption)")
ax.set_title("Effect of network structure on EV diffusion")

plt.tight_layout()
plt.savefig("figures/fig2_network_effects.png", dpi=300)
plt.close()

pol = pd.read_csv("results/policy_runs_summary.csv")

plot_data = pol.copy()

plt.figure(figsize=(7, 4))
ax = sns.barplot(
    data=plot_data,
    x="policy_label",
    y="p_high",
)

ax.axhline(
    y=pol.loc[pol["policy_label"] == "baseline", "p_high"].mean(),
    color="black",
    linestyle="--",
    label="Baseline",
)

ax.set_ylabel("P(high EV adoption)")
ax.set_xlabel("Policy intervention")
ax.set_title("Effectiveness of targeted and timed interventions")
ax.legend()

plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("figures/fig3_policy_effectiveness.png", dpi=300)
plt.close()


pol_eff = pol[pol["policy_label"] != "baseline"].copy()

plt.figure(figsize=(5.5, 4))
ax = sns.scatterplot(
    data=pol_eff,
    x="cost_mean",
    y="p_high",
    hue="policy_type",
    s=70,
)

ax.set_xlabel("Policy cost (proxy)")
ax.set_ylabel("P(high EV adoption)")
ax.set_title("Policy cost–effectiveness frontier")

plt.tight_layout()
plt.savefig("figures/fig4_cost_effectiveness.png", dpi=300)
plt.close()


fail = pol_eff[pol_eff["ineffective"]]

plt.figure(figsize=(4.5, 4))
ax = sns.countplot(
    data=fail,
    x="policy_type",
)

ax.set_ylabel("Number of ineffective cases")
ax.set_xlabel("Policy type")
ax.set_title("Regimes where policy fails")

plt.tight_layout()
plt.savefig("figures/fig5_policy_failures.png", dpi=300)
plt.close()



