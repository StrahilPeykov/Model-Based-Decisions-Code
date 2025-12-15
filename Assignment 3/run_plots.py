"""
Plotting!

Produces:
- Fig 1: Baseline tipping + bistability evidence (heatmaps + tipping curve + bimodality)
- Fig 2: Network structure effects (p_high, t_high, clustering)
- Fig 3: Policy effect size (Δp_high) across networks
- Fig 4: Policy efficiency + timing sensitivity

Usage:
  python run_plots_rubric.py --beta 2.0 --ratio 2.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_bistable(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """
    Prefer rubric definition: bistable if both low and high basins occur.
    If p_low missing, fall back to "mixed outcomes" p_high in (alpha, 1-alpha).
    """
    df = df.copy()
    if "bistable" in df.columns and ("p_low" in df.columns or "p_high" in df.columns):
        pass

    if "p_low" in df.columns and "p_high" in df.columns:
        df["bistable_rubric"] = ((df["p_high"] >= alpha) & (df["p_low"] >= alpha)).astype(int)
    elif "p_high" in df.columns:
        df["bistable_rubric"] = ((df["p_high"] >= alpha) & (df["p_high"] <= 1 - alpha)).astype(int)
    else:
        df["bistable_rubric"] = 0
    return df


def _heatmap(ax, piv: pd.DataFrame, title: str, cbar_label: str, vmin=None, vmax=None, cmap=None):
    x = np.array(piv.columns, dtype=float)
    y = np.array(piv.index, dtype=float)
    img = ax.imshow(
        piv.values,
        origin="lower",
        aspect="auto",
        extent=[x.min(), x.max(), y.min(), y.max()],
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    ax.set_xlabel("Initial adoption X₀")
    ax.set_ylabel("Initial infrastructure I₀")
    cb = plt.colorbar(img, ax=ax)
    cb.set_label(cbar_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--figdir", type=str, default="figures")
    parser.add_argument("--beta", type=float, default=2.0, help="Which beta_I_grid slice to plot.")
    parser.add_argument("--ratio", type=float, default=2.2 | None, help="Used to draw theory line X0=1/ratio on tipping plot.")
    parser.add_argument("--baseline-summary", type=str, default="baseline_sweep_summary.csv")
    parser.add_argument("--baseline-raw", type=str, default="baseline_sweep.csv")
    parser.add_argument("--network-summary", type=str, default="network_comparison_summary.csv")
    parser.add_argument("--policy-summary", type=str, default="policy_runs_summary.csv")
    args = parser.parse_args()

    results = Path(args.results)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load
    # ----------------------------
    base_sum = pd.read_csv(results / args.baseline_summary)
    base_raw = pd.read_csv(results / args.baseline_raw)

    base_sum = _ensure_bistable(base_sum)

    if "beta_I_grid" in base_sum.columns:
        if args.beta not in set(base_sum["beta_I_grid"].unique()):
            # fallback: pick closest available beta
            betas = np.sort(base_sum["beta_I_grid"].unique())
            beta = float(betas[np.argmin(np.abs(betas - args.beta))])
        else:
            beta = args.beta
        dfb = base_sum[base_sum["beta_I_grid"] == beta].copy()
    else:
        beta = args.beta
        dfb = base_sum.copy()

    # ----------------------------
    # fig 1 — Baseline: tipping + bistability evidence
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    # A: p_high heatmap
    piv_ph = dfb.pivot(index="I0_grid", columns="X0", values="p_high").sort_index()
    _heatmap(
        axes[0, 0],
        piv_ph,
        title=f"A. P(high adoption) at β_I={beta:g}",
        cbar_label="p_high",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )

    # B: bistable heatmap (rubric definition)
    piv_bi = dfb.pivot(index="I0_grid", columns="X0", values="bistable_rubric").sort_index()
    _heatmap(
        axes[0, 1],
        piv_bi,
        title="B. Bistable region (two basins observed)",
        cbar_label="bistable (0/1)",
        vmin=0,
        vmax=1,
        cmap="binary",
    )

    # C: tipping curve: mean p_high by X0 (averaged over I0)
    curve = dfb.groupby("X0", as_index=False)["p_high"].mean()
    axes[1, 0].plot(curve["X0"], curve["p_high"])
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("C. Tipping curve: mean p_high vs X₀ (avg over I₀)")
    axes[1, 0].set_xlabel("X₀")
    axes[1, 0].set_ylabel("mean p_high")

    if args.ratio is not None and args.ratio > 0:
        xcrit = 1.0 / args.ratio
        axes[1, 0].axvline(xcrit, linestyle="--")
        axes[1, 0].text(xcrit, 0.05, f" 1/ratio={xcrit:.2f}", rotation=90, va="bottom")

    # D: show bimodality in the bistable band (final adoption distribution)
    # Take raw runs whose (X0, I0, beta) fall in bistable cells
    bi_cells = dfb[dfb["bistable_rubric"] == 1][["X0", "I0_grid"]].drop_duplicates()
    if not bi_cells.empty:
        raw = base_raw.copy()
        # keep same beta slice if present
        if "beta_I_grid" in raw.columns and "beta_I_grid" in dfb.columns:
            raw = raw[raw["beta_I_grid"] == beta]

        # inner-join on (X0, I0_grid)
        raw_bi = raw.merge(bi_cells, on=["X0", "I0_grid"], how="inner")

        if "X_final" in raw_bi.columns and len(raw_bi) > 0:
            axes[1, 1].hist(raw_bi["X_final"].dropna().to_numpy(), bins=20, range=(0, 1))
            axes[1, 1].set_title("D. Final adoption X* in bistable cells (bimodality)")
            axes[1, 1].set_xlabel("X*")
            axes[1, 1].set_ylabel("count")
        else:
            axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    fig.savefig(figdir / "fig1_baseline_rubric.png", dpi=300)
    plt.close(fig)

    # ----------------------------
    # fig 2 — Network structure effects
    # ----------------------------
    net_path = results / args.network_summary
    if net_path.exists():
        net = pd.read_csv(net_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # A: p_high
        if "p_high" in net.columns:
            x = net["network_label"]
            y = net["p_high"]
            axes[0].bar(x, y)
            axes[0].set_ylim(0, 1)
            axes[0].set_title("A. Adoption probability (p_high)")
            axes[0].set_xlabel("Network")
            axes[0].set_ylabel("p_high")

        # B: speed
        if "t_high_mean" in net.columns:
            axes[1].bar(net["network_label"], net["t_high_mean"])
            axes[1].set_title("B. Time to tipping (mean t_high)")
            axes[1].set_xlabel("Network")
            axes[1].set_ylabel("t_high")

        # C: clustering / largest cluster
        if "cluster_max_mean" in net.columns:
            axes[2].bar(net["network_label"], net["cluster_max_mean"])
            axes[2].set_title("C. Largest adopter cluster (mean)")
            axes[2].set_xlabel("Network")
            axes[2].set_ylabel("cluster_max")

        fig.savefig(figdir / "fig2_network_effects_rubric.png", dpi=300)
        plt.close(fig)

    # ----------------------------
    # fig 3–4 — Policy
    # ----------------------------
    pol_path = results / args.policy_summary
    if pol_path.exists():
        pol = pd.read_csv(pol_path)

        # Fig 3: effectiveness (p_high) by policy and network
        if {"policy_label", "network_label", "p_high"}.issubset(pol.columns):
            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            # simple grouped bars
            policies = pol["policy_label"].unique().tolist()
            nets = pol["network_label"].unique().tolist()
            x = np.arange(len(policies))
            width = 0.8 / max(1, len(nets))

            for j, netlab in enumerate(nets):
                sub = pol[pol["network_label"] == netlab].set_index("policy_label").reindex(policies)
                ax.bar(x + j * width, sub["p_high"].to_numpy(), width=width, label=netlab)

            ax.set_ylim(0, 1)
            ax.set_xticks(x + width * (len(nets) - 1) / 2)
            ax.set_xticklabels(policies, rotation=20, ha="right")
            ax.set_title("Policy effectiveness: P(high adoption)")
            ax.set_ylabel("p_high")
            ax.legend()
            fig.savefig(figdir / "fig3_policy_effectiveness_rubric.png", dpi=300)
            plt.close(fig)

        # Fig 4: efficiency if cost columns present
        if {"policy_label", "network_label", "p_high_per_cost"}.issubset(pol.columns):
            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            policies = pol["policy_label"].unique().tolist()
            nets = pol["network_label"].unique().tolist()
            x = np.arange(len(policies))
            width = 0.8 / max(1, len(nets))

            for j, netlab in enumerate(nets):
                sub = pol[pol["network_label"] == netlab].set_index("policy_label").reindex(policies)
                ax.bar(x + j * width, sub["p_high_per_cost"].to_numpy(), width=width, label=netlab)

            ax.set_xticks(x + width * (len(nets) - 1) / 2)
            ax.set_xticklabels(policies, rotation=20, ha="right")
            ax.set_title("Policy efficiency: p_high per unit cost")
            ax.set_ylabel("p_high_per_cost")
            ax.legend()
            fig.savefig(figdir / "fig4_policy_efficiency_rubric.png", dpi=300)
            plt.close(fig)

    print("Saved figures to:", figdir)


if __name__ == "__main__":
    main()
