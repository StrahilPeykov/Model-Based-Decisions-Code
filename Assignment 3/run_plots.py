"""
Plotting!

Produces:
- Fig 1: Baseline tipping + bistability evidence (heatmaps + tipping curve + bimodality)
- Fig 2: Network structure effects (p_high, t_high, clustering) with uncertainty when available
- Fig 3: Policy effect size (Δp_high vs baseline) across networks (readable labels)
- Fig 4: Policy efficiency scatter (cost vs Δp_high) + timing sensitivity (baseline line)

Usage:
  python run_plots.py --beta 2.0 --ratio 2.2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PINK = "#ff69b4"  # hot pink
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.8


def _ensure_bistable(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """
    Rubric definition: bistable if both low and high basins occur.
    If p_low missing, fall back to "mixed outcomes" p_high in (alpha, 1-alpha).
    """
    df = df.copy()
    if "p_low" in df.columns and "p_high" in df.columns:
        df["bistable"] = ((df["p_high"] >= alpha) & (df["p_low"] >= alpha)).astype(int)
    elif "p_high" in df.columns:
        df["bistable"] = ((df["p_high"] >= alpha) & (df["p_high"] <= 1 - alpha)).astype(int)
    else:
        df["bistable"] = 0
    return df


def _heatmap(
    ax,
    piv: pd.DataFrame,
    title: str,
    cbar_label: str,
    vmin=None,
    vmax=None,
    cmap=None,
    overlay_contour: Optional[float] = None,
):
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

    if overlay_contour is not None:
        # contour expects grid coords; we use linspace indices and then map via extent visually
        try:
            Xg, Yg = np.meshgrid(np.arange(piv.shape[1]), np.arange(piv.shape[0]))
            cs = ax.contour(
                Xg,
                Yg,
                piv.values,
                levels=[overlay_contour],
                colors=[PINK],
                linewidths=1.8,
            )
            for c in cs.collections:
                c.set_alpha(0.9)
        except Exception:
            pass

    cb = plt.colorbar(img, ax=ax)
    cb.set_label(cbar_label)


def _closest_available_beta(df: pd.DataFrame, requested: float) -> float:
    betas = np.sort(df["beta_I_grid"].unique())
    return float(betas[np.argmin(np.abs(betas - requested))])


def _binom_se(p: float, n: int) -> float:
    if n <= 1 or not np.isfinite(p):
        return np.nan
    return float(np.sqrt(max(p * (1 - p), 0.0) / n))


def _mean_se(std: float, n: int) -> float:
    if n <= 1 or not np.isfinite(std):
        return np.nan
    return float(std / np.sqrt(n))


def _short_label(s: str, max_len: int = 18) -> str:
    """
    Compress long policy labels into readable tokens.
    Keeps timing and intensity cues when present.
    """
    s = str(s)

    # common normalisations
    s = s.replace("barabasi_albert", "BA").replace("erdos_renyi", "ER").replace("small_world", "SW")
    s = s.replace("seed:degree", "seed_deg").replace("seed:centrality", "seed_cent")
    s = s.replace("infra_boost", "infra").replace("subsidy", "subs")
    s = s.replace(":", "_")

    # compress floats like 0.050000 -> 0.05
    s = re.sub(r"(\d)\.(\d{2})\d+", r"\1.\2", s)

    if len(s) <= max_len:
        return s

    # keep meaningful tail (often includes s0/d10 etc)
    head = s[: max_len - 4]
    return head + "…"


def _policy_effect_table(pol: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a tidy table with Δp_high relative to baseline per network.
    Expects policy summary to have: policy_label, network_label, p_high, and ideally n.
    """
    pol = pol.copy()

    # baseline per network
    base = pol[pol["policy_label"].astype(str).str.lower().eq("baseline")]
    if base.empty:
        # fallback: try startswith baseline
        base = pol[pol["policy_label"].astype(str).str.lower().str.startswith("baseline")]

    if base.empty:
        # if no baseline exists, cannot compute Δ; return empty
        return pd.DataFrame()

    base_map = base.set_index("network_label")["p_high"].to_dict()

    pol["p_base"] = pol["network_label"].map(base_map)
    pol["delta_p_high"] = pol["p_high"] - pol["p_base"]

    # keep non-baseline policies only
    out = pol[~pol["policy_label"].astype(str).str.lower().str.startswith("baseline")].copy()

    # optional uncertainty (binomial SE)
    if "n" in out.columns:
        out["se_p"] = out.apply(lambda r: _binom_se(r["p_high"], int(r["n"])), axis=1)
        # SE for difference (approx): sqrt(se^2 + se_base^2)
        base_se = {}
        if "n" in base.columns:
            for _, r in base.iterrows():
                base_se[r["network_label"]] = _binom_se(r["p_high"], int(r["n"]))
        out["se_delta"] = np.sqrt(out["se_p"] ** 2 + out["network_label"].map(base_se).fillna(0.0) ** 2)
    else:
        out["se_delta"] = np.nan

    # label compression for plotting
    out["policy_short"] = out["policy_label"].apply(_short_label)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    parser.add_argument("--figdir", type=str, default="figures")
    parser.add_argument("--beta", type=float, default=2.0, help="Which beta_I_grid slice to plot.")
    parser.add_argument("--ratio", type=float, default=None, help="Draw benchmark line X0=1/ratio on tipping plot.")
    parser.add_argument("--baseline-summary", type=str, default="baseline_sweep_summary.csv")
    parser.add_argument("--baseline-raw", type=str, default="baseline_sweep.csv")
    parser.add_argument("--network-summary", type=str, default="network_comparison_summary.csv")
    parser.add_argument("--policy-summary", type=str, default="policy_runs_summary.csv")
    parser.add_argument("--top-policies", type=int, default=10, help="Max policies per network to plot in Fig 3/4.")
    args = parser.parse_args()

    results = Path(args.results)
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Load baseline
    # ----------------------------
    base_sum = pd.read_csv(results / args.baseline_summary)
    base_raw = pd.read_csv(results / args.baseline_raw)

    base_sum = _ensure_bistable(base_sum)

    if "beta_I_grid" in base_sum.columns:
        beta = args.beta
        if beta not in set(base_sum["beta_I_grid"].unique()):
            beta = _closest_available_beta(base_sum, beta)
        dfb = base_sum[base_sum["beta_I_grid"] == beta].copy()
    else:
        beta = args.beta
        dfb = base_sum.copy()

    # ----------------------------
    # Fig 1 — Baseline: tipping + bistability evidence
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    piv_ph = dfb.pivot(index="I0_grid", columns="X0", values="p_high").sort_index()
    _heatmap(
        axes[0, 0],
        piv_ph,
        title=f"A. P(high adoption) at β_I={beta:g}",
        cbar_label="p_high",
        vmin=0,
        vmax=1,
        cmap="viridis",
        overlay_contour=0.5,  # tipping boundary cue
    )

    piv_bi = dfb.pivot(index="I0_grid", columns="X0", values="bistable").sort_index()
    _heatmap(
        axes[0, 1],
        piv_bi,
        title="B. Bistable region (two basins observed)",
        cbar_label="bistable (0/1)",
        vmin=0,
        vmax=1,
        cmap="binary",
    )

    # tipping curve: mean p_high by X0 (avg over I0)
    curve = dfb.groupby("X0", as_index=False)["p_high"].mean().sort_values("X0")
    axes[1, 0].plot(curve["X0"], curve["p_high"], linewidth=2.5, color=PINK)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("C. Tipping curve: mean p_high vs X₀ (avg over I₀)")
    axes[1, 0].set_xlabel("X₀")
    axes[1, 0].set_ylabel("mean p_high")

    if args.ratio is not None and args.ratio > 0:
        xcrit = 1.0 / float(args.ratio)
        axes[1, 0].axvline(xcrit, linestyle="--", linewidth=2.0, color=PINK)
        axes[1, 0].text(xcrit, 0.05, f" 1/ratio={xcrit:.2f}", rotation=90, va="bottom")

    # bimodality in bistable cells
    bi_cells = dfb[dfb["bistable"] == 1][["X0", "I0_grid"]].drop_duplicates()
    if not bi_cells.empty:
        raw = base_raw.copy()
        if "beta_I_grid" in raw.columns and "beta_I_grid" in dfb.columns:
            raw = raw[raw["beta_I_grid"] == beta]

        raw_bi = raw.merge(bi_cells, on=["X0", "I0_grid"], how="inner")

        if "X_final" in raw_bi.columns and len(raw_bi) > 0:
            axes[1, 1].hist(
                raw_bi["X_final"].dropna().to_numpy(),
                bins=20,
                range=(0, 1),
                edgecolor=PINK,
                linewidth=1.2,
            )
            axes[1, 1].axvline(0.2, linestyle="--", color=PINK, linewidth=1.8)
            axes[1, 1].axvline(0.8, linestyle="--", color=PINK, linewidth=1.8)
            axes[1, 1].set_title("D. Final adoption X* in bistable cells (bimodality)")
            axes[1, 1].set_xlabel("X*")
            axes[1, 1].set_ylabel("count")
        else:
            axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    fig.savefig(figdir / "fig1_baseline.png", dpi=300)
    plt.close(fig)

    # ----------------------------
    # Fig 2 — Network structure effects (with uncertainty if possible)
    # ----------------------------
    net_path = results / args.network_summary
    if net_path.exists():
        net = pd.read_csv(net_path).copy()

        # prefer stable ordering
        if "network_label" in net.columns:
            order = ["barabasi_albert", "erdos_renyi", "grid", "small_world"]
            net["network_label"] = pd.Categorical(net["network_label"], categories=order, ordered=True)
            net = net.sort_values("network_label")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # A: p_high with binomial SE if n provided
        if "p_high" in net.columns and "network_label" in net.columns:
            y = net["p_high"].to_numpy()
            x = np.arange(len(net))
            axes[0].bar(net["network_label"].astype(str), y, edgecolor=PINK, linewidth=1.2)
            if "n" in net.columns:
                se = np.array([_binom_se(p, int(n)) for p, n in zip(net["p_high"], net["n"])])
                axes[0].errorbar(
                    x, y, yerr=1.96 * se,
                    fmt="none", ecolor=PINK, elinewidth=2.0, capsize=4
                )
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(net["network_label"].astype(str))
            axes[0].set_ylim(0, 1)
            axes[0].set_title("A. Adoption probability (p_high) with 95% CI")
            axes[0].set_xlabel("Network")
            axes[0].set_ylabel("p_high")

        # B: time to tipping (clarify: mean over successful runs if your summary is defined that way)
        if "t_high_mean" in net.columns and "network_label" in net.columns:
            y = net["t_high_mean"].to_numpy()
            x = np.arange(len(net))
            axes[1].bar(net["network_label"].astype(str), y, edgecolor=PINK, linewidth=1.2)
            # add SE if available
            if "t_high_std" in net.columns and "n" in net.columns:
                se = np.array([_mean_se(s, int(n)) for s, n in zip(net["t_high_std"], net["n"])])
                axes[1].errorbar(
                    x, y, yerr=1.96 * se,
                    fmt="none", ecolor=PINK, elinewidth=2.0, capsize=4
                )
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(net["network_label"].astype(str))
            axes[1].set_title("B. Time to tipping (mean t_high)")
            axes[1].set_xlabel("Network")
            axes[1].set_ylabel("t_high")

        # C: largest adopter cluster
        if "cluster_max_mean" in net.columns and "network_label" in net.columns:
            y = net["cluster_max_mean"].to_numpy()
            x = np.arange(len(net))
            axes[2].bar(net["network_label"].astype(str), y, edgecolor=PINK, linewidth=1.2)
            if "cluster_max_std" in net.columns and "n" in net.columns:
                se = np.array([_mean_se(s, int(n)) for s, n in zip(net["cluster_max_std"], net["n"])])
                axes[2].errorbar(
                    x, y, yerr=1.96 * se,
                    fmt="none", ecolor=PINK, elinewidth=2.0, capsize=4
                )
                axes[2].set_xticks(x)
                axes[2].set_xticklabels(net["network_label"].astype(str))
            axes[2].set_title("C. Largest adopter cluster (mean)")
            axes[2].set_xlabel("Network")
            axes[2].set_ylabel("cluster_max")

        fig.savefig(figdir / "fig2_network_effects.png", dpi=300)
        plt.close(fig)

    # ----------------------------
    # Fig 3–4 — Policy: Δp_high vs baseline + efficiency scatter
    # ----------------------------
    pol_path = results / args.policy_summary
    if pol_path.exists():
        pol = pd.read_csv(pol_path).copy()

        required = {"policy_label", "network_label", "p_high"}
        if required.issubset(pol.columns):
            eff = _policy_effect_table(pol)
            if not eff.empty:
                # keep top policies per network by |Δp_high|
                eff["abs_delta"] = eff["delta_p_high"].abs()
                eff = eff.sort_values(["network_label", "abs_delta"], ascending=[True, False])
                eff = eff.groupby("network_label").head(int(args.top_policies)).copy()

                # Fig 3: Δp_high bars by network (readable)
                nets = eff["network_label"].unique().tolist()
                fig, axes = plt.subplots(len(nets), 1, figsize=(12, 4 * len(nets)), constrained_layout=True)
                if len(nets) == 1:
                    axes = [axes]

                for ax, netlab in zip(axes, nets):
                    sub = eff[eff["network_label"] == netlab].sort_values("delta_p_high", ascending=False)
                    x = np.arange(len(sub))
                    ax.bar(sub["policy_short"], sub["delta_p_high"].to_numpy(), edgecolor=PINK, linewidth=1.2)
                    ax.axhline(0.0, linestyle="--", color=PINK, linewidth=2.0)
                    if "se_delta" in sub.columns and sub["se_delta"].notna().any():
                        ax.errorbar(
                            x,
                            sub["delta_p_high"].to_numpy(),
                            yerr=1.96 * sub["se_delta"].to_numpy(),
                            fmt="none",
                            ecolor=PINK,
                            elinewidth=2.0,
                            capsize=4,
                        )
                        ax.set_xticks(x)
                        ax.set_xticklabels(sub["policy_short"].tolist(), rotation=20, ha="right")

                    ax.set_title(f"Policy effect size (Δp_high vs baseline) — {netlab}")
                    ax.set_ylabel("Δp_high")
                    ax.set_xlabel("Policy (short label)")
                    ax.set_ylim(-1, 1)

                fig.savefig(figdir / "fig3_policy_effect_size.png", dpi=300)
                plt.close(fig)

                # Fig 4: efficiency scatter (cost vs Δp_high), if cost available
                cost_col = None
                for c in ["cost", "policy_cost", "total_cost"]:
                    if c in eff.columns:
                        cost_col = c
                        break

                if cost_col is not None:
                    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

                    for netlab in nets:
                        sub = eff[eff["network_label"] == netlab]
                        ax.scatter(sub[cost_col], sub["delta_p_high"], label=netlab)

                    ax.axhline(0.0, linestyle="--", color=PINK, linewidth=2.0)
                    ax.set_title("Policy efficiency: cost vs Δp_high (relative to baseline)")
                    ax.set_xlabel(cost_col)
                    ax.set_ylabel("Δp_high")
                    ax.legend()
                    fig.savefig(figdir / "fig4_policy_efficiency_scatter.png", dpi=300)
                    plt.close(fig)

    print("Saved figures to:", figdir)


if __name__ == "__main__":
    main()
