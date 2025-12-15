"""
Plotting (LaTeX-ready, rubric-aligned).

Produces:
- Fig 1: Baseline tipping + bistability evidence (heatmaps + tipping curve + bimodality)
- Fig 2: Network structure effects (p_high, t_high, clustering) with uncertainty when available
- Fig 3: Policy effect size (Δp_high vs baseline) across networks (readable labels)
- Fig 4: Policy efficiency scatter (cost vs Δp_high) (proxy if needed)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- helpers -------------------

def _ensure_bistable(df: pd.DataFrame, alpha: float = 0.1) -> pd.DataFrame:
    """
    Rubric definition:
      - bistable iff both basins occur (requires p_high and p_low):
            p_high >= alpha AND p_low >= alpha
      - fallback if p_low missing:
            alpha <= p_high <= 1 - alpha
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
    ax.set_xlabel("Initial adoption $X_0$")
    ax.set_ylabel("Initial infrastructure $I_0$")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    if overlay_contour is not None:
        # default styling; will inherit Matplotlib defaults
        try:
            Xv, Yv = np.meshgrid(x, y)
            ax.contour(Xv, Yv, piv.values, levels=[overlay_contour], linewidths=1.2)
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


def _short_label(s: str, max_len: int = 22) -> str:
    """Shorten long policy labels for readable x-axis ticks."""
    s = str(s)
    s = s.replace("barabasi_albert", "BA").replace("erdos_renyi", "ER").replace("small_world", "SW")
    s = s.replace("seed:degree", "seed_deg").replace("seed:centrality", "seed_cent")
    s = s.replace("infra_boost", "infra").replace("subsidy", "subs")
    s = s.replace(":", "_")
    s = re.sub(r"(\d)\.(\d{2})\d+", r"\1.\2", s)  # 0.050000 -> 0.05

    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _policy_effect_table(pol: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Δp_high vs baseline per network.
    Requires: policy_label, network_label, p_high. Uses n if present for approximate SE.
    """
    pol = pol.copy()

    base = pol[pol["policy_label"].astype(str).str.lower().eq("baseline")]
    if base.empty:
        base = pol[pol["policy_label"].astype(str).str.lower().str.startswith("baseline")]
    if base.empty:
        return pd.DataFrame()

    base_map = base.set_index("network_label")["p_high"].to_dict()

    pol["p_base"] = pol["network_label"].map(base_map)
    pol["delta_p_high"] = pol["p_high"] - pol["p_base"]

    out = pol[~pol["policy_label"].astype(str).str.lower().str.startswith("baseline")].copy()

    if "n" in out.columns:
        out["se_p"] = out.apply(lambda r: _binom_se(float(r["p_high"]), int(r["n"])), axis=1)
        base_se = {}
        if "n" in base.columns:
            for _, r in base.iterrows():
                base_se[r["network_label"]] = _binom_se(float(r["p_high"]), int(r["n"]))
        out["se_delta"] = np.sqrt(out["se_p"] ** 2 + out["network_label"].map(base_se).fillna(0.0) ** 2)
    else:
        out["se_delta"] = np.nan

    out["policy_short"] = out["policy_label"].apply(_short_label)
    return out


def _save_png(fig, figdir: Path, stem: str):
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png", dpi=300, bbox_inches="tight")


# ------------ main ------------------

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
    # Fig 1 — Baseline
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    piv_ph = dfb.pivot(index="I0_grid", columns="X0", values="p_high").sort_index()
    _heatmap(
        axes[0, 0],
        piv_ph,
        title=f"A. $P(X^* > 0.8)$ at $\\beta_I={beta:g}$",
        cbar_label="$p_{high}$",
        vmin=0,
        vmax=1,
        cmap="viridis",
        overlay_contour=0.5,
    )

    piv_bi = dfb.pivot(index="I0_grid", columns="X0", values="bistable").sort_index()
    _heatmap(
        axes[0, 1],
        piv_bi,
        title="B. Bistable region (both basins observed)",
        cbar_label="bistable (0/1)",
        vmin=0,
        vmax=1,
        cmap="binary",
    )

    curve = dfb.groupby("X0", as_index=False)["p_high"].mean().sort_values("X0")
    axes[1, 0].plot(curve["X0"], curve["p_high"], linewidth=2.0)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("C. Tipping curve: $\\mathbb{E}[p_{high}]$ vs $X_0$ (avg over $I_0$)")
    axes[1, 0].set_xlabel("$X_0$")
    axes[1, 0].set_ylabel("mean $p_{high}$")

    if args.ratio is not None and args.ratio > 0:
        xcrit = 1.0 / float(args.ratio)
        axes[1, 0].axvline(xcrit, linestyle="--", linewidth=1.4)
        axes[1, 0].text(xcrit, 0.05, f"$1/\\text{{ratio}}={xcrit:.2f}$", rotation=90, va="bottom")

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
            )
            axes[1, 1].axvline(0.2, linestyle="--", linewidth=1.2)
            axes[1, 1].axvline(0.8, linestyle="--", linewidth=1.2)
            axes[1, 1].set_title("D. Final adoption $X^*$ in bistable cells (bimodality)")
            axes[1, 1].set_xlabel("$X^*$")
            axes[1, 1].set_ylabel("count")
        else:
            axes[1, 1].axis("off")
    else:
        axes[1, 1].axis("off")

    _save_png(fig, figdir, "fig1_baseline")
    plt.close(fig)

    # ----------------------------
    # Fig 2 — Network structure effects
    # ----------------------------
    net_path = results / args.network_summary
    if net_path.exists():
        net = pd.read_csv(net_path).copy()

        if "network_label" in net.columns:
            order = ["barabasi_albert", "erdos_renyi", "grid", "small_world"]
            net["network_label"] = pd.Categorical(net["network_label"], categories=order, ordered=True)
            net = net.sort_values("network_label")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        if {"p_high", "network_label"}.issubset(net.columns):
            y = net["p_high"].to_numpy()
            x = np.arange(len(net))
            axes[0].bar(net["network_label"].astype(str), y)
            if "n" in net.columns:
                se = np.array([_binom_se(float(p), int(n)) for p, n in zip(net["p_high"], net["n"])])
                axes[0].errorbar(x, y, yerr=1.96 * se, fmt="none", capsize=4)
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(net["network_label"].astype(str))
            axes[0].set_ylim(0, 1)
            axes[0].set_title("A. Adoption probability $p_{high}$ (95% CI if available)")
            axes[0].set_xlabel("Network")
            axes[0].set_ylabel("$p_{high}$")

        if {"t_high_mean", "network_label"}.issubset(net.columns):
            y = net["t_high_mean"].to_numpy()
            x = np.arange(len(net))
            axes[1].bar(net["network_label"].astype(str), y)
            if {"t_high_std", "n"}.issubset(net.columns):
                se = np.array([_mean_se(float(s), int(n)) for s, n in zip(net["t_high_std"], net["n"])])
                axes[1].errorbar(x, y, yerr=1.96 * se, fmt="none", capsize=4)
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(net["network_label"].astype(str))
            axes[1].set_title("B. Time to tipping (mean $t_{high}$)")
            axes[1].set_xlabel("Network")
            axes[1].set_ylabel("$t_{high}$")

        if {"cluster_max_mean", "network_label"}.issubset(net.columns):
            y = net["cluster_max_mean"].to_numpy()
            axes[2].bar(net["network_label"].astype(str), y)
            axes[2].set_title("C. Largest adopter cluster (mean)")
            axes[2].set_xlabel("Network")
            axes[2].set_ylabel("cluster\\_max")

        _save_png(fig, figdir, "fig2_network_effects")
        plt.close(fig)

    # ----------------------------
    # Fig 3–4 — Policy
    # ----------------------------
    pol_path = results / args.policy_summary
    if pol_path.exists():
        pol = pd.read_csv(pol_path).copy()

        if {"policy_label", "network_label", "p_high"}.issubset(pol.columns):
            eff = _policy_effect_table(pol)
            if not eff.empty:
                eff["abs_delta"] = eff["delta_p_high"].abs()
                eff = eff.sort_values(["network_label", "abs_delta"], ascending=[True, False])
                eff = eff.groupby("network_label").head(int(args.top_policies)).copy()

                nets = eff["network_label"].unique().tolist()

                # Fig 3: Δp_high per network
                fig, axes = plt.subplots(len(nets), 1, figsize=(12, 4 * len(nets)), constrained_layout=True)
                if len(nets) == 1:
                    axes = [axes]

                for ax, netlab in zip(axes, nets):
                    sub = eff[eff["network_label"] == netlab].sort_values("delta_p_high", ascending=False)
                    x = np.arange(len(sub))

                    ax.bar(sub["policy_short"], sub["delta_p_high"].to_numpy())
                    ax.axhline(0.0, linestyle="--", linewidth=1.2)

                    if sub["se_delta"].notna().any():
                        ax.errorbar(
                            x,
                            sub["delta_p_high"].to_numpy(),
                            yerr=1.96 * sub["se_delta"].to_numpy(),
                            fmt="none",
                            capsize=4,
                        )
                        ax.set_xticks(x)

                    ax.set_xticklabels(sub["policy_short"].tolist(), rotation=20, ha="right")

                    base_row = pol[
                        (pol["network_label"] == netlab)
                        & (pol["policy_label"].astype(str).str.lower().str.startswith("baseline"))
                    ]
                    base_ph = float(base_row["p_high"].iloc[0]) if not base_row.empty else np.nan

                    ax.set_title(
                        f"Policy effect size $\\Delta p_{{high}}$ vs baseline — {netlab} "
                        f"(baseline $p_{{high}}$={base_ph:.2f})"
                    )
                    ax.set_ylabel("$\\Delta p_{high}$")
                    ax.set_xlabel("Policy")
                    ax.set_ylim(-1, 1)

                _save_png(fig, figdir, "fig3_policy_effect_size")
                plt.close(fig)

                # Fig 4: cost vs Δp_high (proxy if needed)
                cost_col = None
                for c in ["cost", "policy_cost", "total_cost"]:
                    if c in eff.columns:
                        cost_col = c
                        break
                if cost_col is None and "p_high_per_cost" in eff.columns:
                    eff["cost_proxy"] = np.where(
                        eff["p_high_per_cost"] > 0,
                        eff["p_high"] / eff["p_high_per_cost"],
                        np.nan,
                    )
                    cost_col = "cost_proxy"

                if cost_col is not None:
                    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

                    for netlab in nets:
                        sub = eff[eff["network_label"] == netlab]
                        ax.scatter(sub[cost_col], sub["delta_p_high"], label=netlab)

                    ax.axhline(0.0, linestyle="--", linewidth=1.2)
                    title = "Policy efficiency: cost vs $\\Delta p_{high}$ (relative to baseline)"
                    if cost_col == "cost_proxy":
                        title += " (cost proxy)"
                    ax.set_title(title)
                    ax.set_xlabel(cost_col)
                    ax.set_ylabel("$\\Delta p_{high}$")
                    ax.legend()

                    _save_png(fig, figdir, "fig4_policy_efficiency_scatter")
                    plt.close(fig)

    print("Saved figures to:", figdir)


if __name__ == "__main__":
    main()
