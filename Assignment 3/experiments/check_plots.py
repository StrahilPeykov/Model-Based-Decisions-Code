"""
Data validation and plot output check script
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

results = Path("results")
figdir = Path("figures")

print("=" * 60)
print("DATA VALIDATION & PLOT OUTPUT CHECK")
print("=" * 60)

# ----------------------------
# 1. Check Data Files Exist
# ----------------------------
print("\n1. CHECKING DATA FILES:")
files_to_check = [
    "baseline_sweep_summary.csv",
    "network_comparison.csv", 
    "policy_runs.csv",
    "policy_runs_summary.csv"
]

for file in files_to_check:
    filepath = results / file
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"✓ {file}: {size_mb:.2f} MB")
    else:
        print(f"✗ MISSING: {file}")

# ----------------------------
# 2. Load and Validate Data
# ----------------------------
print("\n2. DATA VALIDATION:")

# Baseline data
baseline_summary = pd.read_csv(results / "baseline_sweep_summary.csv")
print(f"Baseline rows: {len(baseline_summary)}")
print(f"Beta_I values: {sorted(baseline_summary['beta_I_grid'].unique())}")
print(f"X0 range: [{baseline_summary['X0'].min():.2f}, {baseline_summary['X0'].max():.2f}]")
print(f"I0 range: [{baseline_summary['I0_grid'].min():.2f}, {baseline_summary['I0_grid'].max():.2f}]")
print(f"p_high range: [{baseline_summary['p_high'].min():.3f}, {baseline_summary['p_high'].max():.3f}]")

# Count scenario types
baseline_summary['scenario_type'] = np.select(
    [
        baseline_summary['p_high'] < 0.1,
        (baseline_summary['p_high'] >= 0.2) & (baseline_summary['p_high'] <= 0.8),
        baseline_summary['p_high'] > 0.8
    ],
    ['hard', 'bistable', 'stable'],
    default='other'
)
print(f"\nScenario distribution:")
print(baseline_summary['scenario_type'].value_counts())

# Network data
network_runs = pd.read_csv(results / "network_comparison.csv")
print(f"\nNetwork runs: {len(network_runs)}")
print(f"Network types: {sorted(network_runs['network_label'].unique())}")
print(f"Network adoption rate (mean): {network_runs['high_adoption'].mean():.3f}")

# Policy data
policy_runs = pd.read_csv(results / "policy_runs.csv")
print(f"\nPolicy runs: {len(policy_runs)}")
print(f"Policy types: {sorted(policy_runs['policy_type'].unique())}")
print(f"Unique policies: {policy_runs['policy_label'].nunique()}")

# ----------------------------
# 3. Check Policy Parameters
# ----------------------------
print("\n3. POLICY PARAMETER RANGES:")

# Extract parameters from policy labels or columns
def extract_policy_params(policy_label):
    params = {}
    parts = policy_label.split(':')
    for part in parts:
        if part.startswith('f'):
            params['fraction'] = float(part[1:])
        elif part.startswith('s'):
            params['start'] = int(part[1:])
        elif part.startswith('d'):
            params['duration'] = int(part[1:])
        elif 'da0' in part:
            params['delta_a0'] = float(part[3:])
        elif 'boost' in part:
            params['boost'] = float(part[5:])
    return params

# Check parameter ranges for each policy type
for ptype in policy_runs['policy_type'].unique():
    if ptype != 'baseline':
        subset = policy_runs[policy_runs['policy_type'] == ptype]
        print(f"\n{ptype.upper()}:")
        
        if 'policy_fraction' in subset.columns:
            print(f"  Fraction: [{subset['policy_fraction'].min():.3f}, {subset['policy_fraction'].max():.3f}]")
        if 'policy_start' in subset.columns:
            print(f"  Start: [{subset['policy_start'].min()}, {subset['policy_start'].max()}]")
        if 'policy_duration' in subset.columns:
            print(f"  Duration: [{subset['policy_duration'].min()}, {subset['policy_duration'].max()}]")
        if 'policy_delta_a0' in subset.columns:
            print(f"  Delta_a0: [{subset['policy_delta_a0'].min():.3f}, {subset['policy_delta_a0'].max():.3f}]")
        if 'policy_boost' in subset.columns:
            print(f"  Boost: [{subset['policy_boost'].min():.3f}, {subset['policy_boost'].max():.3f}]")

# ----------------------------
# 4. Calculate Key Statistics
# ----------------------------
print("\n4. KEY STATISTICS:")

# Baseline vs Policy comparison
baseline_success = policy_runs[policy_runs['policy_type'] == 'baseline']['high_adoption'].mean()
print(f"Baseline success rate: {baseline_success:.3f}")

for ptype in policy_runs['policy_type'].unique():
    if ptype != 'baseline':
        ptype_success = policy_runs[policy_runs['policy_type'] == ptype]['high_adoption'].mean()
        improvement = ptype_success - baseline_success
        print(f"{ptype} success rate: {ptype_success:.3f} (Δ={improvement:+.3f})")

# Network comparison
print(f"\nNetwork success rates:")
for network in network_runs['network_label'].unique():
    network_success = network_runs[network_runs['network_label'] == network]['high_adoption'].mean()
    n_runs = len(network_runs[network_runs['network_label'] == network])
    print(f"  {network}: {network_success:.3f} (n={n_runs})")

# ----------------------------
# 5. Check Figure Outputs
# ----------------------------
print("\n5. FIGURE OUTPUTS:")

figures = [
    "fig1_baseline_phase_diagrams.png",
    "fig2_network_structure_analysis.png",
    "fig3_policy_effectiveness.png",
    "fig4_network_policy_interactions.png",
    "fig5_policy_insights_recommendations.png"
]

for fig in figures:
    figpath = figdir / fig
    if figpath.exists():
        size_kb = figpath.stat().st_size / 1024
        print(f"✓ {fig}: {size_kb:.0f} KB")
    else:
        print(f"✗ MISSING: {fig}")

# ----------------------------
# 6. Data Quality Checks
# ----------------------------
print("\n6. DATA QUALITY CHECKS:")

# Check for missing values
print("Missing values:")
for df_name, df in [("baseline", baseline_summary), ("network", network_runs), ("policy", policy_runs)]:
    missing = df.isnull().sum().sum()
    total = df.size
    print(f"  {df_name}: {missing}/{total} ({missing/total*100:.1f}%)")

# Check for extreme values
print("\nExtreme values check:")
print(f"  p_high values: [{baseline_summary['p_high'].min():.3f}, {baseline_summary['p_high'].max():.3f}]")
if baseline_summary['p_high'].max() > 1.0 or baseline_summary['p_high'].min() < 0.0:
    print("  ⚠️  WARNING: p_high outside [0,1] range!")

# Check policy cost distribution
if 'policy_cost' in policy_runs.columns:
    print(f"\nPolicy cost statistics:")
    print(f"  Min: {policy_runs['policy_cost'].min():.1f}")
    print(f"  Max: {policy_runs['policy_cost'].max():.1f}")
    print(f"  Mean: {policy_runs['policy_cost'].mean():.1f}")
    print(f"  Zero-cost policies: {(policy_runs['policy_cost'] == 0).sum()}/{len(policy_runs)}")

# ----------------------------
# 7. Statistical Significance
# ----------------------------
print("\n7. STATISTICAL SIGNIFICANCE:")

from scipy import stats

# Compare baseline vs each policy type
print("Mann-Whitney U tests (vs baseline):")
baseline_data = policy_runs[policy_runs['policy_type'] == 'baseline']['high_adoption']

for ptype in policy_runs['policy_type'].unique():
    if ptype != 'baseline':
        ptype_data = policy_runs[policy_runs['policy_type'] == ptype]['high_adoption']
        u_stat, p_value = stats.mannwhitneyu(baseline_data, ptype_data, alternative='two-sided')
        print(f"  {ptype}: p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")

# ----------------------------
# 8. Summary Report
# ----------------------------
print("\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)

total_runs = len(baseline_summary) + len(network_runs) + len(policy_runs)
print(f"Total simulation runs: {total_runs:,}")

# Calculate bistable scenarios
bistable_count = ((baseline_summary['p_high'] >= 0.2) & (baseline_summary['p_high'] <= 0.8)).sum()
print(f"Bistable parameter combinations: {bistable_count}/{len(baseline_summary)}")

# Best performing policy
if 'policy_type' in policy_runs.columns and 'high_adoption' in policy_runs.columns:
    best_policy = policy_runs[policy_runs['policy_type'] != 'baseline'].groupby('policy_type')['high_adoption'].mean().idxmax()
    best_value = policy_runs[policy_runs['policy_type'] != 'baseline'].groupby('policy_type')['high_adoption'].mean().max()
    print(f"Best performing policy: {best_policy} ({best_value:.3f})")

# Best performing network
if 'network_label' in network_runs.columns and 'high_adoption' in network_runs.columns:
    best_network = network_runs.groupby('network_label')['high_adoption'].mean().idxmax()
    best_net_value = network_runs.groupby('network_label')['high_adoption'].mean().max()
    print(f"Best performing network: {best_network} ({best_net_value:.3f})")

print("\nPLOT INTERPRETATION GUIDE:")
print("-" * 40)
print("Fig1: Look for yellow/orange regions (p_high > 0.8 = success)")
print("Fig2: Compare bar heights - higher = better adoption")
print("Fig3: Left panels = effectiveness, right panels = cost-benefit")
print("Fig4: Shows which policies work best on which networks")
print("Fig5: Practical recommendations for policymakers")

# ----------------------------
# 9. Generate Quick Visual Check
# ----------------------------
print("\n" + "=" * 60)
print("QUICK VISUAL CHECK")
print("=" * 60)

# Create a simple diagnostic plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Baseline heatmap (quick version)
beta_choice = 2.0 if 2.0 in baseline_summary['beta_I_grid'].unique() else baseline_summary['beta_I_grid'].iloc[0]
baseline_subset = baseline_summary[baseline_summary['beta_I_grid'] == beta_choice]
if not baseline_subset.empty:
    heat_data = baseline_subset.pivot(index='X0', columns='I0_grid', values='p_high')
    im1 = axes[0].imshow(heat_data.values, origin='lower', aspect='auto', cmap='RdBu_r', vmin=0, vmax=1)
    axes[0].set_title(f'Baseline (β={beta_choice})')
    axes[0].set_xlabel('I0')
    axes[0].set_ylabel('X0')
    plt.colorbar(im1, ax=axes[0])

# Plot 2: Network comparison
if not network_runs.empty:
    network_means = network_runs.groupby('network_label')['high_adoption'].mean()
    network_stds = network_runs.groupby('network_label')['high_adoption'].std()
    x_pos = range(len(network_means))
    axes[1].bar(x_pos, network_means.values, yerr=network_stds.values, capsize=5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(network_means.index, rotation=45)
    axes[1].set_title('Network Comparison')
    axes[1].set_ylabel('P(High Adoption)')
    axes[1].set_ylim(0, 1)

# Plot 3: Policy comparison
if not policy_runs.empty:
    policy_means = policy_runs.groupby('policy_type')['high_adoption'].mean()
    policy_counts = policy_runs.groupby('policy_type')['high_adoption'].count()
    x_pos = range(len(policy_means))
    colors = ['gray' if ptype == 'baseline' else 'steelblue' for ptype in policy_means.index]
    axes[2].bar(x_pos, policy_means.values, color=colors)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(policy_means.index, rotation=45)
    axes[2].set_title('Policy Comparison')
    axes[2].set_ylabel('P(High Adoption)')
    axes[2].set_ylim(0, 1)
    
    # Add value labels
    for i, (mean, count) in enumerate(zip(policy_means.values, policy_counts.values)):
        axes[2].text(i, mean + 0.02, f'{mean:.2f}\n(n={count})', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
check_plot_path = figdir / "data_validation_check.png"
plt.savefig(check_plot_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Diagnostic plot saved: {check_plot_path}")
print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
print("\nRED FLAGS TO WATCH FOR:")
print("1. Missing data files")
print("2. p_high values outside [0,1] range")
print("3. Policy costs that are all 0 or NaN")
print("4. Network types with < 10 runs")
print("5. Statistical tests showing no significance (p > 0.05)")
print("6. Empty or corrupt figure files")