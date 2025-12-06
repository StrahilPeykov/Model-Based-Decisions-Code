# Assignment 3 experiment scripts

New utilities that leave the provided model code untouched.

## Shared pieces
- `sim_utils.py` — network-flexible runner with policy hooks.
- `networks.py` — builders for `grid`, `small_world`, `erdos_renyi`, `barabasi_albert`.

## Scripts
- `run_baseline_sweeps.py` — grid over `X0`/`I0` (and `beta_I` list).  
  Example: `python experiments/run_baseline_sweeps.py --network-type erdos_renyi --n-nodes 300 --reps 5`
- `run_beta_sensitivity.py` — varies `beta_I` for fixed `X0`/`I0`.  
  Example: `python experiments/run_beta_sensitivity.py --x0 0.1 --i0 0.05 --beta-list 0.5 1.0 1.5 2.0`
- `run_phase_trajectories.py` — short time-series for low/mid/high starts.  
  Example: `python experiments/run_phase_trajectories.py --collect-series`
- `run_network_experiments.py` — compares network topologies.  
  Example: `python experiments/run_network_experiments.py --network-types grid small_world erdos_renyi`
- `run_policy_experiments.py` — baseline vs targeted/timed interventions.  
  Example: `python experiments/run_policy_experiments.py --collect-series --network-type small_world`

All outputs land in `results/` by default; adjust via `--output` flags on each script.
