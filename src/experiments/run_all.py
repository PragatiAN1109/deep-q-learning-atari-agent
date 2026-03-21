"""
src/experiments/run_all.py
---------------------------
Master experiment runner — executes all 4 experiment tasks and
saves results to the experiments/ directory.

Tasks covered:
  Task 1 — Hyperparameter sweep (α and γ)
  Task 2 — Exploration strategy comparison (epsilon-greedy vs softmax)
  Task 3 — Epsilon decay rate comparison
  Task 4 — Performance metrics collection (via src/utils/metrics.py)

All experiments use 300 episodes by default (configurable via --episodes).
This keeps total runtime manageable on CPU while showing meaningful trends.

Usage:
    # Run all experiments (from repo root):
    python src/experiments/run_all.py

    # Fewer episodes for a quick test:
    python src/experiments/run_all.py --episodes 100

    # Run a specific task only:
    python src/experiments/run_all.py --task 1
    python src/experiments/run_all.py --task 2
    python src/experiments/run_all.py --task 3

Outputs:
    experiments/hyperparameter_results.csv
    experiments/exploration_comparison.csv
    experiments/epsilon_decay_results.csv
    experiments/plots/reward_hp_sweep.png
    experiments/plots/reward_exploration.png
    experiments/plots/epsilon_decay_curves.png
    experiments/plots/epsilon_decay_rewards.png
"""

import sys, os, csv, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
from src.experiments.engine     import run_experiment
from src.experiments.exploration import softmax_action, epsilon_greedy_action
from src.experiments.plot_utils  import (plot_reward_curves,
                                          plot_epsilon_curves,
                                          plot_comparison_grid)
from src.utils.metrics           import compute_metrics, print_metrics_table

EXP_DIR  = os.path.join(ROOT, "experiments")
PLOT_DIR = os.path.join(EXP_DIR, "plots")


# ═══════════════════════════════════════════════════════════════
# CSV WRITER HELPER
# ═══════════════════════════════════════════════════════════════
def write_results_csv(
    rows:     list,
    fieldnames: list,
    filepath: str,
) -> None:
    """Write a list of dicts to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  [CSV] Saved → {filepath}")


# ═══════════════════════════════════════════════════════════════
# TASK 1 — Hyperparameter Sweep (α and γ)
# ═══════════════════════════════════════════════════════════════
def task1_hyperparameter_sweep(n_episodes: int) -> None:
    """
    Vary learning rate (α) and discount factor (γ) independently.
    All other parameters held constant (baseline values).

    α values tested : 0.0001, 0.0005, 0.001
    γ values tested : 0.90,   0.95,   0.99
    """
    print("\n" + "="*60)
    print("  TASK 1 — Hyperparameter Sweep (α and γ)")
    print("="*60)

    BASE_LR    = 0.0005
    BASE_GAMMA = 0.99

    # ── α sweep (gamma fixed at baseline) ────────────────────
    lr_values = [0.0001, 0.0005, 0.001]
    lr_rewards  = {}
    lr_epsilons = {}
    rows = []

    print("\n  [α sweep] fixing γ=0.99")
    for lr in lr_values:
        label = f"α={lr}"
        rewards, steps, epsilons = run_experiment(
            learning_rate=lr, gamma=BASE_GAMMA,
            total_episodes=n_episodes, label=label, log_freq=n_episodes
        )
        stats = compute_metrics(rewards, steps)
        print_metrics_table(label, stats)
        lr_rewards[label]  = rewards
        lr_epsilons[label] = epsilons
        rows.append({
            "experiment"    : "lr_sweep",
            "learning_rate" : lr,
            "gamma"         : BASE_GAMMA,
            "decay_rate"    : 0.005,
            "avg_reward"    : stats["avg_reward"],
            "std_reward"    : stats["std_reward"],
            "best_mov_avg"  : stats["best_moving_avg"],
            "best_mov_ep"   : stats["best_moving_avg_episode"],
            "avg_steps"     : stats["avg_steps"],
        })

    # ── γ sweep (lr fixed at baseline) ────────────────────────
    gamma_values = [0.90, 0.95, 0.99]
    gamma_rewards  = {}
    gamma_epsilons = {}

    print("\n  [γ sweep] fixing α=0.0005")
    for g in gamma_values:
        label = f"γ={g}"
        rewards, steps, epsilons = run_experiment(
            learning_rate=BASE_LR, gamma=g,
            total_episodes=n_episodes, label=label, log_freq=n_episodes
        )
        stats = compute_metrics(rewards, steps)
        print_metrics_table(label, stats)
        gamma_rewards[label]  = rewards
        gamma_epsilons[label] = epsilons
        rows.append({
            "experiment"    : "gamma_sweep",
            "learning_rate" : BASE_LR,
            "gamma"         : g,
            "decay_rate"    : 0.005,
            "avg_reward"    : stats["avg_reward"],
            "std_reward"    : stats["std_reward"],
            "best_mov_avg"  : stats["best_moving_avg"],
            "best_mov_ep"   : stats["best_moving_avg_episode"],
            "avg_steps"     : stats["avg_steps"],
        })

    # ── Save CSV ──────────────────────────────────────────────
    fields = ["experiment","learning_rate","gamma","decay_rate",
              "avg_reward","std_reward","best_mov_avg","best_mov_ep","avg_steps"]
    write_results_csv(rows, fields,
                      os.path.join(EXP_DIR, "hyperparameter_results.csv"))

    # ── Plots ─────────────────────────────────────────────────
    plot_reward_curves(lr_rewards,
        title="Task 1a — Learning Rate Sweep (γ=0.99)",
        fname="reward_hp_lr_sweep.png")
    plot_reward_curves(gamma_rewards,
        title="Task 1b — Gamma Sweep (α=0.0005)",
        fname="reward_hp_gamma_sweep.png")


# ═══════════════════════════════════════════════════════════════
# TASK 2 — Exploration Strategy Comparison
# ═══════════════════════════════════════════════════════════════
def task2_exploration_comparison(n_episodes: int) -> None:
    """
    Compare epsilon-greedy vs softmax (Boltzmann) exploration.

    Epsilon-greedy: uses same decay schedule as baseline
    Softmax:        uses temperature τ=1.0 (fixed, no decay)
    Softmax cool:   uses temperature τ=0.5 (cooler, more greedy)
    """
    print("\n" + "="*60)
    print("  TASK 2 — Exploration Strategy Comparison")
    print("="*60)

    configs = [
        # (label, selector, param, description)
        ("Epsilon-Greedy (decay=0.005)",
         None,             None,  "Standard epsilon-greedy, exponential decay"),
        ("Softmax τ=1.0",
         softmax_action,   1.0,   "Boltzmann, temperature=1.0 (constant)"),
        ("Softmax τ=0.5",
         softmax_action,   0.5,   "Boltzmann, temperature=0.5 (cooler, greedier)"),
    ]

    all_rewards  = {}
    all_epsilons = {}
    rows = []

    for label, selector, param, desc in configs:
        rewards, steps, epsilons = run_experiment(
            action_selector   = selector,
            exploration_param = param,
            total_episodes    = n_episodes,
            label             = label,
            log_freq          = n_episodes,
        )
        stats = compute_metrics(rewards, steps)
        print_metrics_table(label, stats)
        all_rewards[label]  = rewards
        all_epsilons[label] = epsilons
        rows.append({
            "strategy"      : label,
            "description"   : desc,
            "avg_reward"    : stats["avg_reward"],
            "std_reward"    : stats["std_reward"],
            "best_mov_avg"  : stats["best_moving_avg"],
            "best_mov_ep"   : stats["best_moving_avg_episode"],
            "avg_steps"     : stats["avg_steps"],
        })

    fields = ["strategy","description","avg_reward","std_reward",
              "best_mov_avg","best_mov_ep","avg_steps"]
    write_results_csv(rows, fields,
                      os.path.join(EXP_DIR, "exploration_comparison.csv"))

    plot_reward_curves(all_rewards,
        title="Task 2 — Exploration Strategy: Epsilon-Greedy vs Softmax",
        fname="reward_exploration.png")


# ═══════════════════════════════════════════════════════════════
# TASK 3 — Epsilon Decay Rate Experiments
# ═══════════════════════════════════════════════════════════════
def task3_epsilon_decay(n_episodes: int) -> None:
    """
    Experiment with 3 different epsilon decay rates:
      - Fast  : 0.015 → reaches epsilon_min quickly (~ep 200)
      - Medium: 0.005 → baseline (reaches min ~ep 500)
      - Slow  : 0.002 → stays exploratory much longer (~ep 1000+)
    """
    print("\n" + "="*60)
    print("  TASK 3 — Epsilon Decay Rate Experiments")
    print("="*60)

    decay_values = [
        (0.015, "Fast decay  (rate=0.015)"),
        (0.005, "Medium decay (rate=0.005) — baseline"),
        (0.002, "Slow decay  (rate=0.002)"),
    ]

    all_rewards  = {}
    all_epsilons = {}
    rows = []

    for rate, label in decay_values:
        rewards, steps, epsilons = run_experiment(
            decay_rate     = rate,
            total_episodes = n_episodes,
            label          = label,
            log_freq       = n_episodes,
        )
        stats = compute_metrics(rewards, steps)
        print_metrics_table(label, stats)
        all_rewards[label]  = rewards
        all_epsilons[label] = epsilons
        rows.append({
            "decay_rate"    : rate,
            "label"         : label,
            "avg_reward"    : stats["avg_reward"],
            "std_reward"    : stats["std_reward"],
            "best_mov_avg"  : stats["best_moving_avg"],
            "best_mov_ep"   : stats["best_moving_avg_episode"],
            "avg_steps"     : stats["avg_steps"],
        })

    fields = ["decay_rate","label","avg_reward","std_reward",
              "best_mov_avg","best_mov_ep","avg_steps"]
    write_results_csv(rows, fields,
                      os.path.join(EXP_DIR, "epsilon_decay_results.csv"))

    # Save both reward and epsilon plots + combined grid
    plot_reward_curves(all_rewards,
        title="Task 3 — Epsilon Decay Rate: Reward Comparison",
        fname="epsilon_decay_rewards.png")
    plot_epsilon_curves(all_epsilons,
        title="Task 3 — Epsilon Decay Rate: Epsilon over Episodes",
        fname="epsilon_decay_curves.png")
    plot_comparison_grid(all_rewards, all_epsilons,
        title="Task 3 — Epsilon Decay: Reward & Epsilon Comparison",
        fname="epsilon_decay_grid.png")


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run DQN controlled experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episodes", type=int, default=300,
        help="Episodes per experiment run (reduce for faster testing)"
    )
    parser.add_argument(
        "--task", type=int, default=0,
        help="Run single task: 1=hp sweep, 2=exploration, 3=epsilon. "
             "0=run all tasks"
    )
    args = parser.parse_args()
    n    = args.episodes

    os.makedirs(PLOT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  DQN Controlled Experiments")
    print(f"  Episodes per run : {n}")
    print(f"  Results → {EXP_DIR}")
    print(f"  Plots   → {PLOT_DIR}")
    print(f"{'='*60}")

    if args.task == 0 or args.task == 1:
        task1_hyperparameter_sweep(n)
    if args.task == 0 or args.task == 2:
        task2_exploration_comparison(n)
    if args.task == 0 or args.task == 3:
        task3_epsilon_decay(n)

    print(f"\n{'='*60}")
    print("  ✅  All experiments complete.")
    print(f"  Results → {EXP_DIR}")
    print(f"  Plots   → {PLOT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
