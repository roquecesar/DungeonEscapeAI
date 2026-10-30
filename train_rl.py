"""Train the Q-learning agent, log episodes with pandas, plot reward curves.

Usage:
    python train_rl.py
    python train_rl.py --episodes 500 --alpha 0.3 --gamma 0.99
    python train_rl.py --sweep      # compare hyperparameter settings
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save PNGs without needing a display
import matplotlib.pyplot as plt
import pandas as pd

from dungeon.level import Level
from dungeon.rl import MazeEnv, QAgent, greedy_path, train

OUT = Path("results")


def run_once(level, episodes, alpha, gamma, epsilon_decay, seed):
    env = MazeEnv(level)
    agent = QAgent(env.n_states, env.n_actions, alpha=alpha, gamma=gamma,
                   epsilon_decay=epsilon_decay, seed=seed)
    log = train(env, agent, episodes)
    return log, greedy_path(env, agent) is not None


def smooth(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window, min_periods=1).mean()


def plot_single(log: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(log["episode"], log["reward"], alpha=0.3, label="raw")
    ax[0].plot(log["episode"], smooth(log["reward"]), label="20-ep average")
    ax[0].set_ylabel("Total reward")
    ax[0].set_title(title)
    ax[0].legend()
    ax[1].plot(log["episode"], log["steps"], color="tab:orange")
    ax[1].set_ylabel("Steps to finish")
    ax[1].set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--alpha", type=float, default=0.5, help="learning rate")
    p.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    p.add_argument("--decay", type=float, default=0.98, help="epsilon decay per episode")
    p.add_argument("--size", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true", help="compare hyperparameters")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    level = Level.generate(args.size, args.size, seed=args.seed)

    if not args.sweep:
        log, solved = run_once(level, args.episodes, args.alpha, args.gamma,
                               args.decay, args.seed)
        log.to_csv(OUT / "training_log.csv", index=False)
        plot_single(log, OUT / "reward_curve.png",
                    f"Q-learning (alpha={args.alpha}, gamma={args.gamma})")
        print(log.tail())
        print(f"Policy solves maze: {solved}")
        print(f"Saved {OUT / 'training_log.csv'} and {OUT / 'reward_curve.png'}")
        return

    settings = [(a, g, d) for a in (0.1, 0.5, 0.9) for g in (0.9, 0.99) for d in (0.95, 0.99)]
    rows = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for a, g, d in settings:
        log, solved = run_once(level, args.episodes, a, g, d, args.seed)
        last = log.tail(50)
        rows.append({"alpha": a, "gamma": g, "eps_decay": d,
                     "avg_reward_last50": last["reward"].mean(),
                     "avg_steps_last50": last["steps"].mean(),
                     "solved": solved})
        ax.plot(log["episode"], smooth(log["reward"], 30), label=f"a={a} g={g} d={d}")
    summary = pd.DataFrame(rows).sort_values("avg_reward_last50", ascending=False)
    summary.to_csv(OUT / "sweep_summary.csv", index=False)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (30-ep avg)")
    ax.set_title("Hyperparameter sweep")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "sweep_comparison.png", dpi=150)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
