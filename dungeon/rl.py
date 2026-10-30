"""Tabular Q-learning agent that learns to solve a maze (NumPy + pandas)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .level import DIRECTIONS, Level, Pos


class MazeEnv:
    """RL view of a Level. State = tile index, actions = up/down/left/right.

    Rewards: +1.0 reaching the exit, -0.01 per step, -0.05 for bumping a wall.
    """

    GOAL_REWARD = 1.0
    STEP_PENALTY = -0.01
    WALL_PENALTY = -0.05

    def __init__(self, level: Level, max_steps: int | None = None) -> None:
        self.level = level
        self.n_states = level.width * level.height
        self.n_actions = len(DIRECTIONS)
        self.max_steps = max_steps or self.n_states * 4
        self.pos: Pos = level.start
        self.steps = 0

    def state_index(self, pos: Pos) -> int:
        return pos[1] * self.level.width + pos[0]

    def reset(self) -> int:
        self.pos = self.level.start
        self.steps = 0
        return self.state_index(self.pos)

    def step(self, action: int) -> tuple[int, float, bool]:
        dx, dy = DIRECTIONS[action]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        self.steps += 1
        if self.level.is_walkable(nx, ny):
            self.pos = (nx, ny)
            reward = self.STEP_PENALTY
        else:
            reward = self.WALL_PENALTY
        done = False
        if self.pos == self.level.exit:
            reward = self.GOAL_REWARD
            done = True
        elif self.steps >= self.max_steps:
            done = True
        return self.state_index(self.pos), reward, done


class QAgent:
    """Epsilon-greedy Q-learning with a NumPy Q-table."""

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.5,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.98, seed: int | None = None) -> None:
        self.q = np.zeros((n_states, n_actions))
        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate, decays each episode
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        row = self.q[state]
        best = np.flatnonzero(row == row.max())  # random tie-break
        return int(self.rng.choice(best))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> None:
        target = r if done else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env: MazeEnv, agent: QAgent, episodes: int = 300) -> pd.DataFrame:
    """Train the agent and return a per-episode log as a pandas DataFrame."""
    records = []
    for ep in range(1, episodes + 1):
        state = env.reset()
        total, done = 0.0, False
        while not done:
            action = agent.choose_action(state)
            nxt, reward, done = env.step(action)
            agent.update(state, action, reward, nxt, done)
            state = nxt
            total += reward
        records.append({"episode": ep, "reward": total, "steps": env.steps,
                        "epsilon": agent.epsilon, "success": env.pos == env.level.exit})
        agent.decay_epsilon()
    return pd.DataFrame(records)


def greedy_path(env: MazeEnv, agent: QAgent, limit: int | None = None) -> list[Pos] | None:
    """Follow the learned policy with no exploration. None if it hasn't converged."""
    limit = limit or env.n_states
    pos = env.level.start
    path = [pos]
    visited = {pos}
    for _ in range(limit):
        if pos == env.level.exit:
            return path
        action = int(np.argmax(agent.q[env.state_index(pos)]))
        dx, dy = DIRECTIONS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not env.level.is_walkable(*nxt) or nxt in visited:
            return None
        pos = nxt
        visited.add(pos)
        path.append(pos)
    return path if pos == env.level.exit else None


def solve_with_q_learning(level: Level, episodes: int = 300, seed: int | None = None):
    """Train on a level, return (path or None, training log)."""
    env = MazeEnv(level)
    agent = QAgent(env.n_states, env.n_actions, seed=seed)
    log = train(env, agent, episodes)
    return greedy_path(env, agent), log
