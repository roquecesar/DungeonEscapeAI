import numpy as np
import pandas as pd

from dungeon.level import Level
from dungeon.rl import MazeEnv, QAgent, greedy_path, train

SMALL = ["#######",
         "#S...E#",
         "#.###.#",
         "#.....#",
         "#######"]


def test_env_goal_reward_and_done():
    level = Level.from_strings(["#####", "#SE.#", "#####"])
    env = MazeEnv(level)
    env.reset()
    _, reward, done = env.step(3)  # right
    assert done and reward == MazeEnv.GOAL_REWARD


def test_env_wall_penalty_keeps_position():
    level = Level.from_strings(SMALL)
    env = MazeEnv(level)
    env.reset()
    _, reward, done = env.step(0)  # up into wall
    assert reward == MazeEnv.WALL_PENALTY and not done
    assert env.pos == level.start


def test_q_update_moves_value_toward_target():
    agent = QAgent(4, 4, alpha=0.5, gamma=0.9)
    agent.update(0, 1, 1.0, 2, True)
    assert agent.q[0, 1] == 0.5


def test_epsilon_decays_but_not_below_min():
    agent = QAgent(2, 2, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == 0.1


def test_training_log_is_dataframe_with_expected_columns():
    env = MazeEnv(Level.from_strings(SMALL))
    agent = QAgent(env.n_states, env.n_actions, seed=0)
    log = train(env, agent, episodes=20)
    assert isinstance(log, pd.DataFrame) and len(log) == 20
    assert {"episode", "reward", "steps", "epsilon", "success"} <= set(log.columns)


def test_agent_learns_to_solve_small_maze():
    env = MazeEnv(Level.from_strings(SMALL))
    agent = QAgent(env.n_states, env.n_actions, seed=0)
    log = train(env, agent, episodes=300)
    assert greedy_path(env, agent) is not None
    assert log["reward"].tail(50).mean() > log["reward"].head(50).mean()


def test_q_table_shape():
    env = MazeEnv(Level.generate(9, 9, seed=1))
    agent = QAgent(env.n_states, env.n_actions)
    assert agent.q.shape == (81, 4) and np.all(agent.q == 0)
