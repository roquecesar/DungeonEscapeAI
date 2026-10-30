import pytest

from dungeon.level import Level
from dungeon.pathfinding import astar, manhattan

OPEN = ["#######",
        "#S...E#",
        "#.....#",
        "#######"]
WALLED = ["#######",
          "#S#..E#",
          "#.#...#",
          "#######"]


def test_manhattan():
    assert manhattan((0, 0), (3, 4)) == 7


def test_astar_finds_shortest_path():
    level = Level.from_strings(OPEN)
    path = astar(level, level.start, level.exit)
    assert path[0] == level.start and path[-1] == level.exit
    assert len(path) - 1 == manhattan(level.start, level.exit)


def test_astar_returns_none_when_blocked():
    level = Level.from_strings(WALLED)
    assert astar(level, level.start, level.exit) is None


def test_astar_start_equals_goal():
    level = Level.from_strings(OPEN)
    assert astar(level, level.start, level.start) == [level.start]


def test_astar_rejects_wall_target():
    level = Level.from_strings(OPEN)
    assert astar(level, level.start, (0, 0)) is None


def test_path_steps_are_adjacent_and_walkable():
    level = Level.generate(15, 15, seed=3)
    path = astar(level, level.start, level.exit)
    assert path is not None
    for a, b in zip(path, path[1:]):
        assert manhattan(a, b) == 1
        assert level.is_walkable(*b)


@pytest.mark.parametrize("seed", range(5))
def test_generated_maze_always_solvable(seed):
    level = Level.generate(21, 21, seed=seed)
    assert astar(level, level.start, level.exit) is not None
