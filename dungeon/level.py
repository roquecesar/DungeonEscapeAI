"""Grid level: procedural maze generation and walkability helpers."""
from __future__ import annotations

import random
from typing import Iterator

Pos = tuple[int, int]  # (x, y)
FLOOR, WALL = 0, 1

# Index in this tuple is the action id used by the RL code.
DIRECTIONS: tuple[Pos, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))  # up, down, left, right


class Level:
    """A rectangular grid of floor and wall tiles."""

    def __init__(self, grid: list[list[int]], start: Pos, exit_: Pos) -> None:
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        self.start = start
        self.exit = exit_

    @classmethod
    def from_strings(cls, rows: list[str]) -> "Level":
        """Build a level from text: '#' wall, '.' floor, 'S' start, 'E' exit."""
        grid: list[list[int]] = []
        start, exit_ = (1, 1), (1, 1)
        for y, row in enumerate(rows):
            line = []
            for x, ch in enumerate(row):
                if ch == "S":
                    start = (x, y)
                if ch == "E":
                    exit_ = (x, y)
                line.append(WALL if ch == "#" else FLOOR)
            grid.append(line)
        return cls(grid, start, exit_)

    @classmethod
    def generate(cls, width: int = 21, height: int = 21,
                 seed: int | None = None, loop_chance: float = 0.08) -> "Level":
        """Procedurally generate a maze (randomized DFS / recursive backtracker).

        Width and height are forced odd. Some extra walls are removed afterwards
        so enemies have alternate routes.
        """
        if width < 5 or height < 5:
            raise ValueError("maze must be at least 5x5")
        width |= 1
        height |= 1
        rng = random.Random(seed)
        grid = [[WALL] * width for _ in range(height)]

        grid[1][1] = FLOOR
        stack: list[Pos] = [(1, 1)]
        while stack:
            x, y = stack[-1]
            options = []
            for dx, dy in DIRECTIONS:
                nx, ny = x + 2 * dx, y + 2 * dy
                if 0 < nx < width - 1 and 0 < ny < height - 1 and grid[ny][nx] == WALL:
                    options.append((dx, dy))
            if not options:
                stack.pop()
                continue
            dx, dy = rng.choice(options)
            grid[y + dy][x + dx] = FLOOR
            grid[y + 2 * dy][x + 2 * dx] = FLOOR
            stack.append((x + 2 * dx, y + 2 * dy))

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if grid[y][x] != WALL:
                    continue
                horizontal = grid[y][x - 1] == FLOOR and grid[y][x + 1] == FLOOR
                vertical = grid[y - 1][x] == FLOOR and grid[y + 1][x] == FLOOR
                if (horizontal or vertical) and rng.random() < loop_chance:
                    grid[y][x] = FLOOR

        return cls(grid, start=(1, 1), exit_=(width - 2, height - 2))

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.grid[y][x] == FLOOR

    def neighbors(self, pos: Pos) -> Iterator[Pos]:
        x, y = pos
        for dx, dy in DIRECTIONS:
            if self.is_walkable(x + dx, y + dy):
                yield (x + dx, y + dy)

    def floor_cells(self) -> list[Pos]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.grid[y][x] == FLOOR]
