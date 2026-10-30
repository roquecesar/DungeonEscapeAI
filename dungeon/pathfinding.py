"""A* search on a grid level using a heap-based priority queue (heapq)."""
from __future__ import annotations

import heapq
import itertools

from .level import Level, Pos


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(level: Level, start: Pos, goal: Pos) -> list[Pos] | None:
    """Return the shortest path from start to goal (inclusive), or None.

    4-directional movement, unit cost, Manhattan heuristic (admissible here,
    so the path is optimal).
    """
    if not level.is_walkable(*start) or not level.is_walkable(*goal):
        return None
    if start == goal:
        return [start]

    counter = itertools.count()  # tie-breaker so the heap never compares positions
    open_heap: list[tuple[int, int, Pos]] = [(manhattan(start, goal), next(counter), start)]
    g_cost: dict[Pos, int] = {start: 0}
    came_from: dict[Pos, Pos] = {}
    closed: set[Pos] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        closed.add(current)

        for nxt in level.neighbors(current):
            if nxt in closed:
                continue
            new_g = g_cost[current] + 1
            if new_g < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = new_g
                came_from[nxt] = current
                heapq.heappush(open_heap, (new_g + manhattan(nxt, goal), next(counter), nxt))
    return None
