"""Game entities: Player and Enemy (finite state machine: patrol, chase, return)."""
from __future__ import annotations

from enum import Enum

from .level import Level, Pos
from .pathfinding import astar


class Player:
    def __init__(self, pos: Pos) -> None:
        self.pos = pos

    def move(self, dx: int, dy: int, level: Level) -> bool:
        """Try to move one tile. Returns True if the move happened."""
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if level.is_walkable(nx, ny):
            self.pos = (nx, ny)
            return True
        return False


class EnemyState(Enum):
    PATROL = "patrol"
    CHASE = "chase"
    RETURN = "return"


class Enemy:
    """Enemy with a three-state machine.

    PATROL -> CHASE   when the player is within `detect_radius` path steps
    CHASE  -> RETURN  when the player is farther than `lose_radius` (or unreachable)
    RETURN -> PATROL  when the enemy is back at its home tile
    """

    def __init__(self, pos: Pos, patrol_points: list[Pos] | None = None,
                 detect_radius: int = 6, lose_radius: int = 10) -> None:
        self.pos = pos
        self.home = pos
        self.patrol_points = list(patrol_points) if patrol_points else [pos]
        self.detect_radius = detect_radius
        self.lose_radius = lose_radius
        self.state = EnemyState.PATROL
        self.path: list[Pos] = []
        self._patrol_index = 0

    def _distance_to(self, level: Level, target: Pos) -> int | None:
        path = astar(level, self.pos, target)
        return None if path is None else len(path) - 1

    def _update_state(self, level: Level, player_pos: Pos) -> None:
        dist = self._distance_to(level, player_pos)
        if self.state in (EnemyState.PATROL, EnemyState.RETURN):
            if dist is not None and dist <= self.detect_radius:
                self.state = EnemyState.CHASE
                self.path = []
        elif self.state is EnemyState.CHASE:
            if dist is None or dist > self.lose_radius:
                self.state = EnemyState.RETURN
                self.path = []
        if self.state is EnemyState.RETURN and self.pos == self.home:
            self.state = EnemyState.PATROL
            self.path = []

    def update(self, level: Level, player_pos: Pos) -> None:
        """Advance the enemy by one tile (one game tick)."""
        self._update_state(level, player_pos)

        if self.state is EnemyState.CHASE:
            path = astar(level, self.pos, player_pos)  # re-plan every tick
            self.path = path[1:] if path else []
        elif self.state is EnemyState.RETURN:
            if not self.path:
                path = astar(level, self.pos, self.home)
                self.path = path[1:] if path else []
        else:  # PATROL
            if not self.path:
                self._patrol_index = (self._patrol_index + 1) % len(self.patrol_points)
                target = self.patrol_points[self._patrol_index]
                path = astar(level, self.pos, target)
                self.path = path[1:] if path else []

        if self.path:
            self.pos = self.path.pop(0)
