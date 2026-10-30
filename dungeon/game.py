"""Pygame front-end: the Game class ties Level, Player and Enemy together."""
from __future__ import annotations

import random

import pygame

from .entities import Enemy, EnemyState, Player
from .level import Level, Pos
from .rl import solve_with_q_learning

TILE = 28
HUD_H = 40
COLORS = {
    "bg": (18, 18, 24), "wall": (52, 58, 82), "floor": (30, 32, 44),
    "exit": (80, 200, 120), "player": (90, 170, 255),
    "hint": (255, 255, 255), "text": (230, 230, 240),
}
ENEMY_COLOR = {
    EnemyState.PATROL: (230, 190, 70),
    EnemyState.CHASE: (230, 70, 70),
    EnemyState.RETURN: (170, 120, 220),
}
KEY_MOVES = {
    pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
    pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
    pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
    pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
}


class Game:
    """Owns the window, the main loop, and the current level."""

    def __init__(self, width: int = 21, height: int = 21, n_enemies: int = 3,
                 seed: int | None = None) -> None:
        pygame.init()
        self.width, self.height, self.n_enemies = width | 1, height | 1, n_enemies
        self.seed = seed
        self.screen = pygame.display.set_mode(
            (self.width * TILE, self.height * TILE + HUD_H))
        pygame.display.set_caption("Dungeon Escape AI")
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()
        self.level_number = 0
        self.status = "playing"  # playing | won | caught
        self.hint_path: list[Pos] = []
        self.new_level()

    def new_level(self) -> None:
        self.level_number += 1
        level_seed = None if self.seed is None else self.seed + self.level_number
        self.level = Level.generate(self.width, self.height, seed=level_seed)
        self.player = Player(self.level.start)
        self.enemies = self._spawn_enemies()
        self.status = "playing"
        self.hint_path = []

    def _spawn_enemies(self) -> list[Enemy]:
        rng = random.Random(self.seed)
        far = [c for c in self.level.floor_cells()
               if abs(c[0] - self.level.start[0]) + abs(c[1] - self.level.start[1]) > 10]
        rng.shuffle(far)
        enemies = []
        for i in range(min(self.n_enemies, len(far) // 3)):
            home = far[i * 3]
            enemies.append(Enemy(home, patrol_points=[home, far[i * 3 + 1], far[i * 3 + 2]]))
        return enemies

    def show_ai_hint(self) -> None:
        """Train a Q-learning agent on this maze and display its solution path."""
        path, _ = solve_with_q_learning(self.level, episodes=400, seed=1)
        self.hint_path = path or []

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    if self.status == "caught":
                        self.level_number = 0
                    self.new_level()
                elif event.key == pygame.K_h and self.status == "playing":
                    self.show_ai_hint()
                elif event.key in KEY_MOVES and self.status == "playing":
                    self.tick(*KEY_MOVES[event.key])
        return True

    def tick(self, dx: int, dy: int) -> None:
        """One turn: the player moves, then each enemy moves."""
        if not self.player.move(dx, dy, self.level):
            return
        if self.player.pos == self.level.exit:
            self.status = "won"
            return
        for enemy in self.enemies:
            enemy.update(self.level, self.player.pos)
            if enemy.pos == self.player.pos:
                self.status = "caught"
                return

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()

    def _rect(self, pos: Pos, inset: int = 0) -> pygame.Rect:
        return pygame.Rect(pos[0] * TILE + inset, pos[1] * TILE + HUD_H + inset,
                           TILE - 2 * inset, TILE - 2 * inset)

    def draw(self) -> None:
        self.screen.fill(COLORS["bg"])
        for y in range(self.level.height):
            for x in range(self.level.width):
                color = COLORS["floor"] if self.level.grid[y][x] == 0 else COLORS["wall"]
                pygame.draw.rect(self.screen, color, self._rect((x, y)))
        for cell in self.hint_path:
            pygame.draw.rect(self.screen, COLORS["hint"], self._rect(cell, 11))
        pygame.draw.rect(self.screen, COLORS["exit"], self._rect(self.level.exit, 3))
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, ENEMY_COLOR[enemy.state],
                             self._rect(enemy.pos, 4), border_radius=6)
        pygame.draw.rect(self.screen, COLORS["player"],
                         self._rect(self.player.pos, 4), border_radius=6)
        msg = {
            "playing": f"Level {self.level_number} | Arrows/WASD move  H = AI hint  R = new maze",
            "won": "You escaped!  Press R for the next maze",
            "caught": "Caught!  Press R to restart",
        }[self.status]
        self.screen.blit(self.font.render(msg, True, COLORS["text"]), (8, 10))
