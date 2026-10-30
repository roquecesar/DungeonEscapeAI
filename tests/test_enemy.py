from dungeon.entities import Enemy, EnemyState, Player
from dungeon.level import Level

CORRIDOR = ["#" * 22,
            "#" + "." * 20 + "#",
            "#" * 22]


def make_level():
    return Level.from_strings(CORRIDOR)


def test_enemy_starts_in_patrol():
    assert Enemy((1, 1)).state is EnemyState.PATROL


def test_patrol_to_chase_when_player_close():
    level = make_level()
    enemy = Enemy((5, 1), patrol_points=[(5, 1), (8, 1)], detect_radius=4)
    enemy.update(level, (7, 1))
    assert enemy.state is EnemyState.CHASE


def test_patrol_stays_patrol_when_player_far():
    level = make_level()
    enemy = Enemy((2, 1), patrol_points=[(2, 1), (4, 1)], detect_radius=4)
    enemy.update(level, (19, 1))
    assert enemy.state is EnemyState.PATROL


def test_chase_to_return_when_player_escapes():
    level = make_level()
    enemy = Enemy((2, 1), patrol_points=[(2, 1)], detect_radius=4, lose_radius=6)
    enemy.state = EnemyState.CHASE
    enemy.pos = (5, 1)
    enemy.update(level, (19, 1))
    assert enemy.state is EnemyState.RETURN


def test_return_to_patrol_at_home():
    level = make_level()
    enemy = Enemy((2, 1), patrol_points=[(2, 1)], detect_radius=3)
    enemy.state = EnemyState.RETURN
    enemy.update(level, (19, 1))
    assert enemy.state is EnemyState.PATROL


def test_chase_moves_toward_player():
    level = make_level()
    enemy = Enemy((5, 1), detect_radius=10)
    enemy.update(level, (9, 1))
    assert enemy.pos == (6, 1)


def test_player_cannot_walk_into_wall():
    level = make_level()
    player = Player((1, 1))
    assert not player.move(0, -1, level)
    assert player.pos == (1, 1)
    assert player.move(1, 0, level)
    assert player.pos == (2, 1)
