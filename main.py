"""Run the game:  python main.py"""
from dungeon.game import Game

if __name__ == "__main__":
    Game(width=21, height=21, n_enemies=3).run()
