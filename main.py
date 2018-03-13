import logging
import sys

from game import Game
from nnet import NNet


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.NOTSET)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.NOTSET)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)s]')
    ch.setFormatter(formatter)
    root.addHandler(ch)

if __name__ == "__main__":
    setup_logging()
    logging.info("Learning Go!")
    game = Game(n=18)
    nnet = NNet(action_size=game.getActionSize(), board_size_x=18, board_size_y=18)

