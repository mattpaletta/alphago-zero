import logging
import sys

from game import Game
from nnet import NNet
from coach import Coach

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


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
    nnet = NNet(action_size=game.getActionSize(), board_size_x=19, board_size_y=19)
    c = Coach(game, nnet, )
