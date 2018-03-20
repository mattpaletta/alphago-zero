import logging
import sys

from game import Game
from nnet import NNet
from coach import Coach

# TODO:// These are much lower params than the paper describes.
# TODO:// Turn these into command line arguments with defaults.
args = {
	'num_iters': 3,
	'numEps': 2,
	'tempThreshold': 15,
	'updateThreshold': 0.6,
	'maxlenOfQueue': 10,
	'numMCTSSims': 5,
	'arenaCompare': 40,
	'cpuct': 1,
	
	'checkpoint': './checkpoints/',
	'load_model': False,
	'load_folder_file': ('models/8x100x50', 'best.pth.tar'),
	'numItersForTrainExamplesHistory': 20,
}


def setup_logging():
	root = logging.getLogger()
	root.setLevel(logging.INFO)
	
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)s]')
	ch.setFormatter(formatter)
	root.addHandler(ch)


if __name__ == "__main__":
	setup_logging()
	logging.info("Learning Go!")
	board_size = 19
	
	game = Game(n=board_size)
	nnet = NNet(action_size=game.getActionSize(), board_size_x=board_size, board_size_y=board_size)
	pnet = NNet(action_size=game.getActionSize(), board_size_x=board_size, board_size_y=board_size)
	coach = Coach(game=game, nnet=nnet, pnet=pnet, num_iters=args["num_iters"])
	if args["load_model"]:
		logging.info("Loading training examples")
		coach.loadTrainExamples()
	coach.learn(num_train_episodes=args["numEps"],
	            num_training_examples_to_keep=args["maxlenOfQueue"],
				num_training_examples_per_iter=args["num_iters"],
				checkpoint_folder=args["checkpoint"],
				arena_model_size=args["arenaCompare"],
	            model_update__win_threshold=args["updateThreshold"],
	            num_mcst_sims=args["numMCTSSims"],
	            cpuct=args["cpuct"],
	            know_nothing_training_iters=args["tempThreshold"],
	            max_cpus=4)
