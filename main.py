import logging
import sys
from os import cpu_count

from config import Config
from game import Game
from nnet import NNet
from coach import Coach

# TODO:// These are much lower params than the paper describes.

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
	
	configs = Config().get_args()
	
	board_size = configs["board_size"]

	num_threads = (cpu_count()-1) if configs["num_threads"] == 0 else int(configs["num_threads"])

	game = Game(n=board_size)
	nnet = NNet(action_size=game.getActionSize(), board_size_x=board_size, board_size_y=board_size)
	pnet = NNet(action_size=game.getActionSize(), board_size_x=board_size, board_size_y=board_size)
	coach = Coach(game=game, nnet=nnet, pnet=pnet, num_iters=args["num_iters"])
	if configs["load_model"]:
		logging.info("Loading training examples")
		coach.loadTrainExamples()
	
	coach.learn(num_train_episodes=configs["num_epsisodes"],
	            num_training_examples_to_keep=configs["maxlenOfQueue"],
	            num_training_examples_per_iter=configs["num_iters"],
	            checkpoint_folder=configs["checkpoint_dir"],
	            arena_tournament_size=configs["arena_size"],
	            model_update__win_threshold=configs["updateThreshold"],
	            num_mcst_sims=configs["numMCTSSims"],
	            c_puct=configs["c_puct"],
	            know_nothing_training_iters=configs["tempThreshold"],
	            max_cpus=num_threads)
