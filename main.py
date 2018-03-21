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
	
	board_size = configs.board_size

	num_threads = 1 #(cpu_count()-1) if configs.num_threads == 0 else int(configs.num_threads)

	game = Game(n=board_size)
	nnet = NNet(action_size=game.getActionSize(),
	            board_size=board_size,
	            learning_rate=configs.learning_rate,
	            dropout_rate=configs.dropout_rate,
	            epochs=configs.num_epochs,
				batch_size=configs.batch_size,
	            num_channels=configs.num_channels,
	            log_device_placement=configs.log_device_placement)
	
	pnet = NNet(action_size=game.getActionSize(),
	            board_size=board_size,
	            learning_rate=configs.learning_rate,
	            dropout_rate=configs.dropout_rate,
	            epochs=configs.num_epochs,
				batch_size=configs.batch_size,
	            num_channels=configs.num_channels,
	            log_device_placement=configs.log_device_placement)
	
	coach = Coach(game=game,
	              nnet=nnet,
	              pnet=pnet,
	              num_iters=configs.num_iters)
	if configs.load_model:
		logging.info("Loading training examples")
		coach.loadTrainExamples()
	
	coach.learn(num_train_episodes=configs.num_episodes,
	            num_training_examples_to_keep=configs.maxlenOfQueue,
	            checkpoint_folder=configs.checkpoint_dir,
	            arena_tournament_size=configs.arena_size,
	            model_update__win_threshold=configs.update_threshold,
	            num_mcst_sims=configs.num_mcts_sims,
	            c_puct=configs.c_puct,
	            know_nothing_training_iters=configs.tempThreshold,
	            max_cpus=num_threads)
