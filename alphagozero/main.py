#original code https://github.com/suragnair/alpha-zero-general

import logging
import sys
from os import cpu_count

from alphagozero.config import Config
from alphagozero.game import Game
from alphagozero.nnet import NNet
from alphagozero.coach import Coach

# TODO:// These are much lower params than the paper describes.
from web import WebServer


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
    sys.setrecursionlimit(2500) # arbitrary
    configs = Config().get_args()

    num_threads: int  = ((cpu_count()-1) # type: ignore 
        if configs.num_threads == 0 
        else int(configs.num_threads if configs.num_threads is not None else 0))

    game = Game(n=configs.board_size)
    nnet = NNet(action_size=game.getActionSize(),
                board_size=configs.board_size,
                learning_rate=configs.learning_rate,
                dropout_rate=configs.dropout_rate,
                epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                num_channels=configs.num_channels,
                log_device_placement=configs.log_device_placement,
                network_architecture = configs.network_architecture)

    pnet = NNet(action_size=game.getActionSize(),
                board_size=configs.board_size,
                learning_rate=configs.learning_rate,
                dropout_rate=configs.dropout_rate,
                epochs=configs.num_epochs,
                batch_size=configs.batch_size,
                num_channels=configs.num_channels,
                log_device_placement=configs.log_device_placement,
                network_architecture = configs.network_architecture)
	
    coach = Coach(game=game,
                  nnet=nnet,
                  pnet=pnet,
                  num_iters=configs.num_iters,
                  root_noise = configs.root_noise,
                  board_size = configs.board_size)
    if configs.load_model:
        logging.info("Loading training examples")
        coach.loadTrainExamples()

    if configs.web_server:
        web = WebServer(game=game,
                        nnet=nnet,
                        checkpoint_folder=configs.checkpoint_dir,
                        c_puct=configs.c_puct,
                        num_mcst_sims=configs.num_mcts_sims)
        web.start_web_server()
        exit(0)

    coach.learn(num_train_episodes=configs.num_episodes,
                num_training_examples_to_keep=configs.maxlenOfQueue,
                checkpoint_folder=configs.checkpoint_dir,
                arena_tournament_size=configs.arena_size,
                model_update__win_threshold=configs.update_threshold,
                num_mcst_sims=configs.num_mcts_sims,
                c_puct=configs.c_puct,
                know_nothing_training_iters=configs.tempThreshold,
                max_cpus=num_threads)
