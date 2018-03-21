import logging
import threading
from collections import deque
import functools

import itertools
from multiprocess.dummy import Pool
import numpy as np
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle

from arena import Arena
from MCTS import MCTS


class Coach(object):
	"""
	This class executes the self-play + learning. It uses the functions defined
	in Game and NeuralNet. args are specified in main.py.
	"""
	
	def __init__(self, game, nnet, pnet, num_iters):
		self.game = game
		self.nnet = nnet
		self.pnet = pnet  # the competitor network
		self.num_iters = num_iters
		self.doFirstIterSelfPlay = True  # can be overwritten in loadTrainExamples()

	def learn(self,
	          num_train_episodes,
	          num_training_examples_to_keep,
	          checkpoint_folder,
	          arena_tournament_size,
	          model_update__win_threshold,
	          c_puct,
	          num_mcst_sims,
	          know_nothing_training_iters,
	          max_cpus):
		"""
		Performs numIters iterations with numEps episodes of self-play in each
		iteration. After every iteration, it retrains neural network with
		examples in train_examples (which has a maximium length of maxlenofQueue).
		It then pits the new neural network against the old one and accepts it
		only if it wins >= updateThreshold fraction of games.
		"""
		logging.info("Starting learning loop, using: {0} cores.".format(max_cpus))
		train_examples_history = []
		pool = Pool(processes=max_cpus)
		
		for i in range(self.num_iters):
			
			logging.info('ITER: ' + str(i + 1) + "/" + str(self.num_iters + 1))
			
			# Only play against yourself if either args, or if it's the second iteration
			if self.doFirstIterSelfPlay:
				logging.debug("Doing first iteration self play from configs.")
			else:
				logging.debug("Skipping first iteration self play from configs.")
			
			if self.doFirstIterSelfPlay or i > 0:
				# iteration_train_examples = deque([], maxlen=num_training_examples_per_iter)
				logging.debug("Starting {0} training episodes. Running {1} Async".format(num_train_episodes,
				                                                                         max_cpus))

				
				def self_play(game, nnet, i):
					# logging.debug("Starting MCST")
					mcts = MCTS(game=game, nnet=nnet, cpuct=c_puct, num_mcst_sims=num_mcst_sims)
					x = self.execute_episode(mcts,
											 know_nothing_training_iters=know_nothing_training_iters,
											 current_self_play_iteration=i)

					return x

				iteration_train_examples = pool.map(functools.partial(self_play, self.game, self.nnet), range(num_train_episodes))

				# save the iteration examples to the history
				logging.debug("Storing {0} training examples".format(len(iteration_train_examples)))
				train_examples_history.extend(iteration_train_examples)
			else:
				logging.debug("Skipped self play.")
			
			def save_training_examples():
				# backup history to a file
				# NB! the examples were collected using the model from the previous iteration, so (i-1)
				logging.debug("Saving model from this iteration.")
				self.save_training_examples(iteration=i+1,
			                                checkpoint_folder=checkpoint_folder,
			                                trainExamplesHistory=train_examples_history)
			
			save_training_examples_thread = threading.Thread(target=save_training_examples())
			save_training_examples_thread.start()
			
			# shuffle examples before training
			logging.debug("Flattening training examples.")
			train_examples = np.asarray(list(itertools.chain(*train_examples_history)))

			if len(train_examples) > num_training_examples_to_keep:
				logging.debug("Training examples history limit exceeded ({0}).  Removing oldest examples.".format(
						num_training_examples_to_keep))
				train_examples = np.delete(train_examples, range(len(train_examples) - num_training_examples_to_keep))

			logging.info("Shuffling {0} training examples: ".format(len(train_examples)))
			shuffle(train_examples)
			
			# training new network, keeping a copy of the old one
			logging.debug("Saving this network, loading it as previous network.")
			self.nnet.save_checkpoint(folder=checkpoint_folder, filename='temp.pth.tar')
			self.pnet.load_checkpoint(folder=checkpoint_folder, filename='temp.pth.tar')
			
			prior_mcts = MCTS(self.game, self.pnet, cpuct=c_puct, num_mcst_sims=num_mcst_sims)
			
			logging.info("Training new network using shuffled training examples.")
			self.nnet.train(train_examples)
			new_mcts = MCTS(self.game, self.nnet, cpuct=c_puct, num_mcst_sims=num_mcst_sims)
			
			# TODO:// Run this async, split into separate function.
			logging.info("Testing network against previous version.")
			player1 = lambda x: np.argmax(prior_mcts.getActionProb(x, temp=0))
			player2 = lambda x: np.argmax(new_mcts.getActionProb(x, temp=0))
			
			arena = Arena(player1, player2, self.game)
			
			pwins, nwins, draws = arena.playGames(arena_tournament_size, pool)
			save_training_examples_thread.join()
			
			print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
			if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < model_update__win_threshold:
				print('REJECTING NEW MODEL')
				self.nnet.load_checkpoint(folder=checkpoint_folder, filename='temp.pth.tar')
			else:
				print('ACCEPTING NEW MODEL')
				self.nnet.save_checkpoint(folder=checkpoint_folder, filename=self.get_examples_checkpoint_file(i))
				self.nnet.save_checkpoint(folder=checkpoint_folder, filename='best.pth.tar')

	def execute_episode(self, mcst, know_nothing_training_iters, current_self_play_iteration=0):
		"""
		This function executes one episode of self-play, starting with player 1.
		As the game is played, each turn is added as a training example to
		train_examples. The game is played till the game ends. After the game
		ends, the outcome of the game is used to assign values to each example
		in train_examples.

		It uses a temp=1 if episodeStep < tempThreshold, and thereafter
		uses temp=0.

		Returns:
			train_examples: a list of examples of the form (canonicalBoard,pi,v)
						   pi is the MCTS informed policy vector, v is +1 if
						   the player eventually won the game, else -1.
		"""
		train_examples = []
		board = self.game.getInitBoard()
		self.curPlayer = 1
		episodeStep = 0
		
		# TODO:// Add comments...
		while True:
			episodeStep += 1
			canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
			temp = int(episodeStep < know_nothing_training_iters)
			
			pi = mcst.getActionProb(canonicalBoard,
			                        temp=temp,
			                        current_self_play_iteration=current_self_play_iteration)
			sym = self.game.getSymmetries(canonicalBoard, pi)
			for b, p in sym:
				train_examples.append([b, self.curPlayer, p, None])
			
			action = np.random.choice(len(pi), p=pi)
			board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
			
			r = self.game.getGameEnded(board, self.curPlayer)
			
			if r != 0:
				return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in train_examples]
	
	def get_examples_checkpoint_file(self, iteration):
		return 'checkpoint_' + str(iteration) + '.pth.tar'
	
	def save_training_examples(self, iteration, checkpoint_folder, trainExamplesHistory):
		logging.info("Checking if checkpoint folder exists.")
		
		if not os.path.exists(checkpoint_folder):
			logging.debug("Making checkpoint folder.")
			os.makedirs(checkpoint_folder)
		
		logging.info("Saving examples to checkpoint for iter: " + str(iteration))
		filename = os.path.join(checkpoint_folder, self.get_examples_checkpoint_file(iteration) + ".examples")
		with open(filename, "wb+") as f:
			Pickler(f).dump(trainExamplesHistory)
	
	def loadTrainExamples(self):
		modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
		examplesFile = modelFile + ".examples"
		if not os.path.isfile(examplesFile):
			print(examplesFile)
			r = input("File with trainExamples not found. Continue? [y|n]")
			if r != "y":
				sys.exit()
		else:
			print("File with trainExamples found. Read it.")
			with open(examplesFile, "rb") as f:
				train_examples_history = Unpickler(f).load()
			# examples based on the model were already collected (loaded)
			self.doFirstIterSelfPlay = True

			return train_examples_history
