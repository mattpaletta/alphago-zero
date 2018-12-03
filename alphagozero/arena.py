#original code https://github.com/suragnair/alpha-zero-general

import numpy as np
import logging


class Arena(object):
	"""
	An Arena class where any 2 agents can be pit against each other.
	"""
	def __init__(self, player1, player2, game, display=None):
		"""
		Input:
			player 1,2: two functions that takes board as input, return action
			game: Game object
			display: a function that takes board as input and prints it (e.g.
					 display in othello/OthelloGame). Is necessary for verbose
					 mode.

		see othello/OthelloPlayers.py for an example. See pit.py for pitting
		human players/other baselines with each other.
		"""
		self.player1 = player1
		self.player2 = player2
		self.game = game
		self.display = display

	def playGame(self, verbose=False):
		"""
		Executes one episode of a game.

		Returns:
			either
				winner: player who won the game (1 if player1, -1 if player2)
			or
				draw result returned from the game that is neither 1, -1, nor 0.
		"""
		players = [self.player2, None, self.player1]
		curPlayer = 1
		board = self.game.getInitBoard()
		it = 0
		while self.game.getGameEnded(board, curPlayer) == 0:
			it+=1
			if verbose:
				assert(self.display)
				print("Turn ", str(it), "Player ", str(curPlayer))
				self.display(board)
			action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))

			valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer),1)

			if valids[action]==0:
				print(action)
				assert valids[action] >0
			board, curPlayer = self.game.getNextState(board, curPlayer, action)
		if verbose:
			assert(self.display)
			print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
			self.display(board)
		logging.info("Game completed")
		return self.game.getGameEnded(board, 1)


	def playGames(self, num, pool, verbose=False):
		"""
		Plays num games in which player1 starts num/2 games and player2 starts
		num/2 games.

		Returns:
			oneWon: games won by player1
			twoWon: games won by player2
			draws:  games won by nobody
		"""
		#eps = 0
		maxeps = int(num)

		num = int(num / 2)
		#oneWon = 0
		#twoWon = 0
		#draws = 0
		
		def run_arena(i):
			oneWon = 0
			twoWon = 0
			draws = 0
			eps = 0
			
			gameResult = self.playGame(verbose=verbose)
			if gameResult == -1:
				oneWon += 1
			elif gameResult == 1:
				twoWon += 1
			else:
				draws += 1
			# bookkeeping + plot progress
			eps += 1
			
			# Instead of switching the player1 and player2 objects, just do it implicitly
			if i % 2 == 0:
				return [oneWon, twoWon, draws, eps]
			else:
				return [twoWon, oneWon, draws, eps]
			
		results = pool.map(run_arena, range(num*2))
		
		oneWon, twoWon, draws, eps = np.asarray(results).sum(axis=0)
		print(oneWon, twoWon, draws, eps)
		
		#self.player1, self.player2 = self.player2, self.player1
		"""
		# TODO:// So can this...
		#for _ in range(num):
		def run_arena(i):
			oneWon = twoWon = draws = eps = 0
			gameResult = self.playGame(verbose=verbose)
			if gameResult == -1:
				oneWon += 1
			elif gameResult == 1:
				twoWon += 1
			else:
				draws += 1
			# bookkeeping + plot progress
			eps += 1
			
			return oneWon, twoWon, draws, eps
	
		results = pool.map(run_arena2, range(num))
		"""
		
		return oneWon, twoWon, draws
