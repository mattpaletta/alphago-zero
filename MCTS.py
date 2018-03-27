import logging
import math
from multiprocessing.pool import Pool

from config import Config

import numpy as np

FIRST_PLAYER = 1
SECOND_PLAYER = 2
GAME_ENDED = 0
EPS = 1e-8


class MCTS(object):

	root_node = None

	#keeps track of whose turn it is
	friendly_turn = -1

	def __init__(self, game, nnet, num_mcst_sims, cpuct):
		self.game = game
		self.nnet = nnet
		self.num_mcst_sims = num_mcst_sims
		self.game_action_size = self.game.getActionSize()
		self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
		self.Nsa = {}  # stores #times edge s,a was visited
		self.Ns = {}  # stores #times board s was visited
		self.Ps = {}  # stores initial policy (returned by neural net)
		
		self.Es = {}  # stores game.getGameEnded ended for board s
		self.Vs = {}  # stores game.getValidMoves for board s
		self.num_every_move_valid = 0
		self.c_puct = cpuct
		
	def getActionProb(self, canonicalBoard, temp=1, current_self_play_iteration=0):
		"""
		This function performs numMCTSSims simulations of MCTS starting from
		canonicalBoard.
		Returns:
			probs: a policy vector where the probability of the ith action is
				   proportional to Nsa[(s,a)]^(1./temp)
		"""
		
		# TODO:// Run these all on separate threads
		for i in range(self.num_mcst_sims):
			#logging.info("Starting MCST simulation: {0}/{1}:{2}".format(i+1, self.num_mcst_sims, current_self_play_iteration))
			self.search(canonicalBoard)
		
		s = self.game.stringRepresentation(canonicalBoard)
		counts = np.asarray([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game_action_size)])
		
		# No knowledge from before, so just go to the one with the most visits
		if temp == 0:
			logging.info("No knowledge, choosing action with most visits.")
			best_action = np.argmax(counts)
			probs = np.zeros_like(counts)
			probs[best_action] = 1
		else:
			logging.info("Scaling counts for actions.")
			counts = counts ** (1. / temp)
			probs = counts / float(np.sum(counts))
		
		return probs
	
	def search(self, canonical_board):
		"""
		This function performs one iteration of MCTS. It is recursively called
		till a leaf node is found. The action chosen at each node is one that
		has the maximum upper confidence bound as in the paper.
		Once a leaf node is found, the neural network is called to return an
		initial policy P and a value board_value for the state. This value is propogated
		up the search path. In case the leaf node is a terminal state, the
		outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
		updated.
		NOTE: the return values are the negative of the value of the current
		state. This is done since board_value is in [-1,1] and if board_value is the value of a
		state for the current player, then its value is -board_value for the other player.
		Returns:
			board_value: the negative of the value of the current canonical_board
		"""
		
		board_string = self.game.stringRepresentation(canonical_board)
		
		if board_string not in self.Es:
			self.Es[board_string] = self.game.getGameEnded(board=canonical_board, player=FIRST_PLAYER)
		if self.Es[board_string] != GAME_ENDED:
			# terminal node
			return -self.Es[board_string]
		
		if board_string not in self.Ps:
			logging.debug("Reached leaf node!")
			# The neural network only accept boards of (1, board_size, board_size, 1), so reshape it in numpy.
			# TODO:// This should be (board_size, board_size, 17)
			# TODO:// The first 16 are the current board and the last 7 for each player
			# TODO:// The final one is 1 or -1 indicating who's turn it is.
			board_size = Config().get_args().board_size
	
			np_canonical_board = np.asarray(canonical_board).reshape(board_size, board_size, 1)
			# TODO:// Lock search thread before getting value from NN.
			
			action_prob, board_value = self.nnet.predict(np_canonical_board)
			self.Ps[board_string] = action_prob
			
			valids = self.game.getValidMoves(board=canonical_board, player=FIRST_PLAYER)
			self.Ps[board_string] = self.Ps[board_string] * valids  # masking invalid moves
			sum_Ps_s = np.sum(self.Ps[board_string])
			if sum_Ps_s > 0:
				self.Ps[board_string] /= sum_Ps_s  # re-normalize
			else:
				logging.debug("Every move valid!")
				self.num_every_move_valid += 1
				# if all valid moves were masked make all valid moves equally probable
				
				# NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get
				# overfitting or something else.
				# If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or
				# training process.
				logging.info("All valid moves were masked, do workaround.")
				self.Ps[board_string] = self.Ps[board_string] + valids
				self.Ps[board_string] /= np.sum(self.Ps[board_string])
				
				if self.num_every_move_valid % 12 == 0 or self.num_every_move_valid % 100 == 0:
					logging.warning("All valid moves were masked {0} times. Check NNet.".format(self.num_every_move_valid))
			
			self.Vs[board_string] = valids
			self.Ns[board_string] = 0
			return -board_value
		try:
			valids = self.Vs[board_string]
		except KeyError:
			logging.warning("Manually adding valid moves.")
			self.Vs[board_string] = self.game.getValidMoves(board=canonical_board, player=FIRST_PLAYER)
			valids = self.Vs[board_string]
		
		cur_best = -float('inf')
		best_act = -1
		
		logging.debug("Choosing move with highest confidence")
		# pick the action with the highest upper confidence bound

		# TODO:// What do these equations do?...
		for a in range(self.game.getActionSize()):
			if valids[a]:
				if (board_string, a) in self.Qsa:
					u = self.Qsa[(board_string, a)] + self.c_puct * self.Ps[board_string][a] * math.sqrt(self.Ns[board_string]) / (
								1 + self.Nsa[(board_string, a)])
				else:
					u = self.c_puct * self.Ps[board_string][a] * math.sqrt(self.Ns[board_string] + EPS)  # Q = 0 ?
				
				if u > cur_best:
					cur_best = u
					best_act = a
		
		a = best_act
		next_s, next_player = self.game.getNextState(canonical_board, 1, a)
		next_s = self.game.getCanonicalForm(next_s, next_player)
		
		board_value = self.search(next_s)
		
		if (board_string, a) in self.Qsa:
			self.Qsa[(board_string, a)] = (self.Nsa[(board_string, a)] * self.Qsa[(board_string, a)] + board_value) / (self.Nsa[(board_string, a)] + 1)
			self.Nsa[(board_string, a)] += 1
		
		else:
			self.Qsa[(board_string, a)] = board_value
			self.Nsa[(board_string, a)] = 1
		
		self.Ns[board_string] += 1
		return -board_value
	
	# TODO:// Add this code back in, once other parts are working...
	"""
	# recursively selects hghest value child at each level
	def select(self, node):
		friendly_turn *= -1 #alternates turns going down
		if len(node.children) != 0:
			max_u = 0
			max_node = None
			for x in range(len(node.children)):
				if(node.children[x].U > max_u):
					max_u = node.children[x].U
					max_node = node.children[x]
			return select(max_node)
		#if it doesn't have children kill the recursion
		else:
			return node


	#creates all possible children from current state
	#if this isnt a terminal board position
	def expand(self, node, board_layout):
		if(determine_if_terminal(board_layout) == 0):
			#for each board reachable by the current board
			possible_outcomes = return_value_from_network() #array of adjacent board positions
			for x in range(len(possible_outcomes)):
				child = Node(node, possible_outcomes[x], friendly_turn)
				node.children.append(child)
				update(child)

	#propogates new values up the network
	def update(self, node):
		if(node != root_node):
			node.calculateU()
			update(node.parent)

	#returns -1 if lost from position, 1 if won. o otherwise
	def determine_if_terminal(self, board_layout):
		return False

	def push_value_to_network(self, type, node, value):
		return 1

	def return_value_from_network(self):     #replace this with values from network
		return 1


	#the function. assumes it always starts out on friendly turn
	def find_optimal_path(self, board_layout):
		root_node = Node(None, board_layout, -1)
		while(searching):
			friendly_turn = -1
			temp_node = select(root_node)
			expand(temp_node, temp_node.board)
		most_explore_count = 0
		most_explored = None
		for x in range(len(root_node.children)):
			if(root_node.children[x].N > most_explore_count):
				most_explore_count = root_node.children[x].N
				most_explored = root_node.children[x]
		for x in range(len(root_node.children)):
			if(root_node.children[x]==most_explored):
				self.push_value_to_network("P", root_node.children[x].board, 1)
			else:
				self.push_value_to_network("P", root_node.children[x].board, 0)
		return most_explored.board    #could return position of next move instead?
	"""