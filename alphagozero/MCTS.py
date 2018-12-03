#original code https://github.com/suragnair/alpha-zero-general


import logging
import math
from multiprocessing.pool import Pool

import sys

from alphagozero.game import Game
from alphagozero.nnet import NNet

import numpy as np

FIRST_PLAYER = 1
SECOND_PLAYER = 2
GAME_ENDED = 0
EPS = 1e-8


class MCTS(object):

    root_node = None

    #keeps track of whose turn it is
    friendly_turn = -1

    def __init__(self, game: Game, nnet: NNet, num_mcst_sims: int, cpuct: int, root_noise: bool, board_size: int):
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
        self.root_noise = root_noise
        self.board_size = board_size
    
    def getActionProb(self, canonicalBoard, temp=1, current_self_play_iteration=0):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]^(1./temp)
        """
        
        for i in range(self.num_mcst_sims):
            logging.debug("Starting MCST simulation: {0}/{1}:{2}".format(i+1, self.num_mcst_sims, current_self_play_iteration))
            self.search(canonicalBoard, root_noise=self.root_noise)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = np.asarray([self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game_action_size)])

        # No knowledge from before, so just go to the one with the most visits
        if temp == 0:
            logging.debug("No knowledge, choosing action with most visits.")
            best_action = np.argmax(counts)
            probs = np.zeros_like(counts)
            probs[best_action] = 1
        else:
            logging.debug("Scaling counts for actions.")
            counts = counts ** (1. / temp)
            probs = counts / float(np.sum(counts))

        return probs
    
    def search(self, canonical_board, root_noise=False, root_noise_epsilon=0.25, root_noise_dirichlet=0.03):
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

        Params:
            root_noise:    Additional exploration is achieved by adding Dirichlet
                noise to the prior probabilities in the root node s0 ,
                specifically P(s, a) =(1 −​  ε)p_a  +​  εη_a , 
                where η ∼​  Dir(0.03) and ε =​ 0.25; this noise ensures that all
                moves may be tried, but the search may still overrule bad moves.
            root_noise_epsilon: The value to be used for a random path choice
                defaults 0.25, as given in paper
            root_noise_dirichlet: Param for the noise function. Defaults 0.03,
                as given in the paper


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
    
            np_canonical_board = np.asarray(canonical_board).reshape(self.board_size, self.board_size, 1)
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
        # add a bit of random noise to the root node of the search tree
        # this noise ensures that all moves may be tried, but the search may still overrule bad moves
        priors = self.Ps[board_string]
        if root_noise:
            logging.debug("At root, adding noise")
            diri = np.random.dirichlet(np.full(priors.shape, root_noise_dirichlet), size=None)
            priors = (1 - root_noise_epsilon)*priors + root_noise_epsilon*diri

        cur_best = -float('inf')
        best_act = -1
        
        logging.debug("Choosing move with highest confidence")
        # pick the action with the highest upper confidence bound

        # TODO:// What do these equations do?...
        for a in range(self.game.getActionSize()):
            if valids[a]: # if action is valid
                if (board_string, a) in self.Qsa: # if the action at board state has a mean action value
                    # guess the upper bound proportianally to number of visits
                    u = self.Qsa[(board_string, a)] + self.c_puct * priors[a] * math.sqrt(self.Ns[board_string]) / (
                                1 + self.Nsa[(board_string, a)])
                else:
                    # otherwise guess the upper bound proportionally to number of visits 
                    try:
                        u = self.c_puct * priors[a] * math.sqrt(self.Ns[board_string] + EPS)  # Q = 0 ?
                    except KeyError: # board hasn't been visited before?
                        u = self.c_puct * priors[a] * math.sqrt(EPS)
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        # update board to go to next state
        a = best_act
        next_s, next_player = self.game.getNextState(canonical_board, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        
        board_value = self.search(next_s) # note how there is no noise now that we are out of root
        
        if (board_string, a) in self.Qsa:
            self.Qsa[(board_string, a)] = (self.Nsa[(board_string, a)] * self.Qsa[(board_string, a)] + board_value) / (self.Nsa[(board_string, a)] + 1)
            self.Nsa[(board_string, a)] += 1
        
        else:
            self.Qsa[(board_string, a)] = board_value
            self.Nsa[(board_string, a)] = 1
        
        self.Ns[board_string] += 1
        return -board_value
 
