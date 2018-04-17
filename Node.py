#original code https://github.com/suragnair/alpha-zero-general

import math


class Node(object):
    def __init__(self, parent, board, turn):
        # variable for determining whose turn the board position is from
        self.friendly_turn = turn

        self.parent = parent
        self.board = board

        # will always be between -1 and 1. the value passed back to the network will be divided by N
        # will be either -1 or 1 depending on if its a terminal state
        # naturally from point of view of network. multiply by friendly_turn to get whose turn it is
        self.W = determine_if_terminal(board)
        if (self.W == 0):
            self.W = return_value_from_network()
        self.N = 0  # visit count

        self.P = return_value_from_network()  # probability of being chosen
        self.C = 1  # exploration constant
        self.children = [None]

    def calculateU(self):
        self.N += 1

        # U is score for how valuable this position is in being determined
        self.U = self.W * self.friendly_turn / self.N + self.C * self.P * math.sqrt(
            math.log(self.parent.N) / (1 + self.N))

        # propogate W back up network
        self.parent.W += self.W
        push_value_to_network("W", self.board, self.W / self.N)