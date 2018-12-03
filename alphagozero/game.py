#original code https://github.com/suragnair/alpha-zero-general

import numpy as np
from typing import Tuple, List, Union

from alphagozero.board import Board


class Game(object):
    def __init__(self, n:int =15, nir: int=5) -> None:
        self.n = n
        self.n_in_row = nir

    def getInitBoard(self) -> np.array:
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self) -> Tuple[int, int]:
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self) -> int:
        # return number of actions
        return self.n * self.n + 1

    def getNextState(self, board: Board, player: int, action: int) -> Tuple[np.array, int]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n * self.n:
            return (board, -player)
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.execute_move(move, player)
        return (b.pieces, -player)

    # modified
    def getValidMoves(self, board: Board, player: int) -> np.array:
        # return a fixed size binary vector
        valids = [0] * self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    # modified
    def getGameEnded(self, board: np.array, player: int) -> int:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        n = self.n_in_row

        for w in range(self.n):
            for h in range(self.n):
                if (w in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[i][h] for i in range(w, w + n))) == 1):
                    return board[w][h]
                if (h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w][j] for j in range(h, h + n))) == 1):
                    return board[w][h]
                if (w in range(self.n - n + 1) and h in range(self.n - n + 1) and board[w][h] != 0 and
                        len(set(board[w + k][h + k] for k in range(n))) == 1):
                    return board[w][h]
                if (w in range(self.n - n + 1) and h in range(self.n - n + 1, self.n) and board[w][h] != 0 and
                        len(set(board[w + l][h - l] for l in range(n))) == 1):
                    return board[w][h]
        if b.has_legal_moves():
            return 0
        #return 1e-4
        return 0

    def getCanonicalForm(self, board: np.array, player: int) -> np.array:
        # return state if player==1, else return -state if player==-1
        return player * board

    # modified
    def getSymmetries(self, board: np.array, pi: np.array) -> List[Tuple[np.array, List[np.array]]]:
        # mirror, rotational
        assert(len(pi) == self.n**2 + 1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l: List[Tuple[np.array, List[np.array]]] = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board: np.array) -> str:
        # 8x8 numpy array (canonical board)
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.get_as_string()


def display(board: np.array) -> None:
    return
    n = board.shape[0]

    for y in range(n):
        print(y, "|", end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|", end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1:
                print("b ", end="")
            elif piece == 1:
                print("W ", end="")
            else:
                if x == n:
                    print("-", end="")
                else:
                    print("- ", end="")
        print("|")

    print("   -----------------------")
