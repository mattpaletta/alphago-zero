'''
https://github.com/suragnair/alpha-zero-general/blob/master/gobang/GobangLogic.py

Author: MBoss
Date: Jan 17, 2018.
Board class.
Board data:
  1=white, -1=black, 0=empty
  first dim is column , 2nd is row:
     pieces[1][7] is the square in column 2,
     at the opposite end of the board in row 8.
Squares are stored and manipulated as (x,y) tuples.
x is the column, y is the row.
'''
from typing import List, Tuple

class Board(object):
    def __init__(self, n: int) -> None:
        "Set up initial board configuration."
        self.n: int = n
        # Create the empty board array.
        self.pieces: List[List[int]] = [[0] * self.n] * self.n
        #for i in range(self.n):
        #    self.pieces[i] = [0] * self.n

    # add [][] indexer syntax to the Board
    def __getitem__(self, index: List[int]) -> List[int]:
        return self.pieces[index]

    def get_as_string(self) -> str:
        board_str = "".join(map(str, self.pieces)).replace(" ", "").replace("[", "").replace("]", "")
        return board_str

    def get_legal_moves(self, color: int) -> List[Tuple[int, int]]:
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black
        """
        moves = set()  # stores the legal moves.

        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    moves.add((x, y))
        return list(moves)

    def has_legal_moves(self) -> bool:
        """Returns True if has legal move else False
        """
        # Get all empty locations.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y] == 0:
                    return True
        return False

    def execute_move(self, move: Tuple[int, int], color: int) -> None:
        """Perform the given move on the board; flips pieces as necessary.
        color gives the color pf the piece to play (1=white,-1=black)
        """
        (x, y) = move
        assert self[x][y] == 0
        self[x][y] = color
