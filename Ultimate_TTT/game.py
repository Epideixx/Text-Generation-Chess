# ------------------------------------------
#           Ultimate TicTacToe
# ------------------------------------------

import numpy as np
import copy
import abc

# Abstract class


class Game:
    def __init__(self):
        #__metaclass__ = abc.ABCMeta
        self.board = None
        self.str = None
        self.legal_moves = None
        self.representation = None
        self.game_over = False
        self.winner = 0
        self.draw = False
        self.allMoves = None

    def copy(self):
        return copy.deepcopy(self)

    @abc.abstractmethod
    def mirror(self):
        return

    @abc.abstractmethod
    def push(self, pos, color=0):
        return

    @abc.abstractmethod
    def playRandomMove(self):
        return

    @abc.abstractmethod
    def show(self):
        return

    @abc.abstractmethod
    def get_legal_moves(self):
        return

    def get_symmetries(self, pi):
        syms = []
        syms.append(np.array(self.representation, copy=True))
        return syms

    def MonteCarloValue(self):
        """
        Returns
        -------
        score : int in (-1, 0, 1)
            0 if draw 1 or -1 else 
        """
        board = self.copy()
        player = 1
        while not board.is_game_over():
            board.playRandomMove()
            board = board.mirror()
            player = (player+1) % 2
        if board.draw:
            return 0
        return (1, -1)[player]

    def fen(self):
        return self.str

    def is_game_over(self):
        return self.game_over

    def result(self):
        if not self.game_over:
            return 0
        else:
            if self.draw and self.winner == 0:
                return 0.01
            else:
                return self.winner


class TTT(Game):
    def __init__(self):
        super().__init__()
        self.board = np.zeros((9, 9, 2))
        self.str = "0"*9*9*2+"00"  # String representation
        self.upperBoard = np.zeros((3, 3, 2))  # Big parts (3x3) of the board
        self.legal_moves = np.ones((9, 9))
        self.emptyCells = np.ones((9, 9))
        self.representation = np.concatenate(
            (self.board, np.expand_dims(self.legal_moves, axis=2)), axis=2)
        # Mask to apply to compel to play in a certain part
        self.turnMask = np.zeros((3, 3, 9, 9))
        for i in range(3):
            for j in range(3):
                self.turnMask[i, j, i*3:(i+1)*3, j*3:(j+1)*3] = np.ones((3, 3))

    def mirror(self):
        """
        Inverse the game relatively to the players.
        It enables to pass from player 1 to player 2 and the inverse. 

        Returns
        -------
        newTTT : TTT
            TTT which corresponds to the game seen as in a mirror
        """
        newTTT = self.copy()
        newTTT.board = np.flip(newTTT.board, axis=2)
        newTTT.upperBoard = np.flip(newTTT.upperBoard, axis=2)
        newTTT.str = self.str[9*9:-2]+self.str[:9*9]+self.str[-2:]
        newTTT.winner = -self.winner
        return newTTT

    def push(self, pos, color=0):
        """
        Play a move 

        Parameters
        ----------
        pos : tuple (x:int, y:int)
            Position we want to play on
        color : int = 0 or 1
            Player
        """
        if self.legal_moves[pos[0], pos[1]] == 0:
            self.show()
            print(self.legal_moves)
            print(pos)
            raise Exception("Ilegal move")
        self.board[pos[0], pos[1], color] = 1

        m = 3*(pos[0]//3)
        n = 3*(pos[1]//3)

        # If one subpart is won
        if (1 == self.board[m, n, 0] == self.board[m, n+1, 0] == self.board[m, n+2, 0])\
                or (1 == self.board[m+1, n, 0] == self.board[m+1, n+1, 0] == self.board[m+1, n+2, 0])\
                or (1 == self.board[m+2, n, 0] == self.board[m+2, n+1, 0] == self.board[m+2, n+2, 0])\
                or (1 == self.board[m, n, 0] == self.board[m+1, n, 0] == self.board[m+2, n, 0])\
                or (1 == self.board[m, n+1, 0] == self.board[m+1, n+1, 0] == self.board[m+2, n+1, 0])\
                or (1 == self.board[m, n+2, 0] == self.board[m+1, n+2, 0] == self.board[m+2, n+2, 0])\
                or (1 == self.board[m, n, 0] == self.board[m+1, n+1, 0] == self.board[m+2, n+2, 0])\
                or (1 == self.board[m+2, n, 0] == self.board[m+1, n+1, 0] == self.board[m, n+2, 0]):
            self.endSubTTT(m//3, n//3)

        self.str = self.str[:pos[0]*9+pos[1]]+"1" + \
            self.str[pos[0]*9+pos[1]+1:-2]+str(pos[0])+str(pos[1])
        self.emptyCells[pos[0], pos[1]] = 0
        self.legal_moves = self.emptyCells * \
            self.turnMask[pos[0] % 3, pos[1] % 3]
        if np.max(self.legal_moves) == 0:
            self.legal_moves = self.emptyCells
        if np.max(self.legal_moves) == 0 and self.winner == 0:
            self.game_over = True
            self.draw = True
        self.representation = np.concatenate(
            (self.board, np.expand_dims(self.legal_moves, axis=2)), axis=2)

    def endSubTTT(self, m, n):
        """
        Changements applied because a (3x3) part has been won
        If the game is over, it sets game_over t True

        Parameters
        ----------
        m : int
            First coordinate
        n : int
            Second coordinate
        """
        self.upperBoard[m, n, 0] = 1
        self.emptyCells *= 1-self.turnMask[m, n]
        if (1 == self.upperBoard[0, 0, 0] == self.upperBoard[0, 1, 0] == self.upperBoard[0, 2, 0])\
                or (1 == self.upperBoard[1, 0, 0] == self.upperBoard[1, 1, 0] == self.upperBoard[1, 2, 0])\
                or (1 == self.upperBoard[2, 0, 0] == self.upperBoard[2, 1, 0] == self.upperBoard[2, 2, 0])\
                or (1 == self.upperBoard[0, 0, 0] == self.upperBoard[1, 0, 0] == self.upperBoard[2, 0, 0])\
                or (1 == self.upperBoard[0, 1, 0] == self.upperBoard[1, 1, 0] == self.upperBoard[2, 1, 0])\
                or (1 == self.upperBoard[0, 2, 0] == self.upperBoard[1, 2, 0] == self.upperBoard[2, 2, 0])\
                or (1 == self.upperBoard[0, 0, 0] == self.upperBoard[1, 1, 0] == self.upperBoard[2, 2, 0])\
                or (1 == self.upperBoard[2, 0, 0] == self.upperBoard[1, 1, 0] == self.upperBoard[0, 2, 0]):
            self.game_over = True
            self.winner = 1

    def get_legal_moves(self):
        """
        Returns
        -------
        moves : list
            List of legal moves
        """
        moves = []
        for i in range(9):
            for j in range(9):
                if self.legal_moves[i, j] == 1:
                    moves.append((i, j))
        return moves

    def playRandomMove(self):
        """
        Play a random move
        """
        p = np.random.choice(np.flatnonzero(
            self.legal_moves == self.legal_moves.max()))
        self.push((p//9, p % 9))

    def get_symmetries(self, pi):
        """
        Boards and moves identical by rotation or symetry of the board

        Returns
        -------
        syms : list of tuples (s, pis) with s a board representation and pis a move
        """
        syms = []
        for rot in range(4):
            for flip in [False, True]:
                s = np.rot90(self.representation, k=rot).copy()
                pis = pi.copy()
                for _ in range(rot):
                    rpis = {}
                    for p in pis:
                        rpis[(8-int(p[1]), int(p[0]))] = pis[p]
                    pis = rpis.copy()
                if flip:
                    s = np.flip(s, axis=0)
                    rpis = {}
                    for p in pis:
                        try:
                            rpis[(8-int(p[0]), int(p[1]))] = pis[p]
                        except:
                            print(p)
                    pis = rpis.copy()
                syms.append((s, pis))
        return syms

    def show(self):
        for i in range(9):
            if i % 3 == 0:
                print("-------------")
            line = ""
            for j in range(9):
                if j % 3 == 0:
                    line += "|"
                if self.board[i, j, 0] == 1:
                    line += "x"
                elif self.board[i, j, 1] == 1:
                    line += "o"
                else:
                    line += " "
            print(line+"|")
        print("-------------")
        print("#")
        print("---")
        for i in range(3):
            line = ""
            for j in range(3):
                if self.upperBoard[i, j, 0] == 1:
                    line += "x"
                elif self.upperBoard[i, j, 1] == 1:
                    line += "o"
                else:
                    line += " "
            print(line)
        print("---")


if __name__ == '__main__':
    ttt = TTT()
    ttt.push((0, 0))
    ttt = ttt.mirror()
    ttt.push((0, 2))
    ttt = ttt.mirror()
    ttt.show()
    for i in range(50):
        ttt.playRandomMove()
        ttt = ttt.mirror()
    ttt.show()
