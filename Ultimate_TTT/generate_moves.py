# ------------------------------------------
#             Moves Generation
# ------------------------------------------

import os
from tqdm import tqdm

from MCTS import MCTS
from game import TTT

cpuct = 0.1


def play_one_game(nb_simu, player1="mcts", player2="mcts"):
    """
    Returns
    -------
    game_mem : list of tuple(board, chosen_move, previous_moves)
        Record of the moves that have been played by the winner
    """

    # Def players
    if player1 == "mcts":
        player1 = MCTS(cpuct=cpuct)
    elif player1 == "random":
        raise NotImplementedError(
            "The random player has not been implemented yet.")
    else:
        raise ValueError(
            "For the moment, only two classes of player are available : 'mcts' and 'random'.")

    if player2 == "mcts":
        player2 = MCTS(cpuct=cpuct)
    elif player2 == "random":
        raise NotImplementedError(
            "The random player has not been implemented yet.")
    else:
        raise ValueError(
            "For the moment, only two classes of player are available : 'mcts' and 'random'.")

    # Let's play the game

    mem_moves = []
    game_mem_1 = []
    game_mem_2 = []

    player = 0
    ttt = TTT()
    while not(ttt.game_over):

        if player == 0:
            # Simulations
            for _ in range(nb_simu):
                player1.search(ttt)

            # Play the best move
            best_move = list(player1.getActionProb(ttt, temp=0).keys())[0]
            ttt.push(best_move)

            # Mem
            game_mem_1.append(
                (ttt.rep_board(), ttt.rep_move(best_move), " ".join(mem_moves)))
            mem_moves.append(ttt.rep_move(best_move))

        else:
            # Simulations
            for _ in range(nb_simu):
                player2.search(ttt)

            # Play the best move
            best_move = list(
                player2.getActionProb(ttt, temp=0).keys())[0]
            ttt.push(best_move)

            # Mem
            game_mem_2.append(
                (ttt.rep_board(), ttt.rep_move(best_move), " ".join(mem_moves)))
            mem_moves.append(ttt.rep_move(best_move))

        if not ttt.game_over:
            ttt = ttt.mirror()
            player = 1 - player

    if ttt.winner != 0:
        if player == 0:
            return game_mem_1
        else:
            return game_mem_2

    else:
        return None


def save_games(nb_games, nb_simu=100, filename="fen.txt", rate_mcts_vs_mcts=1, rate_mcts_vs_rnd=0, rate_rnd_vs_rnd=0):
    """
    Save in the chosen file evrey moves played by the winner in multiple games.
    """

    games = 0
    set_to_add = set()

    fen_file = os.path.join(os.path.dirname(__file__), filename)

    # Import the new data
    for _ in tqdm(range(nb_games), desc="Generate games", unit=" games", mininterval=1):
        game = play_one_game(nb_simu=nb_simu)
        if game:
            for (board, move, mem_moves) in game:
                line = board + ' - ' + move + ' - ' + mem_moves
                line.strip()
                set_to_add.add(line)
            games += 1

    # And we write the data in fen.txt
    with open(fen_file, "w") as f:
        for line in tqdm(set_to_add, desc="write fen.txt", unit=" moves", mininterval=1):
            f.write(line + "\n")
    print("Imported : ", games, " games")


if __name__ == '__main__':
    res = play_one_game(nb_simu=60, player1="mcts", player2="mcts")
    print(res)

    save_games(nb_games=10, nb_simu=50)
