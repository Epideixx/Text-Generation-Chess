import os
from tqdm.auto import tqdm
import glob
import chess.pgn
import tensorflow as tf
import numpy as np
import re

MAX_IMPORT = 50000


def importPgn(filename, set_to_add, max_import=10000):

    counter = 0
    total = 0  # Number of games to import (just for tqdm display)

    with open(filename) as f:
        for line in f:
            if "[Result" in line:
                total += 1

    if total > max_import:
        total = max_import

    pbar = tqdm(total=total, desc="read " + filename,
                unit=" games", mininterval=1)

    pgn = open(filename)
    while counter < max_import:

        # Each time it is called, it reads the next game
        game = chess.pgn.read_game(pgn)
        if not game:
            break

        board = game.board()
        moves = game.mainline_moves()
        count = sum(1 for _ in moves)

        # skip unfinished games
        if count <= 5:
            continue

        result = game.headers["Result"]
        # import only resultative games
        if result != "1-0" and result != "0-1":
            continue

        # We just keep moves which makes us win
        mem_moves = ""

        for move in moves:
            if (board.turn == chess.WHITE and result == "1-0") or (board.turn == chess.BLACK and result == "0-1"):
                line = (re.sub(r'\n', ' ', str(board))
                        + " - "
                        + move.uci()
                        + " - "
                        + mem_moves
                        ).strip()

                mem_moves += " " + move.uci()
                set_to_add.add(line)

            board.push(move)

        counter += 1
        pbar.update(1)

    pbar.close()
    return counter


def generate(file_to_write="fen.txt", max_import=MAX_IMPORT):

    games = 0
    moves = 0
    set_to_add = set()

    fen_file = os.path.join(os.path.dirname(__file__), file_to_write)

    # Importing already existing data
    if os.path.exists(fen_file):
        with open(fen_file) as f:
            for line in tqdm(f, desc="read fen.txt", unit=" moves", mininterval=1):
                if line:
                    set_to_add.add(line)
                    max_import -= 1
                    if max_import <= 0:
                        break
    # Import the new data
    for file in glob.glob("pgn/*.pgn"):
        print(file)
        count = importPgn(file, set_to_add, max_import)
        games += count
        max_import -= count
        if max_import <= 0:
            break

    # And we write the data in fen.txt
    with open(fen_file, "w") as f:
        for line in tqdm(set_to_add, desc="write fen.txt", unit=" moves", mininterval=1):
            f.write(line + "\n")
            moves += 1
    print("imported " + str(games) + " games, " + str(moves) + " moves")


if __name__ == '__main__':
    generate()
