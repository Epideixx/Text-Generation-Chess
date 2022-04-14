import os
from shutil import move
from matplotlib.pyplot import cla
from sklearn import datasets
from tqdm import tqdm
import tensorflow as tf

from tokenizer import ChessTokenizer


def import_data(filename="fen.txt", batchsize=32):

    filename = os.path.join(os.path.dirname(__file__), filename)

    boards = []
    moves_to_play = []
    mem_moves = []

    with open(filename) as f:
        for line in tqdm(f, desc="read fen.txt", unit=" moves", mininterval=1):
            if line:
                board_move = [string.strip() for string in line.split('-')]
                boards.append(board_move[0])
                moves_to_play.append(board_move[2] + ' ' + board_move[1])
                mem_moves.append('<Start> ' + board_move[2])

    dataset = list(zip(boards, moves_to_play, mem_moves))

    return dataset


# Test
if __name__ == '__main__':
    dataset = import_data(filename="test.txt")
    print(dataset[0])

    # boards = list(zip(*dataset))[0]
    # print(boards[0])

    # tokenizer = ChessTokenizer()
    # tokenizer.fit_on_texts(boards)
    # phrase_test = boards[2]
    # token = tokenizer(phrase_test)
    # print(token)

    # print('ok')

    dataset = list(zip(*dataset))
    print(len(dataset[2]))
    dataset = tf.data.Dataset.from_tensor_slices(
        (list(dataset[0]), list(dataset[1]), list(dataset[2])))
    dataset = dataset.batch(batch_size=32)
    print(dataset)
    print("ok")
