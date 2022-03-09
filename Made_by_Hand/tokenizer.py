import os
from matplotlib.pyplot import cla
from tqdm import tqdm
import tensorflow as tf


class ChessTokenizer():

    def __init__(self) -> None:

        self.tokenizer_boards = tf.keras.preprocessing.text.Tokenizer(
            filters='')
        self.tokenizer_moves = tf.keras.preprocessing.text.Tokenizer(
            filters='')

    def import_data(self, filename="fen.txt", batchsize=32):

        filename = os.path.join(os.path.dirname(__file__), filename)

        boards = []
        moves_to_play = []
        mem_moves = []

        with open(filename) as f:
            for line in tqdm(f, desc="read fen.txt", unit=" moves", mininterval=1):
                if line:
                    board_move = [string.strip() for string in line.split('-')]
                    boards.append(board_move[0])
                    moves_to_play.append(board_move[1] + " <end>")
                    mem_moves.append("<start> " + board_move[2])

        self.tokenizer_boards.fit_on_texts(boards)
        boards = self.tokenizer_boards.texts_to_sequences(boards)
        boards = tf.keras.preprocessing.sequence.pad_sequences(
            boards, padding='post')

        self.tokenizer_moves.fit_on_texts(moves_to_play)
        self.tokenizer_moves.fit_on_texts(mem_moves)

        moves_to_play = self.tokenizer_moves.texts_to_sequences(moves_to_play)
        moves_to_play = tf.keras.preprocessing.sequence.pad_sequences(
            moves_to_play, padding='post', maxlen=2)

        mem_moves = self.tokenizer_moves.texts_to_sequences(mem_moves)
        mem_moves = tf.keras.preprocessing.sequence.pad_sequences(
            mem_moves, padding='post', maxlen=300)

        dataset = tf.data.Dataset.from_tensor_slices(
            (boards, moves_to_play, mem_moves))
        dataset = dataset.shuffle(20).batch(batchsize)

        return dataset


# Test
if __name__ == '__main__':
    tokenizer = ChessTokenizer()
    dataset = tokenizer.import_data(filename="test.txt")
    for batch, (boards, move_to_play, moves_mem) in enumerate(dataset):
        print(batch)
        print(boards)
        print(moves_mem)

        if batch >= 0:
            break

    print('ok')
