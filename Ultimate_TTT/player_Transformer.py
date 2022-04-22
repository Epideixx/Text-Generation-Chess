# ---------------------------------------
#          Transformer player
# ---------------------------------------

import os
import sys
  
# adding Folder_2 to the system path
path_transfo_modules = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Made_by_Hand")
sys.path.insert(1, path_transfo_modules)

import numpy as np
from tokenizer import ChessTokenizer as TTTTokenizer
from import_data import import_data
from transformer import Transformer
import tensorflow as tf

length_board = 140
max_moves_in_game = 81
vocab_moves = 90
vocab_board = 10

class Transfo_player():

    def __init__(self, player_folder):
        """
        Parameters
        ----------
        player_folder : string
            Folder in which the weights are saved, and also the tokenizers
        """

        # For the moment we have directly the parameters but it would be could to load them from the folder
        self.transfo = Transformer(vocab_board = vocab_board, vocab_moves=vocab_moves,
                      length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=8, dropout=0.1)
        self.transfo.load_weights(os.path.join(player_folder, "model_weights"))

        encoder_filepath = os.path.join(player_folder, 'encoder_tokenizer')
        self.encoder_tokenizer = TTTTokenizer()
        self.encoder_tokenizer.load(encoder_filepath)

        decoder_filepath = os.path.join(player_folder, 'decoder_tokenizer')
        self.decoder_tokenizer = TTTTokenizer()
        self.decoder_tokenizer.load(decoder_filepath)



    def choose_move(self, board, previous_moves):
        board = self.encoder_tokenizer.texts_to_sequences(board, maxlen=length_board)
        previous_moves = self.decoder_tokenizer.texts_to_sequences(previous_moves, maxlen=max_moves_in_game)
        output_token = self.transfo.predict(board, previous_moves)
        output_moves = self.decoder_tokenizer.sequences_to_texts(output_token)
        return output_moves

        

if __name__ == '__main__':
    test = Transfo_player(os.path.join(os.path.dirname(__file__), "Test2_2104"))
    moves = test.choose_move(['|..x|x..|.o.|||...|..x|...|||oxo|...|...|||x..|o..|...|||..x|o..|ox.|||...|...|o.o|||.o.|...|...|||..x|..o|.xx|||x..|...|...||||||.........'], ['C1 H5 E7 E3 D0 C0 I0 G1 A3 C2 H8 E6 E2 F6 H2 F8 H7 D3 A2 A7'])
    print(moves)
    print('ok')

