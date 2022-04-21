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

        with open(os.path.join(player_folder, 'encoder_tokenizer')) as encoder_tokenizer_data:
            data = encoder_tokenizer_data.read()
            self.encoder_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

        with open(os.path.join(player_folder, 'decoder_tokenizer')) as decoder_tokenizer_data:
            data = decoder_tokenizer_data.read()
            self.decoder_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)



    def choose_move(self, board, previous_moves):
        pass
        

if __name__ == '__main__':
    test = Transfo_player(os.path.join(os.path.dirname(__file__), "Test1_2104"))
    print(test.decoder_tokenizer.sequences_to_texts([[1, 4, 10]]))
    print('ok')

