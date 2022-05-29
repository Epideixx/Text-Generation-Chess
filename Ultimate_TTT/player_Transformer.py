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
model_size = 32

class Transfo_player():

    def __init__(self, player_folder):
        """
        Parameters
        ----------
        player_folder : string
            Folder in which the weights are saved, and also the tokenizers
        """

        # For the moment we have directly the parameters but it would be could to load them from the folder
        self.transfo = Transformer(vocab_board = vocab_board, vocab_moves=vocab_moves, model_size = model_size,
                      length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=8, dropout=0.1)
        self.transfo.load_weights(os.path.join(player_folder, "model_weights")).expect_partial()

        encoder_filepath = os.path.join(player_folder, 'encoder_tokenizer')
        self.encoder_tokenizer = TTTTokenizer()
        self.encoder_tokenizer.load(encoder_filepath)

        decoder_filepath = os.path.join(player_folder, 'decoder_tokenizer')
        self.decoder_tokenizer = TTTTokenizer()
        self.decoder_tokenizer.load(decoder_filepath)



    def choose_move(self, board, previous_moves):
        board = ["S" + board]
        board = self.encoder_tokenizer.texts_to_sequences(board, maxlen=length_board)

        previous_moves = ["<Start> " + previous_moves]
        previous_moves = self.decoder_tokenizer.texts_to_sequences(previous_moves, maxlen=max_moves_in_game)
        
        output_token, attention_encoder, masked_attention_decoder, attention_decoder = self.transfo.predict(board, previous_moves)
        output_moves = self.decoder_tokenizer.sequences_to_texts(output_token)
        return output_moves, attention_encoder, masked_attention_decoder, attention_decoder

        

if __name__ == '__main__':
    test = Transfo_player(os.path.join(os.path.dirname(__file__), "Cluster_26_05"))
    moves, _, _, _ = test.choose_move('|..x|x..|.o.|||...|..x|...|||oxo|...|...|||x..|o..|...|||..x|o..|ox.|||...|...|o.o|||.o.|...|...|||..x|..o|.xx|||x..|...|...||||||.........', 'C1 H5 E7 E3 D0 C0 I0 G1 A3 C2 H8 E6 E2 F6 H2 F8 H7 D3 A2 A7')
    print(moves)
    print('ok')

    moves, _, _, _ = test.choose_move('|..x|x..|.o.|||...|..x|...|||oxo|...|...|||x..|o..|...|||..x|o..|ox.|||...|...|o.o|||.o.|...|...|||..x|..o|.xx|||x..|...|...||||||.........', 'C1 H5 E7 E3 D0 C0 I0 G1 A3 C2 H8 E6 E2 F6 H2 F8 H7 D3')
    print(moves)
    print('ok')

    moves, _, _, _ = test.choose_move("|xox|x.x|oox|||o.o|.x.|xxo|||ooo|xx.|.o.|||..x|xoo|xo.|||.x.|o.x|oox|||xxo|oox|xo.|||o.o|x.o|xo.|||oo.|.xx|x.x|||xoo|xox|xxo||||||ox.x.ooxx", 'I6 I1 I5 I8 G6 B2 D6 C1 G3 A1 C4 G5 B7 F4 I3 H0 F1 I4 H5 E6 D2 A6 A0 B0 F0 G2 B6 F2 H8 E7 D3 C0 I0 G0 A2 A7 B4 E3 E1 D5 A8 B8 F6 H1 F5 G7 A5 C7 H4 D4 A3 C2 I7 I2 H6 D7 C3 F3 E8 F7') #E5
    print(moves)