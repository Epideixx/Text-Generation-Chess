# ------------------------------------------------------
#                   Training
# ------------------------------------------------------


# importing sys
import sys
  
# adding Folder_2 to the system path
sys.path.insert(1, 'C:/Users/jonat/OneDrive/Documents/CentraleSupelec/2A/Echecs2A/Text-Generation-Chess/Made_by_Hand')

print(sys.path)

import numpy as np
from tokenizer import ChessTokenizer as TTTTokenizer
from import_data import import_data
from transformer import Transformer
import tensorflow as tf
import os


length_board = 140
max_moves_in_game = 81
vocab_moves = 90
vocab_board = 10

transfo = Transformer(vocab_board = vocab_board, vocab_moves=vocab_moves,
                      length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=4, dropout=0.1)
transfo2 = Transformer(vocab_moves=vocab_moves,
                       length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=4, dropout=0.1)

filename = os.path.join(os.path.dirname(__file__), "fen.txt")
dataset = import_data(filename=filename)
dataset = list(zip(*dataset))

encoder_tokenize = TTTTokenizer()
decoder_tokenize = TTTTokenizer()

encoder_tokenize.fit_on_texts(list(dataset[0]))
decoder_tokenize.fit_on_texts(list(dataset[1]))
decoder_tokenize.fit_on_texts(list(dataset[2]))


tok_encoder = encoder_tokenize.texts_to_sequences(
    list(dataset[0]), maxlen=length_board)
tok_decoder = decoder_tokenize.texts_to_sequences(
    list(dataset[2]), maxlen=max_moves_in_game)
tok_output = decoder_tokenize.texts_to_sequences(
    list(dataset[1]))

x = tf.data.Dataset.from_tensor_slices(
    (tok_encoder, tok_decoder))
y = tf.data.Dataset.from_tensor_slices(tok_output)

transfo.fit(x=x, y=y, batch_size=32, num_epochs=1, wandb_api=False)
