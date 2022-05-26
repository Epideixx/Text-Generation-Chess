# ------------------------------------------------------
#                   Training
# ------------------------------------------------------

import tensorflow as tf
from transformer import Transformer
from import_data import import_data
from tokenizer import ChessTokenizer
import numpy as np
import os


length_board = 64
max_moves_in_game = 300
vocab_moves = 64*(7*4 + 8)

transfo = Transformer(vocab_moves=vocab_moves,
                      length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=8, dropout=0.1)



dataset = import_data(filename="fen.txt")
dataset = list(zip(*dataset))

encoder_tokenize = ChessTokenizer()
decoder_tokenize = ChessTokenizer()

encoder_tokenize.fit_on_texts(list(dataset[0]))
decoder_tokenize.fit_on_texts(list(dataset[1]))
decoder_tokenize.fit_on_texts(list(dataset[2]))

tok_encoder = encoder_tokenize.texts_to_sequences(
    list(dataset[0]), maxlen=length_board)
tok_decoder = decoder_tokenize.texts_to_sequences(
    list(dataset[2]), maxlen=max_moves_in_game)
tok_output = decoder_tokenize.texts_to_sequences(
    list(dataset[1]),  maxlen=max_moves_in_game)

x = tf.data.Dataset.from_tensor_slices(
    (tok_encoder, tok_decoder))
y = tf.data.Dataset.from_tensor_slices(tok_output)

transfo.fit(x=x, y=y, batch_size=32, num_epochs=20, wandb_api=False)

