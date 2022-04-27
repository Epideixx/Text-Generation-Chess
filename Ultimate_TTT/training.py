# ------------------------------------------------------
#                   Training
# ------------------------------------------------------


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

transfo = Transformer(vocab_board = vocab_board, vocab_moves=vocab_moves,
                      length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=4, dropout=0.2)

filename = os.path.join(os.path.dirname(__file__), "test.txt")
dataset = import_data(filename=filename)
dataset = list(zip(*dataset))

encoder_tokenize = TTTTokenizer(char_level = True)
decoder_tokenize = TTTTokenizer()


encoder_tokenize.fit_on_texts(list(dataset[0]))
decoder_tokenize.fit_on_texts(list(dataset[1]))
decoder_tokenize.fit_on_texts(list(dataset[2]))


tok_encoder = encoder_tokenize.texts_to_sequences(
    list(dataset[0]), maxlen=length_board)
tok_decoder = decoder_tokenize.texts_to_sequences(
    list(dataset[2]), maxlen=max_moves_in_game)
tok_output = decoder_tokenize.texts_to_sequences(
    list(dataset[1]), maxlen=max_moves_in_game)

x = tf.data.Dataset.from_tensor_slices(
    (tok_encoder, tok_decoder))
y = tf.data.Dataset.from_tensor_slices(tok_output)


# Everything to save
folder_to_save = os.path.join(os.path.dirname(__file__), "Official_25_04_22")
if not os.path.exists(folder_to_save):
    os.makedirs(folder_to_save)


encoder_filepath = os.path.join(folder_to_save, "encoder_tokenizer")
encoder_tokenize.save(encoder_filepath)

decoder_filepath = os.path.join(folder_to_save, "decoder_tokenizer")
decoder_tokenize.save(decoder_filepath)

transfo.fit(x=x, y=y, batch_size=64, num_epochs=5, wandb_api=True, file_to_save = None, validation_split = 0.02)
print(transfo.summary())
