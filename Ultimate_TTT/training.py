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
model_size = 128

transfo = Transformer(vocab_board = vocab_board, vocab_moves=vocab_moves, model_size = model_size,
                      length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=8, dropout=0.1)

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
folder_to_save = os.path.join(os.path.dirname(__file__), "Train_20_05_22")
if not os.path.exists(folder_to_save):
    os.makedirs(folder_to_save)


encoder_filepath = os.path.join(folder_to_save, "encoder_tokenizer")
encoder_tokenize.save(encoder_filepath)

decoder_filepath = os.path.join(folder_to_save, "decoder_tokenizer")
decoder_tokenize.save(decoder_filepath)

transfo.fit(x=x, y=y, batch_size=2048, num_epochs=15, wandb_api=True, file_to_save = folder_to_save, validation_split = 0.0)

