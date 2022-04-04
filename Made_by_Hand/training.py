# ------------------------------------------------------
#                   Training
# ------------------------------------------------------

import tensorflow as tf
from transformer import Transformer
from import_data import import_data
from tokenizer import ChessTokenizer
import numpy as np

# To delete
test = True
length_board = 64
max_moves_in_game = 300
vocab_moves = 64*(7*4 + 8)

transfo = Transformer(vocab_moves=vocab_moves,
                      length_board=length_board, max_moves_in_game=max_moves_in_game, num_layers=4)


dataset = import_data(filename="test.txt")
dataset = list(zip(*dataset))

encoder_tokenize = ChessTokenizer()
decoder_tokenize = ChessTokenizer()

encoder_tokenize.fit_on_texts(list(dataset[0]))
decoder_tokenize.fit_on_texts(list(dataset[1]))
decoder_tokenize.fit_on_texts(list(dataset[2]))

tok_encoder = encoder_tokenize(
    list(dataset[0]), maxlen=length_board)
tok_decoder = decoder_tokenize(
    list(dataset[2]), maxlen=max_moves_in_game)
tok_output = decoder_tokenize(
    list(dataset[1]))

x = tf.data.Dataset.from_tensor_slices(
    (tok_encoder, tok_decoder))
y = tf.data.Dataset.from_tensor_slices(tok_output)

print("ok 1")

transfo.fit(x=x, y=y, batch_size=32, num_epochs=1, wandb_api=False)
if test:
    layer = transfo.encoder.encoder_block[0].dense_1
    weights = layer.get_weights()

trainable_var = transfo.trainable_variables
print(len(trainable_var))

"""
transfo.fit(x=x, y=y, batch_size=32, num_epochs=1, wandb_api=False)
if test:
    weights = [w - transfo.encoder.encoder_block[0].dense_1.get_weights()[i]
               for i, w in enumerate(weights)]
    print(weights)

x_input_for_shape = [tok_encoder.shape, tok_decoder.shape]

print("ok 2")
print("Input shape : ", x_input_for_shape)
# transfo.call((tok_encoder[0:30], tok_decoder[0:30]))

transfo.build(input_shape=x_input_for_shape)

print(transfo.summary())

transfo.save("test_transfo")

print("ok")
"""
