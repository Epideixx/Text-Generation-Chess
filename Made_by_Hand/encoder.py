# ------------------------------------------------------
#                   Encoder
# ------------------------------------------------------

import numpy as np
import tensorflow as tf
import os

from attention import MultiHeadAttention

from embedding import TextEmbedder
from tokenizer import ChessTokenizer
from import_data import import_data


class EncoderBlock(tf.keras.Model):

    def __init__(self, vocab_size: int, model_size: int, h: int = 8, dropout: float = 0.0):
        """ 
        Architecture of the Encoder block

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary, e.g. maximum number of words in the language
        model_size : int
            Depth of the embedding, e.g size of each vector representing a token
        h : int, default = 8
            Number of heads
        dropout : float, default = 0.0
            Rate of dropout
        """

        super(EncoderBlock, self).__init__()
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.h = h
        self.dropout = dropout

        self.input_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Multi-Head Attention and Normalization
        self.attention = MultiHeadAttention(self.model_size, self.h)
        self.attention_norm = tf.keras.layers.BatchNormalization()

        # FFN and Normalization
        self.dense_1 = tf.keras.layers.Dense(
            self.model_size * 4, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(
            self.model_size)
        self.dropoutlayer = tf.keras.layers.Dropout(rate=dropout)
        self.ffn_norm = tf.keras.layers.BatchNormalization()

    def call(self, input: tf.Tensor, padding_mask: tf.Tensor = None, training: bool = False):
        """
        Parameters
        ----------
        input : tf.Tensor, shape = (..., vocab_size, model_size)
            Input of the Encoder block, from another block or the Embedding layer
        padding_mask : None or tf.Tensor, shape = ()
            Mask to apply on the input to avoid considering some part of it

        Returns
        -------
        TO COMPLETE
        """
        input_norm = self.input_norm(input)
        mha_output, attention_block = self.attention(
            input_norm, input_norm, input_norm, padding_mask)
        mha_output = self.dropoutlayer(mha_output, training=training)
        add_1 = input + mha_output  # Residual connection
        ffn_in = self.attention_norm(add_1)  # Normalization

        ffn_out = self.dense_1(ffn_in)
        ffn_out = self.dropoutlayer(ffn_out, training=training)
        ffn_out = self.dense_2(ffn_out)
        ffn_out = self.dropoutlayer(ffn_out, training=training)
        ffn_out = add_1 + ffn_out
        ffn_out = self.ffn_norm(ffn_out)

        block_output = ffn_out

        return block_output, attention_block


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size: int, model_size: int, h: int = 8, num_encoder: int = 2, dropout: float = 0.0):
        """ 
        Architecture of the Encoder block

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary, e.g. maximum number of words in the language
        model_size : int
            Depth of the embedding, e.g size of each vector representing a token
        h : int, default = 8
            Number of heads
        num_encoder : int, default = 2
            Number of encoders
        dropout : float, default = 0.0
            Rate of dropout
        """
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.h = h
        self.num_encoder = num_encoder
        self.dropout = dropout

        self.encoder_block = [EncoderBlock(
            vocab_size, model_size, h, dropout) for _ in range(num_encoder)]

    def call(self, input: tf.Tensor, padding_mask: tf.Tensor = None, training: bool = False):
        """
        Pass throught the Encoder

        Parameters
        ----------
        input : tf.Tensor, shape = (..., vocab_size, model_size)
            Input of the Encoder block, from another block or the Embedding layer
        padding_mask : None or tf.Tensor, shape = ()
            TO COMPLETE

        Returns
        -------
        TO COMPLETE
        """

        output = input
        attentions = []

        for i in range(self.num_encoder):
            output, attention = self.encoder_block[i](
                output, padding_mask, training)
            attentions.append(attention)

        attention = tf.concat(attentions, axis=-1)

        return output, attention


if __name__ == '__main__':

    vocab_size = 500
    model_size = 40

    encoder = Encoder(vocab_size=vocab_size, model_size=model_size)

    tokenizer = ChessTokenizer()
    dataset = import_data(filename="test.txt")

    embedder = TextEmbedder(vocab_size=vocab_size, depth_emb=model_size)

    boards, move_to_play, moves_mem = (list(l) for l in zip(*dataset))

    tokenizer.fit_on_texts(boards)
    tok_boards = tokenizer.texts_to_sequences(boards[0:15])
    embedded = embedder(tok_boards)

    encoder_output, attention = encoder(embedded)

    print(encoder_output)

    print("ok")
