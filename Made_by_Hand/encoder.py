# ------------------------------------------------------
#                   Encoder
# ------------------------------------------------------

import numpy as np
import tensorflow as tf
import os

from attention import MultiHeadAttention

from embedding import TextEmbedder
from tokenizer import ChessTokenizer


class EncoderBlock(tf.keras.Model):

    def __init__(self, vocab_size: int, model_size: int, h: int = 8):
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
        """

        super(EncoderBlock, self).__init__()
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.h = h

        # Multi-Head Attention and Normalization
        self.attention = MultiHeadAttention(self.model_size, self.h)
        self.attention_norm = tf.keras.layers.BatchNormalization()

        # FFN and Normalization
        self.dense_1 = tf.keras.layers.Dense(
            self.model_size * 4, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(
            self.model_size)
        self.ffn_norm = tf.keras.layers.BatchNormalization()

    def __call__(self, input: tf.Tensor, padding_mask: tf.Tensor = None):
        """
        Parameters
        ----------
        input : tf.Tensor, shape = (..., vocab_size, model_size)
            Input of the Encoder block, from another block or the Embedding layer
        padding_mask : None or tf.Tensor, shape = ()
            TO COMPLETE

        Returns
        -------
        """

        mha_output, attention_block = self.attention(
            input, input, input, padding_mask)
        add_norm_1 = input + mha_output  # Residual connection
        add_norm_1 = self.attention_norm(add_norm_1)  # Normalization

        ffn_in = add_norm_1

        ffn_out = self.dense_1(ffn_in)
        ffn_out = self.dense_2(ffn_out)
        ffn_out = ffn_in + ffn_out
        ffn_out = self.ffn_norm(ffn_out)

        block_output = ffn_out

        return block_output, attention_block


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size: int, model_size: int, h: int = 8, num_encoder: int = 2):
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
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.h = h
        self.num_encoder = num_encoder

        self.encoder_block = [EncoderBlock(
            vocab_size, model_size, h) for _ in range(num_encoder)]

    def __call__(self, input: tf.Tensor, padding_mask: tf.Tensor = None):
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
        """

        output = input
        attentions = []

        for i in range(self.num_encoder):
            output, attention = self.encoder_block[i](output, padding_mask)
            attentions.append(attention)

        attention = tf.concat(attentions, axis=-1)

        return output, attention


if __name__ == '__main__':

    vocab_size = 500
    model_size = 40

    encoder = Encoder(vocab_size=vocab_size, model_size=model_size)

    tokenizer = ChessTokenizer()
    dataset = tokenizer.import_data(filename="test.txt")

    embedder = TextEmbedder(vocab_size=vocab_size, depth_emb=model_size)
    for batch, (boards, move_to_play, moves_mem) in enumerate(dataset):
        embedded = embedder(boards)

        encoder_output, attention = encoder(embedded)

        print(encoder_output)

        if batch >= 0:
            break

    print("ok")
