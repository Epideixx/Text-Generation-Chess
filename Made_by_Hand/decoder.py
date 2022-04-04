# ------------------------------------------------------
#                   Decoder
# ------------------------------------------------------

import numpy as np
import tensorflow as tf
import os

from attention import MultiHeadAttention

from embedding import TextEmbedder
from tokenizer import ChessTokenizer
from encoder import Encoder
from import_data import import_data


class DecoderBlock(tf.keras.Model):

    def __init__(self, vocab_size: int, model_size: int, h: int = 8):
        """ 
        Architecture of the Decoder block

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary, e.g. maximum number of words in the output language
        model_size : int
            Depth of the embedding, e.g size of each vector representing a token
        h : int, default = 8
            Number of heads
        """

        super(DecoderBlock, self).__init__()
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.h = h

        self.input_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # (Masked) Multi-Head Attention and Normalization
        self.masked_attention = MultiHeadAttention(self.model_size, self.h)
        self.masked_attention_norm = tf.keras.layers.BatchNormalization()

        # Multi-Head Attention and Normalization
        self.attention = MultiHeadAttention(self.model_size, self.h)
        self.attention_norm = tf.keras.layers.BatchNormalization()

        # FFN and Normalization
        self.dense_1 = tf.keras.layers.Dense(
            self.model_size * 4, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(
            self.model_size)
        self.ffn_norm = tf.keras.layers.BatchNormalization()

    def __call__(self, input: tf.Tensor, encoder_output: tf.Tensor, padding_mask: tf.Tensor = None):
        """
        Parameters
        ----------
        input : tf.Tensor, shape = (..., vocab_size, model_size)
            Input of the Decoder block, from another block or the Embedding layer
        encoder_output : tf.Tensor, shape = (..., vocab_size, model_size)
            Output of the Encoder block
        padding_mask : None or tf.Tensor, shape = ()
            TO COMPLETE

        Returns
        -------
        """

        input_norm = self.input_norm(input)
        masked_mha_output, masked_attention_block = self.attention(
            input_norm, input_norm, input_norm, padding_mask)
        add_1 = input + masked_mha_output  # Residual connection
        add_norm_1 = self.attention_norm(add_1)  # Normalization

        Q_mha = add_norm_1

        mha_output, attention_block = self.masked_attention(
            Q_mha, encoder_output, encoder_output, padding_mask)
        add_2 = add_1 + mha_output  # Residual connection
        add_norm_2 = self.attention_norm(add_2)  # Normalization

        ffn_in = add_norm_2

        ffn_out = self.dense_1(ffn_in)
        ffn_out = self.dense_2(ffn_out)
        ffn_out = ffn_in + ffn_out
        ffn_out = self.ffn_norm(ffn_out)

        block_output = ffn_out

        return block_output, masked_attention_block, attention_block


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size: int, model_size: int, h: int = 8, num_decoder: int = 2):
        """ 
        Architecture of the Decoder

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary, e.g. maximum number of words in the output language
        model_size : int
            Depth of the embedding, e.g size of each vector representing a token
        h : int, default = 8
            Number of heads
        num_encoder : int, default = 2
            Number of decoders
        """
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.model_size = model_size
        self.h = h
        self.num_decoder = num_decoder

        self.decoder_blocks = [DecoderBlock(
            vocab_size, model_size, h) for _ in range(num_decoder)]

    def __call__(self, input: tf.Tensor, encoder_output: tf.Tensor, padding_mask: tf.Tensor = None):
        """
        Pass throught the Decoder

        Parameters
        ----------
        input : tf.Tensor, shape = (..., vocab_size, model_size)
            Input of the Decoder block, from another block or the Embedding layer
        encoder_output : tf.Tensor, shape = (..., vocab_size, model_size)
            Output of the Encoder block
        padding_mask : None or tf.Tensor, shape = ()
            TO COMPLETE

        Returns
        -------
        """

        output = input
        masked_attentions = []
        attentions = []

        for i in range(self.num_decoder):
            output, masked_attention, attention = self.decoder_blocks[i](
                output, encoder_output, padding_mask)
            masked_attentions.append(masked_attention)
            attentions.append(attention)

        masked_attention = tf.concat(masked_attentions, axis=-1)
        attention = tf.concat(attentions, axis=-1)

        return output, masked_attention, attention


if __name__ == '__main__':

    vocab_size_board = 500
    vocab_size_moves = 64*63
    model_size = 40

    encoder = Encoder(vocab_size=vocab_size_board, model_size=model_size)
    decoder = Decoder(vocab_size=vocab_size_moves, model_size=model_size)

    tokenizer = ChessTokenizer()
    dataset = import_data(filename="test.txt")

    embedder_boards = TextEmbedder(
        vocab_size=vocab_size_board, depth_emb=model_size)
    embedder_moves = TextEmbedder(
        vocab_size=vocab_size_moves, depth_emb=model_size)

    boards, move_to_play, moves_mem = (list(l) for l in zip(*dataset))

    tokenizer.fit_on_texts(boards)
    tok_boards = tokenizer.texts_to_sequences(boards[0:30], maxlen=15)

    tokenizer.fit_on_texts(moves_mem)
    tok_moves_mem = tokenizer.texts_to_sequences(moves_mem[0:30], maxlen=500)

    boards_embedded = embedder_boards(tok_boards)
    moves_mem_embedded = embedder_moves(tok_moves_mem)

    encoder_output, attention_encoder = encoder(boards_embedded)
    decoder_output, masked_attention_decoder, attention_decoder = decoder(
        moves_mem_embedded, encoder_output)

    print(encoder_output)
    print(decoder_output)

    print("ok")
