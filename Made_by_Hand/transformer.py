# ------------------------------------------------------
#                   Transformer
# ------------------------------------------------------

import tensorflow as tf
import numpy as np

from tokenizer import ChessTokenizer
from embedding import TextEmbedder
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

from import_data import import_data


class Transformer(tf.keras.Model):

    def __init__(self, vocab_board=15, vocab_moves=2000, model_size=10, max_moves_in_game=300, length_board=127, num_layers=2, h=8):
        """
        Parameters
        ----------
        vocab_board : int, default = 15
            Vocab size of the chess pieces
        vocab_moves : int, default = 2000
            Vocab size of possible moves
        model_size : int, default = 10
            Depth of the embedding model
        max_moves_in_game : int, default = 300
            Max number of moves in a game
        length_board : int, default = 127
            Size of the encoding of the board
        num_layers : int, default = 2
            Number of encoders et of decoders
        h : int, default = 10
            Number of heads in the Multi Head Attention method

        """
        super(Transformer, self).__init__()
        self.max_moves_in_game = max_moves_in_game
        self.length_board = length_board

        self.encoder_tokenize = ChessTokenizer()
        self.encoder_embedding = TextEmbedder(
            vocab_size=vocab_board, depth_emb=model_size)
        self.encoder_PE = PositionalEncoding(
            seq_length=length_board, depth=model_size)
        self.encoder = Encoder(
            vocab_size=vocab_board, model_size=model_size, h=h, num_encoder=num_layers)

        self.decoder_tokenize = ChessTokenizer()
        self.decoder_embedding = TextEmbedder(
            vocab_size=vocab_moves, depth_emb=model_size)
        self.decoder_PE = PositionalEncoding(
            seq_length=max_moves_in_game, depth=model_size)
        self.decoder = Decoder(
            vocab_size=vocab_moves, model_size=model_size, h=h, num_decoder=num_layers)

        # Final layer to associate the data to one word
        self.final = tf.keras.layers.Dense(vocab_moves, activation="softmax")

    def __call__(self, input_encoder: tf.Tensor, input_decoder: tf.Tensor):
        """
        Parameters
        ----------
        input_encoder : tf.Tensor
            Textual input of the encoder
        input_decoder : tf.Tensor
            Textual input of the decdoder

        Returns
        -------
        output : tf.Tensor
            TO COMPLETE
        """

        self.encoder_tokenize.fit_on_texts(input_encoder)
        tok_encoder = self.encoder_tokenize(
            input_encoder, maxlen=self.length_board)
        emb_encoder = self.encoder_embedding(tok_encoder)
        pes_encoder = self.encoder_PE()
        in_encoder = emb_encoder + pes_encoder
        output_encoder, attention_encoder = self.encoder(
            in_encoder, padding_mask=None)  # For the moment

        self.decoder_tokenize.fit_on_texts(input_decoder)
        tok_decoder = self.decoder_tokenize(
            input_decoder, maxlen=self.max_moves_in_game)
        emb_decoder = self.decoder_embedding(tok_decoder)
        pes_decoder = self.decoder_PE()
        in_decoder = emb_decoder + pes_decoder
        output_decoder, masked_attention_decoder, attention_decoder = self.decoder(
            in_decoder, output_encoder, padding_mask=None)  # For the moment

        print("Juste pour vérifier : ", output_decoder.shape)

        # Not sure about this part
        output = tf.keras.layers.Flatten()(output_decoder)
        output = tf.concat([tf.expand_dims(self.final(output), axis=1)
                           for _ in range(2)], axis=1)  # Because move + <end>

        return output

    def predict(self, input_encoder: tf.Tensor, input_decoder: tf.Tensor):
        """
        Parameters
        ----------
        input_encoder : tf.Tensor
            Textual input of the encoder
        input_decoder : tf.Tensor
            Textual input of the decdoder

        Returns
        -------
        output : tf.Tensor
            TO COMPLETE
        """

        output = self.__call__(input_encoder=input_encoder,
                               input_decoder=input_decoder)
        output = tf.argmax(output, axis=-1)
        print(output.numpy())
        print(self.decoder_tokenize.tokenizer.index_word)
        output = tf.concat([self.decoder_tokenize.tokenizer.index_word[output.numpy()[
            i][0]] for i in range(output.shape[0])], axis=0)

        return output


# Test
if __name__ == '__main__':

    # ---- Test 1 ----
    transfo = Transformer(vocab_moves=570)

    dataset = import_data(filename="test.txt")
    boards, move_to_play, moves_mem = zip(*dataset)

    output = transfo.predict(boards[0:45], moves_mem[0:45])
    print("Le transformateur prédit : ", output)

    print('ok')

    # Vérifier les shapes ... ==> En gros faut juste se démerder pour flatten ou un truc comme ça
    # Penser à supprimer les print dans le code
