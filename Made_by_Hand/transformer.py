# ------------------------------------------------------
#                   Transformer
# ------------------------------------------------------

import tensorflow as tf

from tokenizer import ChessTokenizer
from embedding import TextEmbedder
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder

from import_data import import_data


class Transformer(tf.keras.Model):

    def __init__(self, vocab_board=15, vocab_moves=500, model_size=10, max_moves_in_game=300, length_board=127, num_layers=2, h=8):
        """
        TO COMPLETE
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

        # To continue but first a small test
        output = self.final(output_decoder)

        return output


# Test
if __name__ == '__main__':

    # ---- Test 1 ----
    transfo = Transformer()

    dataset = import_data(filename="test.txt")
    boards, move_to_play, moves_mem = zip(*dataset)

    output = transfo(boards, moves_mem)
    print(output)

    print('ok')
