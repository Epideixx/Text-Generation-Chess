# ------------------------------------------------------
#                   Transformer
# ------------------------------------------------------

from black import out
import tensorflow as tf
import numpy as np
import wandb
import os
<<<<<<< HEAD
from tqdm import tqdm
import time
import pickle

# --------------------- ENCODER ------------------------

# Positional encoding


def positional_embedding(pos, model_size):
    """ Creates a vector to add on the embedded text to encode the position information"""

    PE = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            PE[:, i] = np.sin(pos / 1000 ** (i / model_size))
        else:
            PE[:, i] = np.cos(pos / 1000 ** ((i - 1) / model_size))
    return PE


# Check this part !

max_length = 500
MODEL_SIZE = 32

pes = []
for i in range(max_length):
    pes.append(positional_embedding(i, MODEL_SIZE))

pes = np.concatenate(pes, axis=0)
pes = tf.constant(pes, dtype=tf.float32)


# Multi-head Attention


class MultiHeadAttention(tf.keras.Model):

    def __init__(self, model_size, h):

        # Enable to avoid calling arguments for other functions ?
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h  # Share on different heads
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h  # Number of heads
        self.wq = [tf.keras.layers.Dense(self.query_size)
                   for _ in range(h)]  # Linear layers
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        # Final processing of the concatenated data
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, mask=None):
        """Apply the Multi-Head Attention algorithm"""
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        key = value

        heads = []

        # For each head we apply the Scaled Dot-Product Attention described in the original paper
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query),
                              self.wk[i](key), transpose_b=True)

            # Here we scale the score as described in the paper
            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, query_len, value_len)

            # mask must be broadcastable to (batch, query_len, value_len)
            if mask is not None:
                score *= mask

                # asign masked positions to -1e9
                # so that their values after softmax are zeros
                score = tf.where(tf.equal(score, 0),
                                 tf.ones_like(score) * -1e9, score)

            alignment = tf.nn.softmax(score, axis=2)
            # alignment has shape (batch, query_len, value_len)

            head = tf.matmul(alignment, self.wv[i](value))
            # head has shape (batch, decoder_len, value_size)
            heads.append(head)

        # Concatenate all the attention heads
        # so that the last dimension summed up to model_size
        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)
        # heads has shape (batch, query_len, model_size)
        return heads


# Encoder

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        """ Architecture of the Encoder"""
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.pes = pes

        # One Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, model_size)  # TO EXPLORE

        # num_layers Multi-Head Attention and Normalization layers
        self.norm_input = tf.keras.layers.BatchNormalization()
        self.attention = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.BatchNormalization()
                               for _ in range(num_layers)]

        # num_layers FFN and Normalization layers
        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='sigmoid') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size, activation='sigmoid') for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization()
                         for _ in range(num_layers)]

        # NOTE : Etonnant de pas lier avant les couches genre avec des add etc ..;

    def call(self, sequence, padding_mask=None):

        # EMBEDDING + POSITION

        embed_out = self.embedding(sequence)
        embed_out += pes[:sequence.shape[1], :]

        sub_in = embed_out
        sub_in = self.norm_input(sub_in)

        # MULTIHEAD ATTENTION
        # We will have num_layers of (Attention + FFN)
        for i in range(self.num_layers):

            sub_out = self.attention[i](sub_in, sub_in, padding_mask)
            sub_out = sub_in + sub_out  # Residual connection
            sub_out = self.attention_norm[i](sub_out)  # Normalization

            # FEED FORWARD IN THE NN
            # The FFN input is the output of the Multi-Head Attention
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            # Add the residual connection
            ffn_out = ffn_in + ffn_out
            # Normalize the output
            ffn_out = self.ffn_norm[i](ffn_out)

            # Assign the FFN output to the next layer's Multi-Head Attention input
            sub_in = ffn_out

        # Return the result when done
        return ffn_out

    def save(self, file="encoder"):

        file = os.path.join(os.path.dirname(__file__), file)

        if not os.path.exists(file):
            os.makedirs(file)
=======
>>>>>>> Refactor

from tokenizer import ChessTokenizer
from embedding import TextEmbedder
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from metrics import MaskedAccuracy, MaskedSparseCategoricalEntropy


from import_data import import_data


class Transformer(tf.keras.Model):

<<<<<<< HEAD
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        """ Decoder architecture """

        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers  # Number of layers of decoders
        self.h = h  # Number of heads
        self.pes = pes
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.norm_input = tf.keras.layers.BatchNormalization()

        self.attention_bot = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [
            tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [
            tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
=======
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
>>>>>>> Refactor

        """
        super(Transformer, self).__init__()
        self.max_moves_in_game = max_moves_in_game
        self.length_board = length_board

        self.encoder_embedding = TextEmbedder(
            vocab_size=vocab_board, depth_emb=model_size)
        self.encoder_PE = PositionalEncoding(
            seq_length=length_board, depth=model_size)
        self.encoder = Encoder(
            vocab_size=vocab_board, model_size=model_size, h=h, num_encoder=num_layers)

        self.decoder_embedding = TextEmbedder(
            vocab_size=vocab_moves, depth_emb=model_size)
        self.decoder_PE = PositionalEncoding(
            seq_length=max_moves_in_game, depth=model_size)
        self.decoder = Decoder(
            vocab_size=vocab_moves, model_size=model_size, h=h, num_decoder=num_layers)

        # Final layer to associate the data to one word
<<<<<<< HEAD
        self.dense = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, sequence, encoder_output, padding_mask=None):

        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)
        embed_out += pes[:sequence.shape[1], :]

        bot_sub_in = embed_out

        bot_sub_in = self.norm_input(bot_sub_in)

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            bot_sub_out = []

            bot_sub_out = self.attention_bot[i](
                bot_sub_in, bot_sub_in, padding_mask)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = self.attention_mid[i](
                mid_sub_in, encoder_output)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        # Flatten because my Transformer is very different concerning input decoder and output decoder
        ffn_out = tf.keras.layers.Flatten()(ffn_out)
        logits = tf.concat([tf.expand_dims(self.dense(ffn_out), axis=1)
                           for _ in range(2)], axis=1)  # Because move + <end>

        return logits

    def save(self, file="decoder"):

        file = os.path.join(os.path.dirname(__file__), file)

        if not os.path.exists(file):
            os.makedirs(file)

        self.save_weights(file)

    def load(self, file="decoder"):

        self.load_weights(file)

=======
        self.final = tf.keras.layers.Dense(vocab_moves, activation="softmax")
>>>>>>> Refactor

        # For training ==> TO MAKE EVOLVE
        self.optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)
        self.accuracy = MaskedAccuracy()
        self.loss = MaskedSparseCategoricalEntropy()

    def call(self, input):
        """
        Parameters
        ----------
        input :  (tf.Tensor, tf.Tensor)
            (Tokenized input of the encoder, Tokenized input of the decoder

        Returns
        -------
        output : tf.Tensor
            TO COMPLETE
        """
        input_encoder, input_decoder = input
        tok_encoder = input_encoder
        emb_encoder = self.encoder_embedding(tok_encoder)
        pes_encoder = self.encoder_PE()
        in_encoder = emb_encoder + pes_encoder
        output_encoder, attention_encoder = self.encoder(
            in_encoder, padding_mask=None)  # For the moment

        tok_decoder = input_decoder
        emb_decoder = self.decoder_embedding(tok_decoder)
        pes_decoder = self.decoder_PE()
        in_decoder = emb_decoder + pes_decoder
        output_decoder, masked_attention_decoder, attention_decoder = self.decoder(
            in_decoder, output_encoder, padding_mask=None)  # For the moment

        # Not sure about this part
        output = tf.keras.layers.Flatten()(output_decoder)
        output = self.final(output)

<<<<<<< HEAD
        self.max_moves_in_game = max_moves_in_game

        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        self.optimizer = tf.keras.optimizers.Adam()

    def import_data(self, filename="fen.txt", batchsize=32):

        filename = os.path.join(os.path.dirname(__file__), filename)

        boards = []
        moves_to_play = []
        mem_moves = []

        with open(filename) as f:
            for line in tqdm(f, desc="read fen.txt", unit=" moves", mininterval=1):
                if line:
                    board_move = [string.strip() for string in line.split('-')]
                    boards.append(board_move[0])
                    moves_to_play.append(board_move[1] + " <end>")
                    mem_moves.append("<start> " + board_move[2])

        self.tokenizer_boards = tf.keras.preprocessing.text.Tokenizer(
            filters='')
        self.tokenizer_boards.fit_on_texts(boards)
        boards = self.tokenizer_boards.texts_to_sequences(boards)
        boards = tf.keras.preprocessing.sequence.pad_sequences(
            boards, padding='post')

        self.tokenizer_moves = tf.keras.preprocessing.text.Tokenizer(
            filters='')
        self.tokenizer_moves.fit_on_texts(moves_to_play)
        self.tokenizer_moves.fit_on_texts(mem_moves)

        moves_to_play = self.tokenizer_moves.texts_to_sequences(moves_to_play)
        moves_to_play = tf.keras.preprocessing.sequence.pad_sequences(
            moves_to_play, padding='post', maxlen=2)

        mem_moves = self.tokenizer_moves.texts_to_sequences(mem_moves)
        mem_moves = tf.keras.preprocessing.sequence.pad_sequences(
            mem_moves, padding='post', maxlen=self.max_moves_in_game)

        dataset = tf.data.Dataset.from_tensor_slices(
            (boards, moves_to_play, mem_moves))
        dataset = dataset.shuffle(20).batch(batchsize)
=======
        return output
>>>>>>> Refactor

    def predict(self, input_encoder: tf.Tensor, input_decoder: tf.Tensor):
        """
        Parameters
        ----------
        input_encoder : tf.Tensor
            Tokenized input of the encoder
        input_decoder : tf.Tensor
            Tokenized input of the decdoder

        Returns
        -------
        output : tf.Tensor
            TO COMPLETE
        """
        input = input_encoder, input_decoder
        output = self(input=input)
        output = tf.argmax(output, axis=-1)

        output = [output.numpy()[i][0] for i in range(output.shape[0])]

        return output

    def train_step(self, encoder_inputs, transfo_real_outputs, decoder_inputs):

        with tf.GradientTape() as tape:

            input = encoder_inputs, decoder_inputs
            transfo_predict_outputs = self(input=input)
            loss = self.loss(transfo_real_outputs,
                             transfo_predict_outputs)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        accuracy = self.accuracy(
            transfo_real_outputs, transfo_predict_outputs)

        return loss, accuracy

    def fit(self, x: tf.Tensor, y: tf.Tensor, batch_size: int = 32, num_epochs: int = 1, wandb_api=True):

        if wandb_api:
            wandb.init(project="Chess-Transformer", entity="epideixx")

        dataset = tf.data.Dataset.zip((x, y))
        dataset = dataset.shuffle(32000).batch(batch_size=batch_size)

        for _ in range(num_epochs):

            for batch, ((encoder_inputs, decoder_inputs), transfo_real_outputs) in enumerate(dataset):

                loss, accuracy = self.train_step(
                    encoder_inputs, transfo_real_outputs, decoder_inputs)
                if wandb_api:
                    wandb.log({"train_loss": loss, "train_accuracy": accuracy})

                if batch % 30 == 0:
                    if not os.path.exists(os.path.join(os.path.dirname(__file__), "test_transfo")):
                        os.makedirs(os.path.join(
                            os.path.dirname(__file__), "test_transfo"))
                    filename = os.path.join(os.path.dirname(__file__),
                                            "test_transfo", "test_transfo")
                    self.save_weights(filename)


# Test
if __name__ == '__main__':

    # ---- Test 1 ----
    """
    transfo = Transformer(vocab_moves=570)

    dataset = import_data(filename="test.txt")
    boards, move_to_play, moves_mem = zip(*dataset)

    output = transfo.predict(boards[0:45], moves_mem[0:45])
    print("Le transformateur prédit : ", output)

    print('ok')
    """

    # Vérifier les shapes ... ==> En gros faut juste se démerder pour flatten ou un truc comme ça
    # Penser à supprimer les print dans le code

    # ---- Test 2 ----
    length_board = 64
    max_moves_in_game = 300
    vocab_moves = 64*(7*4 + 8)

    transfo = Transformer(vocab_moves=vocab_moves,
                          length_board=length_board, num_layers=4)

    dataset = import_data(filename="test.txt")

    data = dataset[:45]
    enc_input = [x[0] for x in dataset[:45]]
    dec_input = [x[2] for x in dataset[:45]]

    encoder_tokenize = ChessTokenizer()
    decoder_tokenize = ChessTokenizer()

    encoder_tokenize.fit_on_texts(enc_input)
    decoder_tokenize.fit_on_texts(dec_input)

    enc_input = encoder_tokenize.texts_to_sequences(
        enc_input, maxlen=length_board)
    dec_input = decoder_tokenize.texts_to_sequences(
        dec_input, maxlen=max_moves_in_game)

    output_decoder = transfo.call((enc_input, dec_input))

    transfo.build(input_shape=[enc_input.shape, dec_input.shape])  # Chelou ...

    print("ok")
