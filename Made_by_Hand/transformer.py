# Dependencies

from chess import Board
import numpy as np
import tensorflow as tf
import os
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

        self.save_weights(file)

    def load(self, file="encoder"):

        self.load_weights(file)


# --------------------- DECODER ------------------------

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

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='sigmoid') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size, activation='sigmoid') for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.BatchNormalization()
                         for _ in range(num_layers)]

        # Final layer to associate the data to one word
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


class Transformer():

    def __init__(self, vocab_board=34, vocab_moves=4034, model_size=MODEL_SIZE, max_moves_in_game=500, num_layers=1, h=1):
        """
        vocab_bard = 34 because 2*16 pieces + empty + security
        vocab_moves = 4034 because 64*63 possible moves + <start> + <end>
        max_moves_in_game = 500 because we will not treat the case of very long games for the moment
        """
        self.encoder = Encoder(vocab_board, model_size, num_layers, h)
        self.decoder = Decoder(vocab_moves, model_size, num_layers, h)

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

        return dataset

    def loss_func(self, targets, logits):

        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = self.crossentropy(targets, logits, sample_weight=mask)

        return loss

    def train_step(self, source_seq, target_seq_in, target_seq_out):
        with tf.GradientTape() as tape:

            encoder_output = self.encoder(source_seq)

            mask = 1 - tf.cast(tf.equal(target_seq_in, 0), dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.matmul(mask, mask, transpose_a=True)

            decoder_output = self.decoder(target_seq_in, encoder_output, mask)

            loss = self.loss_func(target_seq_out, decoder_output)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        # print(gradients)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def predict(self, board_to_predict, mem_moves):

        board = self.tokenizer_boards.texts_to_sequences([board_to_predict])
        encoder_output = self.encoder(tf.constant(board))
        decoder_input = tf.constant(
            list(self.tokenizer_moves.texts_to_sequences([mem_moves])), dtype=tf.int64)

        decoder_input = tf.keras.preprocessing.sequence.pad_sequences(
            decoder_input, padding='post', maxlen=self.max_moves_in_game)

        mask = 1 - tf.cast(tf.equal(decoder_input, 0), dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.matmul(mask, mask, transpose_a=True)

        decoder_output = self.decoder(decoder_input, encoder_output, mask)

        output = tf.expand_dims(tf.argmax(decoder_output, -1)[:, -1], axis=1)

        try:
            output = self.tokenizer_moves.index_word[output.numpy()[0][0]]
        except KeyError:
            output = None
        return output

    def train(self, file_data, epochs=10):

        dataset = self.import_data(file_data)

        for e in range(epochs):
            for batch, (boards, encoded_move_to_play, encoded_mem_moves) in enumerate(dataset.take(-1)):

                loss = self.train_step(boards, encoded_mem_moves,
                                       encoded_move_to_play)

            print('Epoch {} Loss {:.4f}'.format(e + 1, loss.numpy()))

    def save(self, folder="transformer"):

        folder = os.path.join(os.path.dirname(__file__), folder)

        if not os.path.exists(folder):
            os.makedirs(folder)

        encoder = os.path.join(folder, "encoder")
        self.encoder.save(file=encoder)

        decoder = os.path.join(folder, "decoder")
        self.decoder.save(file=decoder)

        # other_parameters = os.path.join(folder, "optimizer")
        # tf.saved_model.save(
        #     self.optimizer, export_dir=other_parameters)

    def load(self, folder="transformer"):

        encoder = os.path.join(folder, "encoder")
        self.encoder.load(encoder)

        decoder = os.path.join(folder, "decoder")
        self.decoder.load(decoder)

        # other_parameters = os.path.join(folder, "optimizer")
        # self.optimizer = tf.saved_model.load(
        #     other_parameters)


if __name__ == '__main__':
    transformer = Transformer()

    encoder = transformer.encoder
    encoder.save_weights("test_encoder_weights")
    encoder.load_weights("test_encoder_weights")

    print("ok")
