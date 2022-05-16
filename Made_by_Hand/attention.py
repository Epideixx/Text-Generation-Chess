# ------------------------------------------------------
#               Multi-Head Attention
# ------------------------------------------------------


from shutil import move
import tensorflow as tf
from embedding import TextEmbedder
from tokenizer import ChessTokenizer
from import_data import import_data


class MultiHeadAttention(tf.keras.Model):

    def __init__(self, model_size: int, h: int = 8):
        """
        Parameters
        ----------
        model_size : int
            Depth of the model
        h : int, default = 8
            Number of heads, set to 8 by default as in the original paper

        """

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

    def call(self, Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor, mask: tf.Tensor = None):
        """
        Apply the Multi-Head Attention algorithm

        Parameters
        ----------
        Q : tf.Tensor, shape = (batch_size, len_Q, model_size)
            Query (full length)
        K : tf.Tensor, shape = (batch_size, len_K, model_size)
            Key (full length)
        V : tf.Tensor, shape = (batch_size, len_V, model_size)
            Value (full length)
        mask : tf.Tensor or None, shape = (batch, len_Q, len_V)
            TO COMPLETE

        Returns
        -------
        output : tf.Tensor, shape = (batch_size, len_Q, model_size)
            Output of the Multi-Head Attention block
        attention : tf.Tensor, shape = (h, batch_size, len_Q, len_K)
            Attention weights on different heads

        """

        if Q.get_shape()[-1] != K.get_shape()[-1]:
            raise ValueError("The last dimension of Q and K must be equal, "
                             f"found {Q.get_shape()[-1]} and "
                             f"{K.get_shape()[-1]}.")

        if K.get_shape()[-2] != V.get_shape()[-2]:
            raise ValueError("The last dimension of Q/K and the first dimension of V must be equal, "
                             f"found {K.get_shape()[-2]} and "
                             f"{V.get_shape()[-2]}.")

        heads_output = []
        heads_attention = []

        # For each head we apply the Scaled Dot-Product Attention described in the original paper
        for i in range(self.h):
            attention = tf.matmul(self.wq[i](Q),
                                  self.wk[i](K), transpose_b=True)

            # Here we scale the score as described in the paper
            attention /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))
            # score has shape (batch, query_len, key_len)

            # mask must be broadcastable to (batch, query_len, value_len)
            if mask is not None:

                # cast mask to binary tensor (0.0 or 1.0)
                mask = tf.cast(tf.cast(mask, tf.bool), tf.float32)
                # set logits to -inf where mask=0 to ignore them
                # during packpropagation
                attention += (1.0 - mask) * -1e9

            attention_head = tf.nn.softmax(attention, axis=-1)
            # alignment has shape (batch, query_len, key_len)

            output_head = attention_head @ self.wv[i](V)
            # head has shape (batch, decoder_len, value_size)
            heads_output.append(output_head)
            heads_attention.append(attention_head)

        # Concatenate all the attention heads
        # so that the last dimension summed up to model_size
        heads_output = tf.concat(heads_output, axis=2)

        attention = tf.stack(heads_attention)
        output = self.wo(heads_output)

        # output has shape (batch, len_Q, model_size)
        return output, attention


# Tests
if __name__ == '__main__':

    tokenizer = ChessTokenizer()
    dataset = import_data(filename="test.txt")

    embedder = TextEmbedder(vocab_size=63*64, depth_emb=10)
    multi_attention = MultiHeadAttention(model_size=40)

    boards, move_to_play, moves_mem = (list(l) for l in zip(*dataset))
    print(boards[0])
    print(move_to_play[0])

    tokenizer.fit_on_texts(boards)
    tokenized_boards = tokenizer.texts_to_sequences(boards[0:15])
    print(len(tokenized_boards))
    print(tokenized_boards)
    embedded_boards = embedder(tokenized_boards)
    print(embedded_boards)
    print(tokenizer.tokenizer.word_index)

    tokenizer.fit_on_texts(move_to_play)
    tokenized_moves = tokenizer.texts_to_sequences(move_to_play[0:15])
    embedded_moves = embedder(tokenized_moves)
    print(embedded_moves[0])
    output, attention = multi_attention(
        embedded_moves, embedded_boards, embedded_boards)

    print(output[0])
    print(attention[0])

    print(embedder.summary())
    print(multi_attention.summary())

    print('ok')
