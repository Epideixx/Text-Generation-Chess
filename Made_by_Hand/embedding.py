# ------------------------------------------------------
#                    Text Embedding
# ------------------------------------------------------

from dataclasses import dataclass
import tensorflow as tf
from tokenizer import ChessTokenizer


class TextEmbedder(tf.keras.Model):

    def __init__(self, vocab_size: int, depth_emb: int):
        """"
        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary, e.g. maximum number of words in the language
        depth_emb : int
            Depth of the embedding, e.g size of each vector representing a token
        """

        super().__init__()
        self.vocab_size = vocab_size
        self.depth = depth_emb
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=depth_emb)
        self.embedding.build(vocab_size)

    def __call__(self, texts_tokenized: tf.Tensor):
        """
        Parameters
        ----------
        texts_tokenized : tf.Tensor, shape = (batch_size, max_lenght_token)
            Text already tokenized

        Returns
        -------
        embedding : tf.Tensor, shape = (batch_size, max_lenght_token, depth_emb)
            Text embedded
        """
        embedding = self.embedding(texts_tokenized)
        return embedding


# Test
if __name__ == '__main__':
    tokenizer = ChessTokenizer()
    dataset = tokenizer.import_data(filename="test.txt")

    embedder = TextEmbedder(vocab_size=63*64, depth_emb=10)
    for batch, (boards, move_to_play, moves_mem) in enumerate(dataset):
        print(type(boards))
        embedded = embedder(boards)
        print(type(embedded))
        print(embedded)

        print(boards[0])
        print(embedded[0])

        print(boards[1])
        print(embedded[1])

        if batch >= 0:
            break

    print('ok')
