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

        super(TextEmbedder, self).__init__()
        self.vocab_size = vocab_size
        self.depth = depth_emb
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=depth_emb)
        self.embedding.build(vocab_size)

    def call(self, texts_tokenized: tf.Tensor):
        """
        Parameters
        ----------
        texts_tokenized : tf.Tensor, shape = (..., max_lenght_token)
            Text already tokenized

        Returns
        -------
        embedding : tf.Tensor, shape = (..., max_lenght_token, depth_emb)
            Text embedded
        """
        embedding = self.embedding(texts_tokenized)
        return embedding


# Test
if __name__ == '__main__':
    tokenizer = ChessTokenizer()
    texts = ["a2g4 a6f5 d2d4 d8d2", "a2g4 5h6g, d4d5"]
    tokenizer.fit_on_texts(texts)
    phrase_test = ["a2g4 "]
    token = tokenizer.texts_to_sequences(phrase_test)

    embedder = TextEmbedder(vocab_size=60, depth_emb=10)
    embedded = embedder(token)
    print(embedded)

    print('ok')
