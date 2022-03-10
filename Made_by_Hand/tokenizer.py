import os
from matplotlib.pyplot import cla
from tqdm import tqdm
import tensorflow as tf


class ChessTokenizer(tf.keras.Model):

    def __init__(self):

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', split=' ', lower=False)

    def fit_on_texts(self, text):
        self.tokenizer.fit_on_texts(text)

    def __call__(self, text, maxlen=None):
        tokens = self.tokenizer.texts_to_sequences(text)
        tokens = tf.keras.preprocessing.sequence.pad_sequences(
            tokens, padding='post', maxlen=maxlen)

        return tokens


# Test
if __name__ == '__main__':
    tokenizer = ChessTokenizer()
    texts = ["a2g4 a6f5 d2d4 d8d2", "a2g4 5h6g, d4d5"]
    tokenizer.fit_on_texts(texts)
    phrase_test = ["a2g4 "]
    token = tokenizer(phrase_test)
    print(token)
    print(tokenizer.tokenizer.word_index)

    print(len(token))
    print(len(phrase_test))
