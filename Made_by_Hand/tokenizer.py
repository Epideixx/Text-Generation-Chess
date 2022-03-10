import os
from matplotlib.pyplot import cla
from tqdm import tqdm
import tensorflow as tf


class ChessTokenizer(tf.keras.Model):

    def __init__(self):

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='')

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
    texts = "Coucou je m'appelle Jonathan Poli et ceci est un test"
    tokenizer.fit_on_texts(texts)
    phrase_test = "Et je bosse avec José cette machine"
    token = tokenizer(phrase_test)
    print(token)
