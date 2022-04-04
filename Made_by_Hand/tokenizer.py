import os
from matplotlib.pyplot import cla
from tqdm import tqdm
import tensorflow as tf


class ChessTokenizer(tf.keras.preprocessing.text.Tokenizer):

    def __init__(self):

        super(ChessTokenizer, self).__init__()
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', lower=False)

    def fit_on_texts(self, texts):
        """
        Updates internal vocabulary 

        Parameters
        ----------
        texts : list of string
            List of boards or moves to update the tokenizer
        """
        if not(type(texts) == list or type(texts) == str):
            texts = texts.numpy().tolist()
            texts = [e.decode("utf-8") for e in texts]
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts, maxlen=None):
        """
        Transforms each text from texts in a list of integers which are the tokens

        Parameters
        ----------
        texts : list of string
            Texts to be tokenized
        maxlen : None or int
            Maximum length of the sequence.
            If not None, every text will be convert to a sequence of maxlen integers
        """
        if not(type(texts) == list or type(texts) == str):
            texts = texts.numpy().tolist()
            texts = [e.decode("utf-8") for e in texts]
        tokens = self.tokenizer.texts_to_sequences(texts)
        tokens = tf.keras.preprocessing.sequence.pad_sequences(
            tokens, padding='post', maxlen=maxlen)

        return tokens


# Test
if __name__ == '__main__':
    tokenizer = ChessTokenizer()
    texts = ["a2g4 a6f5 d2d4 d8d2", "a2g4 5h6g, d4d5"]
    tokenizer.fit_on_texts(texts)
    phrase_test = ["a2g4 a6f5"]
    token = tokenizer.texts_to_sequences(phrase_test)
    print(token)
    print(tokenizer.tokenizer.word_index)

    print(len(token))
    print(len(phrase_test))
