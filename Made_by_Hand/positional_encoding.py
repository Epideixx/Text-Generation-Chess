# ------------------------------------------------------
#                Positional Encoding
# ------------------------------------------------------

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


class PositionalEncoding(tf.keras.Model):

    def __init__(self, seq_length: int, depth: int):
        """
        Parameters
        ----------
        seq_lenght : int
            Maximum lenght of the text
        depth : int
            Depth of the Embedding layer
        """
        super().__init__()
        self.seq_length = seq_length
        self.depth = depth

    def __call__(self):
        """
        Returns
        -------
        PE : tf.Tensor, shape = (seq_length, depth)
            Position Encoder tensor
        """
        PE = np.zeros((self.seq_length, self.depth))
        for pos in range(self.seq_length):
            for i in range(self.depth):
                if i % 2 == 0:
                    PE[pos, i] = np.sin(pos / 10000**(i/self.depth))
                else:
                    PE[pos, i] = np.cos(pos / 10000 ** ((i - 1) / self.depth))

        PE = tf.constant(PE, dtype=tf.float32)

        return PE


if __name__ == '__main__':
    PE = PositionalEncoding(500, 32)
    pes = PE()
    print(pes)

    ax = sns.heatmap(pes)
    plt.show()
