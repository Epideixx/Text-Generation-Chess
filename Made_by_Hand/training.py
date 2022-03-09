# Main file to train the Transformer model

import os
import re
import tokenizers
from tqdm import tqdm
import wandb
import tensorflow as tf
import time
import numpy as np

from transformer import Transformer

# wandb.init(project="Chess-Transformer", entity="epideixx")

transformer = Transformer()
dataset = transformer.import_data("test.txt")

NUM_EPOCHS = 100


for e in range(NUM_EPOCHS):
    for batch, (boards, encoded_move_to_play, encoded_mem_moves) in enumerate(dataset):

        # print(boards[1])
        # print(encoded_move_to_play[1])
        # print(encoded_mem_moves[1])

        loss = transformer.train_step(boards, encoded_mem_moves,
                                      encoded_move_to_play)

        # transformer.save()

        # wandb.log({"train_loss": loss})
