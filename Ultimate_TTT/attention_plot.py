import tensorflow as tf
from matplotlib import pyplot as plt
import os
from time import sleep

from player_Transformer import Transfo_player

folder_name = "Test_26_05_12h15"
folder = os.path.join(os.path.dirname(__file__), folder_name)
transfo = Transfo_player(folder)

board = "|xox|...|oo.|||o.o|.x.|xx.|||oo.|.x.|...|||..x|x..|x..|||.x.|o..|oo.|||xxo|.o.|...|||o.o|x.o|x..|||o..|..x|..x|||xo.|xox|x.o||||||...x....."
move_to_predict = "E1" 
mem_moves = "I6 I1 I5 I8 G6 B2 D6 C1 G3 A1 C4 G5 B7 F4 I3 H0 F1 I4 H5 E6 D2 A6 A0 B0 F0 G2 B6 F2 H8 E7 D3 C0 I0 G0 A2 A7 B4 E3"

board =  "|...|...|x..|||...|...|...|||..o|...|...|||...|..o|...|||...|...|...|||...|o..|...|||...|...|...|||.x.|...|.x.|||...|...|...||||||........."
move_to_predict = "H1"
mem_moves = "D5 A6 C2 H7 F3"

print("Board : ", board)
print("Previous moves : ", mem_moves)
print("Move to predict : ", move_to_predict)

predicted_moves, attention_encoder, masked_attention_decoder, attention_decoder = transfo.choose_move(board, mem_moves)

print("Predicted moves :", predicted_moves)


def plot_attention_head(in_tokens, translated_tokens, attention):
    # The plot is of the attention when a token was generated.
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]

    ax = plt.gca()
    ax.matshow(attention)
    labels_x = in_tokens.split(" ")
    labels_y = translated_tokens.split(" ")

    ax.set_xticks(range(len(labels_x)))
    ax.set_yticks(range(len(labels_y)))

    
    ax.set_xticklabels(labels_x, rotation=90)

    ax.set_yticklabels(labels_y)

print(masked_attention_decoder)

layer = 0
head = 0
# shape: (num_layers, num_heads, seq_len_q, seq_len_k)

# attention = masked_attention_decoder[layer][head]
# print(attention)

# plot_attention_head(mem_moves, mem_moves, attention)

def plot_attention_heads(in_tokens, translated_tokens, attention_heads):
    
    fig = plt.figure(figsize=(16, 8))

    for h, head in enumerate(attention_heads):
        ax = fig.add_subplot(2, 4, h+1)

        plot_attention_head(in_tokens, translated_tokens, head)

        ax.set_xlabel(f'Head {h+1}')

    plt.tight_layout()
    plt.show()

attention = masked_attention_decoder[layer]
plot_attention_heads(mem_moves, mem_moves, attention)