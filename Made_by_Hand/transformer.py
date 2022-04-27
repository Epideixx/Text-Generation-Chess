# ------------------------------------------------------
#                   Transformer
# ------------------------------------------------------

import tensorflow as tf
import numpy as np
import wandb
import os
from tqdm import tqdm

from tokenizer import ChessTokenizer
from embedding import TextEmbedder
from positional_encoding import PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from metrics import MaskedAccuracy, MaskedSparseCategoricalEntropy, ClassicAccuracy


from import_data import import_data


class Transformer(tf.keras.Model):

    def __init__(self, vocab_board=15, vocab_moves=2000, model_size=10, max_moves_in_game=300, length_board=127, num_layers=2, h=8, dropout=0.0):
        """
        Parameters
        ----------
        vocab_board : int, default = 15
            Vocab size of the chess pieces
        vocab_moves : int, default = 2000
            Vocab size of possible moves
        model_size : int, default = 10
            Depth of the embedding model
        max_moves_in_game : int, default = 300
            Max number of moves in a game
        length_board : int, default = 127
            Size of the encoding of the board
        num_layers : int, default = 2
            Number of encoders et of decoders
        h : int, default = 10
            Number of heads in the Multi Head Attention method

        """
        super(Transformer, self).__init__()
        self.max_moves_in_game = max_moves_in_game
        self.length_board = length_board
        self.dropout = dropout

        self.encoder_embedding = TextEmbedder(
            vocab_size=vocab_board, depth_emb=model_size)
        # print("Allez allez :")
        # print(self.encoder_embedding.trainable_variables)
        self.encoder_PE = PositionalEncoding(
            seq_length=length_board, depth=model_size)
        self.encoder = Encoder(
            vocab_size=vocab_board, model_size=model_size, h=h, num_encoder=num_layers, dropout=dropout)

        self.decoder_embedding = TextEmbedder(
            vocab_size=vocab_moves, depth_emb=model_size)
        self.decoder_PE = PositionalEncoding(
            seq_length=max_moves_in_game, depth=model_size)
        self.decoder = Decoder(
            vocab_size=vocab_moves, model_size=model_size, h=h, num_decoder=num_layers, dropout=dropout)

        # Final layer to associate the data to one word
        self.final = tf.keras.layers.Dense(vocab_moves, activation="softmax")

        # For training ==> TO MAKE EVOLVE
        self.optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98,
                                                  epsilon=1e-9)
        self.accuracy = ClassicAccuracy()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def call(self, input, training: bool = False):
        """
        Parameters
        ----------
        input :  (tf.Tensor, tf.Tensor)
            (Tokenized input of the encoder, Tokenized input of the decoder

        Returns
        -------
        output : tf.Tensor
            TO COMPLETE
        """
        input_encoder, input_decoder = input
        tok_encoder = input_encoder
        # print("Test 0 : ", self.encoder_embedding.trainable_variables)
        emb_encoder = self.encoder_embedding(tok_encoder)
        # print("Test 1 : ", self.encoder_embedding.trainable_variables)
        pes_encoder = self.encoder_PE()
        in_encoder = emb_encoder + pes_encoder
        mask_encoder = tf.expand_dims(tf.cast(tf.math.logical_not(tf.math.equal(
            input_encoder, 0)), tf.float32), -1)
        mask_encoder = tf.matmul(
            mask_encoder, mask_encoder, transpose_b=True)  # To solve padding
        output_encoder, attention_encoder = self.encoder(
            in_encoder, padding_mask=mask_encoder, training=training)

        tok_decoder = input_decoder
        emb_decoder = self.decoder_embedding(tok_decoder)
        pes_decoder = self.decoder_PE()
        in_decoder = emb_decoder + pes_decoder
        mask_decoder = tf.expand_dims(tf.cast(tf.math.logical_not(tf.math.equal(
            input_decoder, 0)), tf.float32), -1)
        mask_decoder = tf.matmul(
            mask_decoder, mask_decoder, transpose_b=True)  # To solve padding
        output_decoder, masked_attention_decoder, attention_decoder = self.decoder(
            in_decoder, output_encoder, padding_mask=mask_decoder, training=training)

        output = self.final(output_decoder)

        return output

    def predict(self, input_encoder: tf.Tensor, input_decoder: tf.Tensor):
        """
        Parameters
        ----------
        input_encoder : tf.Tensor
            Tokenized input of the encoder
        input_decoder : tf.Tensor
            Tokenized input of the decdoder

        Returns
        -------
        output : tf.Tensor
            TO COMPLETE
        """
        input = input_encoder, input_decoder
        output = self(input=input)
        output = tf.argmax(output, axis=-1)

        output = output.numpy()

        return output

    def train_step(self, encoder_inputs, transfo_real_outputs, decoder_inputs):

        with tf.GradientTape() as tape:

            input = encoder_inputs, decoder_inputs
            transfo_predict_outputs = self(input=input, training=True)
            loss = self.loss(transfo_real_outputs,
                             transfo_predict_outputs, sample_weight = None)


        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))
        accuracy = self.accuracy(
            transfo_real_outputs, transfo_predict_outputs)

        return loss, accuracy

    def fit(self, x: tf.Tensor, y: tf.Tensor, batch_size: int = 32, num_epochs: int = 1, validation_split = 0.0, wandb_api=True, file_to_save = None):

        if wandb_api:
            wandb.init(project="Chess-Transformer", entity="epideixx")
        
        if validation_split:
            train_size = int((1-validation_split) * len(x))
        else :
            train_size = len(x)

        x_train = x.take(train_size)
        y_train = y.take(train_size)
        x_valid = x.skip(train_size)
        y_valid = y.skip(train_size)
        
        train_dataset = tf.data.Dataset.zip((x_train, y_train))


        train_dataset = train_dataset.shuffle(len(train_dataset)).batch(batch_size=batch_size)

        for epoch in range(num_epochs):

            for batch, ((encoder_inputs, decoder_inputs), transfo_real_outputs) in tqdm(enumerate(train_dataset), desc="Epoch " + str(epoch), unit=" batch", mininterval=1, total = len(train_dataset)):
                
                # emb_weights = self.encoder_embedding.get_weights()
                loss, accuracy = self.train_step(
                    encoder_inputs, transfo_real_outputs, decoder_inputs)
                # print([self.encoder_embedding.get_weights()[i] - emb_weights[i] for i in range(len(emb_weights))])
                if wandb_api:
                    wandb.log({"train_loss": loss, "train_accuracy": accuracy})

                if file_to_save :
                    if batch % 200 == 0: # To edit
                        if not os.path.exists(file_to_save):
                            os.makedirs(file_to_save)
                        print(file_to_save)
                        filename = os.path.join(file_to_save, "model_weights")
                        self.save_weights(filename)
            

            # Validation set
            if validation_split :
                valid_dataset = tf.data.Dataset.zip((x_valid, y_valid)).batch(batch_size=len(x_valid))
                for batch, ((encoder_inputs, decoder_inputs), transfo_real_outputs) in enumerate(valid_dataset):
                    
                    predict_outputs = self(input=(encoder_inputs, decoder_inputs), training=False)
                    loss = self.loss(transfo_real_outputs, predict_outputs)
                    accuracy = self.accuracy(transfo_real_outputs, predict_outputs)
                    if wandb_api:
                        wandb.log({"valid_loss": loss, "valid_accuracy": accuracy})
                    print("Validation loss : ", loss)
                    print("Validation accuracy : ", accuracy)




# Test
if __name__ == '__main__':

    # ---- Test 1 ----
    """
    transfo = Transformer(vocab_moves=570)

    dataset = import_data(filename="test.txt")
    boards, move_to_play, moves_mem = zip(*dataset)

    output = transfo.predict(boards[0:45], moves_mem[0:45])
    print("Le transformateur prédit : ", output)

    print('ok')
    """

    # Vérifier les shapes ... ==> En gros faut juste se démerder pour flatten ou un truc comme ça
    # Penser à supprimer les print dans le code

    # ---- Test 2 ----
    length_board = 64
    max_moves_in_game = 300
    vocab_moves = 64*(7*4 + 8)

    transfo = Transformer(vocab_moves=vocab_moves,
                          length_board=length_board, num_layers=4)

    dataset = import_data(filename="test.txt")

    data = dataset[:45]
    enc_input = [x[0] for x in dataset[:45]]
    dec_input = [x[2] for x in dataset[:45]]

    encoder_tokenize = ChessTokenizer()
    decoder_tokenize = ChessTokenizer()

    encoder_tokenize.fit_on_texts(enc_input)
    decoder_tokenize.fit_on_texts(dec_input)

    enc_input = encoder_tokenize.texts_to_sequences(
        enc_input, maxlen=length_board)
    dec_input = decoder_tokenize.texts_to_sequences(
        dec_input, maxlen=max_moves_in_game)

    output_decoder = transfo((enc_input, dec_input))

    print(output_decoder)

    # transfo.build(input_shape=[enc_input.shape, dec_input.shape])  # Chelou ...

    print("ok")
