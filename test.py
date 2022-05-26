import tensorflow as tf

e = tf.keras.layers.Embedding(5, 3)
e.build(5)

print(e.trainable_variables)