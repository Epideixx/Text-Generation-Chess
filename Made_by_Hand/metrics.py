# ---------------------------------------------
#           Metrics for Transformater
# ---------------------------------------------

import tensorflow as tf


class MaskedSparseCategoricalEntropy(tf.keras.losses.Loss):
    
    def __init__(self, from_logits: bool = False):
        super(MaskedSparseCategoricalEntropy, self).__init__()
        self.from_logits = from_logits
        self.crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight = None):
        """
        Parameters
        ----------
        sample_weight : float in [0.0, 1.0]
            Weight of the last move, the one which is predicted
        """

        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        mask = tf.cast(mask, dtype=tf.float32)

        if sample_weight : 
            sum_mask = tf.reduce_sum(mask, axis = -1)
            sample_weight_in_progress = tf.multiply( tf.transpose(tf.concat([tf.expand_dims(tf.fill(mask.shape[1], (1 - sample_weight)/sum_mask[i]), axis = -1) for i in range(mask.shape[0])], axis = -1)), mask)
            # Add initial sample_weight to the last non-zero value
            # For the moment I give up ...
        loss = tf.cast(self.crossentropy(y_true, y_pred), tf.float32)
        loss *= mask

        # From Valentin
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)
        return loss


class ClassicAccuracy(object):
    """ Computes the standard accuracy masked by the labels
    equal to 0.
    """

    def __init__(self):
        pass

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Returns the accuracy value averaged over batch
        and tokens.
        Parameters
        ----------
        y_true : tf.Tensor, shape=(..., seq_len), dtype=int64
            Labels, masked value are store as 0.
        y_pred : float tf.Tensor, shape=(..., seq_len, vocab_size)
            Prediction (probability or logits) for each token.
        Returns
        -------
        acc : tf.Tensor, shape=(), dtype=float32
            Result of the accuracy masked by the labels
            equal to 0.
        """
        good_preds = tf.equal(y_true, tf.cast(
            tf.argmax(y_pred, axis=-1), dtype=tf.int32))
        good_preds = tf.cast(good_preds, tf.float32)
        acc = tf.reduce_sum(good_preds) / good_preds.shape[0]
        return acc


# ------- Totally from VALENTIN --------

class MaskedAccuracy(object):
    """ Computes the standard accuracy masked by the labels
    equal to 0.
    """

    def __init__(self):
        pass

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Returns the accuracy value averaged over batch
        and tokens.
        Parameters
        ----------
        y_true : tf.Tensor, shape=(..., seq_len), dtype=int64
            Labels, masked value are store as 0.
        y_pred : float tf.Tensor, shape=(..., seq_len, vocab_size)
            Prediction (probability or logits) for each token.
        Returns
        -------
        acc : tf.Tensor, shape=(), dtype=float32
            Result of the accuracy masked by the labels
            equal to 0.
        """
        mask = tf.not_equal(y_true, 0)
        good_preds = tf.equal(y_true, tf.cast(
            tf.argmax(y_pred, axis=-1), dtype=tf.int32))
        good_preds = tf.cast(tf.logical_and(mask, good_preds), tf.float32)
        mask = tf.cast(mask, tf.float32)
        acc = tf.reduce_sum(good_preds) / (tf.reduce_sum(mask) + 1e-8)
        return acc



# Optimizer scheduler (from tensorflow)
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)