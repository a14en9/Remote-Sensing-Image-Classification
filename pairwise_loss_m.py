
import tensorflow as tf
'''
def pairwise_loss_m(logits_1, logits_2, one_hot_labels, margin_1 = 0., margin_2 = 0, weights = 0.8):

    p_1 = tf.multiply(logits_1, one_hot_labels)
    p_1 = tf.reduce_sum(p_1, axis = 1)
    p_1 = -tf.log(p_1)

    p_2 = tf.multiply(logits_2, one_hot_labels)
    p_2 = tf.reduce_sum(p_2, axis = 1)
    p_2 = -tf.log(p_2)

    loss = tf.nn.relu(p_2-p_1-margin_1)
    loss = tf.multiply(tf.reduce_mean(loss, axis =0), weights)

    return loss
'''
def pairwise_loss_m(p1, p2, margin):
    return tf.maximum(0., p1-p2-margin)
