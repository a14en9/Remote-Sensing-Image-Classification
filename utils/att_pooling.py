import tensorflow as tf
import tensorflow.contrib.slim as slim
def full_rank_bilinear_pooling(conv, num_classes, dropout, is_training = None, reuse= None, scope='full_rank_bilinear_pooling'):
    with tf.variable_scope (scope, reuse=reuse):
        phi_I = tf.einsum('ijkm,ijkn->imn', conv, conv)
        phi_I = tf.reshape(phi_I, [-1, 512 * 512])
        phi_I = tf.divide(phi_I, 784.0)
        y_ssqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
        z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
        # z_l2 = slim.dropout(z_l2,dropout,is_training=is_training)
        # logits = slim.fully_connected(z_l2,num_classes, activation_fn=None, weights_initializer=tf.zeros_initializer(),
        #                            biases_initializer=tf.zeros_initializer(), trainable=is_training)
        logits = slim.fully_connected(z_l2, num_classes, activation_fn=None, trainable=is_training)
        return logits
def low_rank_bilinear_pooling(conv, num_classes, dropout, is_training = None, reuse= None, scope='low_rank_bilinear_pooling'):
    with tf.variable_scope (scope, reuse=reuse):
        conv_a = slim.conv2d(conv, num_classes, [1,1], scope='project_a', weights_initializer=tf.contrib.layers.xavier_initializer(0.001),
                    biases_initializer=tf.zeros_initializer(),activation_fn=None, normalizer_fn=None)

        conv_b = slim.conv2d(conv, num_classes, [1, 1], scope='project_b',weights_initializer=tf.contrib.layers.xavier_initializer(0.001),
                             biases_initializer=tf.zeros_initializer(), activation_fn=None, normalizer_fn=None)
        phi_I = tf.einsum('ijkm,ijkn->imn', conv_a, conv_b)
        phi_I = tf.reshape(phi_I, [-1, num_classes * num_classes])
        phi_I = tf.divide(phi_I, 784.0)
        y_ssqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
        z_l2 = tf.nn.l2_normalize(y_ssqrt, dim=1)
        z_l2 = slim.dropout(z_l2, dropout, is_training=is_training)
        logits = slim.fully_connected(z_l2, num_classes, activation_fn=None)
        return logits



