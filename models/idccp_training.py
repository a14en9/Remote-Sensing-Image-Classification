import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import time
slim = tf.contrib.slim
from utils.read_tfrecord import *
from net.vgg import vgg_16, vgg_arg_scope
from utils.SoSF import _variable_with_orth_weight_decay, _cal_for_norm_cov
from utils.MPNCOV import *

# ================ DATASET INFORMATION ======================
# State dataset directory where the tfrecord files are located
dataset_dir = '../../datasets/optimal/train/'

# State where your log file is at. If it doesn't exist, create it.
log_dir = '../../exp/optimal/log/'

# State where your checkpoint file is
# checkpoint_file = '../../net/vgg_16.ckpt'
# checkpoint_file = '../../net/resnet_v1_50.ckpt'
checkpoint_file = tf.train.latest_checkpoint(log_dir)
# State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224

# Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'


# ================= TRAINING INFORMATION ==================
n_times = 8
# State training or not
is_training = True
# State the number of epochs to train
num_epochs = 30
# State your batch size
batch_size = 12
# Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.01
learning_rate_decay_factor = 0.9
num_epochs_before_decay = 10

def d4_transformation(img):
    e=img
    # tf.summary.image('e', e, 1)
    r3 = tf.image.rot90(img)
    # tf.summary.image('r3', r3, 1)
    r2= tf.image.rot90(r3)
    # tf.summary.image('r2', r2, 1)
    r = tf.image.rot90(r2)
    # tf.summary.image('r', r, 1)
    f=tf.image.flip_up_down(img)
    # tf.summary.image('f', f, 1)
    r2f=tf.image.flip_left_right(img)
    # tf.summary.image('r2f', r2f, 1)
    rf=tf.image.rot90(r2f)
    # tf.summary.image('rf', rf, 1)
    r3f=tf.image.rot90(f)
    # tf.summary.image('r3f', r3f, 1)
    rotated_tiles=[e,r,r2,r3,f,r2f,rf,r3f]
    return rotated_tiles

def my_cnn(input):
    with slim.arg_scope([slim.conv2d], reuse=tf.AUTO_REUSE):
        output = slim.conv2d(input, 512*2, [3, 3], activation_fn=tf.nn.relu, scope='my_cnn')
    return output
# ======================= TRAINING PROCESS =========================
def run():
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch

        with tf.variable_scope('Data_input'):
            dataset = get_split('train', dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='')
            images, raw_images, labels = load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size,
                                                    is_training=True)
            tf.summary.image('raw_img', raw_images, 1)
            tf.summary.image('image', images, 1)
            d4_img = d4_transformation(images)
        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        # Create cnn feature inference based on vgg architecture
        with slim.arg_scope(vgg_arg_scope()):
            # with vgg_arg_scope(weight_decay=0.0005):
            net = []
            for i in range(8):
                net_tmp, _ = vgg_16(d4_img[i], num_classes=dataset.num_classes, is_training=is_training,
                                    spatial_squeeze=False,
                                    scope='vgg_16', fc_conv_padding='SAME', reuse=tf.AUTO_REUSE, endpoints='conv5')
                net.append(net_tmp)
        '''
        with slim.arg_scope(resnet_arg_scope()):
            net = []
            for i in range(8):
                net_tmp, _ = resnet_v1_50(d4_img[i],num_classes=dataset.num_classes, is_training=is_training,
                                          reuse=tf.AUTO_REUSE,scope='resnet_v1_50',endpoints = 'block4')
                net.append(net_tmp)
        
        # Create projective_layer
        with tf.variable_scope('projective_layer'):
            proj_conv = []
            for i in range(8):
                proj_conv_tmp = slim.conv2d(net[i],512,[1,1],activation_fn=None,reuse=tf.AUTO_REUSE,scope='proj_conv')
                proj_conv.append(proj_conv_tmp)
        '''

        # Create covariance pooling inference
        with tf.variable_scope('rotate_invariant_cov_pooling'):
            cov_pooling = []
            for i in range(8):
                # cov_pooling_tmp = Covpool(tf.reshape(proj_conv[i], [batch_size, proj_conv_tmp.shape[1], proj_conv_tmp.shape[2], proj_conv_tmp.shape[3]]))
                cov_pooling_tmp = Covpool(tf.reshape(net[i], [batch_size, net_tmp.shape[1], net_tmp.shape[2], net_tmp.shape[3]]))
                cov_pooling.append(cov_pooling_tmp)
        # Create covariance regression inference
        with tf.variable_scope('cov_pooling_regression'):
            ti_pooled = tf.reduce_mean(tf.stack(cov_pooling, axis=3), reduction_indices=[3])
            ti_pooled = tf.nn.l2_normalize(ti_pooled, dim=[1, 2])
            shape = ti_pooled.get_shape().as_list()
            weight1, weight2 = _variable_with_orth_weight_decay('orth_weight0', shape, batch_size, n_times)
            ti_pooled = tf.matmul(tf.matmul(weight2, ti_pooled), weight1, name='matmulout')
            ti_pooled = _cal_for_norm_cov(ti_pooled)
            ti_pooled = Sqrtm(ti_pooled)
            phi_I = Triuvec(ti_pooled)
            # phi_I = tf.reshape(ti_pooled, [-1, ti_pooled.shape[1] * ti_pooled.shape[2]])
            phi_I = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
            phi_I = tf.nn.l2_normalize(phi_I, dim=1)
            logits = slim.fully_connected(phi_I, dataset.num_classes, activation_fn=None)
        # Define the scopes that you want to exclude for restoration
        exclude = ['rotate_invariant_cov_pooling', 'cov_pooling_regression','projective_layer']
        variables_to_restore = slim.get_variables_to_restore(exclude=None)

        with tf.name_scope('softmax'):
            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_input))
            # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
            # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
            loss = slim.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # obtain the regularization losses as well
            total_loss = slim.losses.get_total_loss(add_regularization_losses=True)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=learning_rate_decay_factor,
            staircase=True)
        # Now we can define the optimizer that takes on the learning rate and create the train_op
        var_list_feat = tf.trainable_variables()[:-2]
        var_list_regression = tf.trainable_variables()[-2:]
        #
        opt_feat = tf.train.GradientDescentOptimizer(lr)
        opt_regression = tf.train.MomentumOptimizer(lr,momentum=0.9)  # tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=0.1)
        '''
        gradients_feat = opt_feat.compute_gradients(total_loss, var_list_feat)
        gradients_regression = opt_regression.compute_gradients(total_loss, var_list_regression)

        train_op_feat = opt_feat.apply_gradients(gradients_feat, global_step=global_step)
        train_op_regression = opt_regression.apply_gradients(gradients_regression, global_step=global_step)
        train_op = tf.group(train_op_regression)
        # train_op = tf.group(train_op_feat, train_op_regression)
        '''
        train_op_feat = slim.learning.create_train_op(total_loss,opt_feat,variables_to_train=var_list_feat)
        train_op_regression = slim.learning.create_train_op(total_loss,opt_regression,variables_to_train=var_list_regression)
        # train_op = tf.group(train_op_regression)
        train_op = tf.group(train_op_feat, train_op_regression)
        '''
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            total_loss = control_flow_ops.with_dependencies([updates],total_loss)
        '''
        #
        # State the metrics that you wa[nt to predict. We get a predictions that is not one_hot_encoded.
        with tf.variable_scope('Accruacy_Compute'):
            # predictions = tf.argmax(end_points['Predictions'], 1)
            # probabilities = end_points['Predictions']
            probabilities = slim.softmax(logits, scope='Predictions')
            predictions = tf.argmax(probabilities, 1)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
            metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.scalar('losses/loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()



        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_step(sess, train_op, global_step, total_loss=total_loss):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            start_time = time.time()
            _, total_loss, global_step_count, _ = sess.run([train_op, total_loss, global_step, metrics_op])
            time_elapsed = time.time() - start_time

            # Run the logging to print some results
            logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

            return total_loss, global_step_count

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Define your supervisor for ruadd_image_summariesnning a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=log_dir, summary_op=None, init_fn=restore_fn)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Run the managed session
        with sv.managed_session(config=config) as sess:
            for step in range(0, num_steps_per_epoch * num_epochs):
                # At the start of every epoch, show the vital information:

                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                    logging.info('Current Learning Rate: %s', learning_rate_value)
                    logging.info('Current Streaming Accuracy: %s', accuracy_value)
                    # optionally, print your logits and predictions for a sanity check that things are going fine.
                    logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        [logits, probabilities, predictions, labels])
                    # print('logits: \n', logits_value)
                    # print('Probabilities: \n', probabilities_value)
                    print('predictions: \n', predictions_value)
                    print('Labels:\n', labels_value)

                # Log the summaries every 10 step.
                if step % 10 == 0:
                    total_loss, _ = train_step(sess, train_op, sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)

                # If not, simply run the training step
                else:
                    total_loss, _ = train_step(sess, train_op, sv.global_step)

            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', total_loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


if __name__ == '__main__':
    run()
