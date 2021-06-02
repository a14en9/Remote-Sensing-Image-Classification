import sys
sys.path.append('../../')
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from net.vgg import vgg_16, vgg_arg_scope
import time
slim = tf.contrib.slim
from scripts.RTNet import *
from scripts.att_pooling import *
from scripts.pairwise_loss_m import *
from scripts.read_tfrecord import *
# ================ DATASET INFORMATION ======================

# State dataset directory where the tfrecord files are located
dataset_dir = '../../datasets/AID/train/'

# State where your log file is at. If it doesn't exist, create it.
log_dir = '../../exp/AID/train_20percent/log/'

# comment the line below when fine-tuning
# checkpoint_file = '../../net/vgg_16_3s.ckpt'
# comment the line below during training
checkpoint_file = tf.train.latest_checkpoint(log_dir)
# State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224
# Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'


# ================= TRAINING INFORMATION ==================
# State the number of epochs to train
num_epochs = 100
# State your batch size
batch_size = 24
# Learning rate information and configuration (Up to you to experiment)
num_epochs_before_decay = 10
dropout_keep_prob = 0.5

# State training or not
is_training_classification = True
# State training or not, change it to False when fine-tuning
is_training = True
# Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.01
learning_rate_decay_factor = 0.9

# ======================= TRAINING PROCESS =========================
def run():
    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        stn_init1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).astype('float32')
        stn_init2 = np.array([0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]).astype('float32')
        # stn_init2 = np.array([0.6, 0.0, +0.25, 0.0, 0.6, +0.25, 0.0, 0.0]).astype('float32')
        stn_init3 = np.array([0.8, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0]).astype('float32')
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        with tf.variable_scope('Data_input'):
            dataset = get_split('train', dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='')
            images, raw_images, labels = load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size,
                                                    is_training=True)
            tf.summary.image('raw_img', raw_images, 1)
            tf.summary.image('image', images, 1)
        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
        # get feature for localization
        with tf.variable_scope('STNs'):
            with slim.arg_scope(STNet_arg_scope(weight_decay=0.0005)):

                stned_images_1 = STNet(images, batch_size, image_size, num_class=len(stn_init1), dropout_keep_prob=dropout_keep_prob, stn_init=stn_init1,
                                           scope='STN_1',is_training=is_training, reuse=None)

                stned_images_2 = STNet(images, batch_size, image_size, num_class=len(stn_init2), dropout_keep_prob=dropout_keep_prob, stn_init=stn_init2,
                                       scope='STN_2',is_training=is_training, reuse=False)

                stned_images_3 = STNet(images, batch_size, image_size, num_class=len(stn_init2),dropout_keep_prob=dropout_keep_prob, stn_init=stn_init3,
                                       scope='STN_3', is_training=is_training, reuse=False)
                # input_images = normalise(images)
                # tf.summary.image('input_img', images, 1)
                tf.summary.image('stn_img_1', stned_images_1, 1)
                tf.summary.image('stn_img_2', stned_images_2, 1)
                tf.summary.image('stn_img_3', stned_images_3, 1)
        with slim.arg_scope(vgg_arg_scope()):
            net_1, _ = vgg_16(stned_images_1, num_classes=dataset.num_classes, is_training=is_training, spatial_squeeze=False,
                              scope='vgg_16_1',fc_conv_padding='SAME', reuse= None, endpoints='conv5')
            net_2, _ = vgg_16(stned_images_2, num_classes=dataset.num_classes, is_training=is_training, spatial_squeeze=False,
                              scope='vgg_16_2',fc_conv_padding='SAME', reuse=False, endpoints='conv5')
            net_3, _ = vgg_16(stned_images_3, num_classes=dataset.num_classes, is_training=is_training, spatial_squeeze=False,
                              scope='vgg_16_3',fc_conv_padding='SAME', reuse= False, endpoints='conv5')
        with tf.variable_scope('classification_layer'):
            logits_1 = full_rank_bilinear_pooling(net_1, dataset.num_classes, dropout_keep_prob, is_training= is_training_classification, reuse=None, scope='full_rank_bilinear_pooling_1')
            p1 = slim.softmax(logits_1, scope='Predictions_1')
            logits_2 = full_rank_bilinear_pooling(net_2, dataset.num_classes, dropout_keep_prob, is_training= is_training_classification,reuse=False, scope='full_rank_bilinear_pooling_2')
            p2 = slim.softmax(logits_2, scope='Predictions_2')
            logits_3 = full_rank_bilinear_pooling(net_3, dataset.num_classes, dropout_keep_prob, is_training= is_training_classification,reuse=False, scope='full_rank_bilinear_pooling_3')
            p3 = slim.softmax(logits_3, scope='Predictions_3')
            logits = logits_1 + logits_2 + logits_3
        # Define the scopes that you want to exclude for restoration
        exclude = ['STNs','classification_layer']
        # set it to 'exclude=exclude' during training and set it 'exclude=None' when fine-tuning
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        with tf.variable_scope('Loss_Compute'):
            # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
            # scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)

            '''
            total_loss = slim.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
            '''
            # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks

            loss_intra_1 = slim.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits_1)
            loss_intra_2 = slim.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits_2)
            loss_intra_3 = slim.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits_3)
            loss_inter_1_2 = pairwise_loss(p1, p2, one_hot_labels, margin=0.05)
            loss_inter_1_3 = pairwise_loss(p1, p3, one_hot_labels, margin=0.05)
            slim.losses.add_loss(loss_inter_1_2)
            slim.losses.add_loss(loss_inter_1_3)
            total_loss = slim.losses.get_total_loss()  # obtain the regularization losses as well

            tf.summary.scalar('losses/loss_intra_1', loss_intra_1)
            tf.summary.scalar('losses/loss_intra_2', loss_intra_2)
            tf.summary.scalar('losses/loss_intra_3', loss_intra_3)
            tf.summary.scalar('losses/loss_inter_1_2', loss_inter_1_2)
            tf.summary.scalar('losses/loss_inter_1_3', loss_inter_1_3)
            tf.summary.scalar('losses/Loss', total_loss)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=0.9,
            staircase=True)
        # Now we can define the optimizer that takes on the learning rate and create the train_op
        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.

        var_list_stn_vgg = tf.trainable_variables()[:-6]
        var_list_cl = tf.trainable_variables()[-6:]
        #
        opt_stn_vgg = tf.train.GradientDescentOptimizer(lr)
        opt_cl = tf.train.MomentumOptimizer(lr, momentum=0.9)

        train_opt_stn_vgg = slim.learning.create_train_op(total_loss, opt_stn_vgg, variables_to_train=var_list_stn_vgg)
        train_opt_cl = slim.learning.create_train_op(total_loss, opt_cl, variables_to_train=var_list_cl)

        # comment the line below during fine-tuning
        train_op = tf.group(train_opt_cl)
        # comment the line below during training
        # train_op = tf.group(train_opt_stn_vgg, train_opt_cl)
        #
        # State the metrics that you wa[nt to predict. We get a predictions that is not one_hot_encoded.
        with tf.variable_scope('Accruacy_Compute'):
            probabilities = (p1 + p2 + p3)
            predictions = tf.argmax(probabilities, 1)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)

            tf.summary.scalar('accuracy', accuracy)

            predictions_1 = tf.argmax(p1, 1)
            predictions_2 = tf.argmax(p2, 1)
            predictions_3 = tf.argmax(p3, 1)
            accuracy_1, accuracy_update_1 = tf.contrib.metrics.streaming_accuracy(predictions_1, labels)
            accuracy_2, accuracy_update_2 = tf.contrib.metrics.streaming_accuracy(predictions_2, labels)
            accuracy_3, accuracy_update_3 = tf.contrib.metrics.streaming_accuracy(predictions_3, labels)
            tf.summary.scalar('accuracy_1', accuracy_1)
            tf.summary.scalar('accuracy_2', accuracy_2)
            tf.summary.scalar('accuracy_3', accuracy_3)
            metrics_op = tf.group(accuracy_update, probabilities, accuracy_update_1,p1, accuracy_update_2, p2, accuracy_update_3, p3)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('learning_rate', initial_learning_rate)
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
