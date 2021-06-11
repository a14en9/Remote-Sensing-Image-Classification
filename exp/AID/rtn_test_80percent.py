import sys
sys.path.append('../../')
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from net.vgg import vgg_16, vgg_arg_scope
import time
slim = tf.contrib.slim
from scripts.RTNet import *
from scripts.SoSF import *
from scripts.read_tfrecord import *
# ================ DATASET INFORMATION ======================
# State your log directory where you can retrieve your model
log_dir = '../../exp/AID/80percent/log/'

# Create a new evaluation log directory to visualize the validation process
log_eval = '../../exp/AID/80percent/log_eval/'

# State the dataset directory where the validation set is found
dataset_dir = '../../datasets/AID_80percent/test/'

# State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 1
# State the number of epochs to evaluate
num_epochs = 1
# Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)
dropout_keep_prob = 1.0
# Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'
# State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224
# State training or not
is_training_classification = False
# State training or not
is_training = False

def run():
    # Create log_dir for evaluation information

    if not os.path.exists(log_eval):
        os.mkdir(log_eval)
    predict_file = open(log_eval + '/predictions.txt', 'w+')
    label_file = open(log_eval + '/labels.txt', 'w+')
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        stn_init1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]).astype('float32')
        stn_init2 = np.array([0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0]).astype('float32')
        # stn_init2 = np.array([0.6, 0.0, +0.25, 0.0, 0.6, +0.25, 0.0, 0.0]).astype('float32')
        stn_init3 = np.array([0.8, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0]).astype('float32')
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        with tf.variable_scope('Data_input'):
            dataset = get_split('validation', dataset_dir, file_pattern, file_pattern_for_counting='')
            images, raw_images, labels = load_batch(dataset, batch_size=batch_size, height=image_size,width=image_size,is_training=False)
            tf.summary.image('raw_img', raw_images, 1)
            tf.summary.image('image', images, 1)

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        # get feature for localization
        with tf.variable_scope('STNs'):
            with slim.arg_scope(STNet_arg_scope(weight_decay=0.0005)):
                stned_images_1 = STNet(images, batch_size, image_size, num_class=len(stn_init1),
                                       dropout_keep_prob=dropout_keep_prob, stn_init=stn_init1,
                                       scope='STN_1', is_training=is_training, reuse=None)
                # stned_images_1 = normalise(stned_images_1)

                stned_images_2 = STNet(images, batch_size, image_size, num_class=len(stn_init2),
                                       dropout_keep_prob=dropout_keep_prob, stn_init=stn_init2,
                                       scope='STN_2', is_training=is_training, reuse=False)
                # stned_images_2 = normalise(stned_images_2)

                stned_images_3 = STNet(images, batch_size, image_size, num_class=len(stn_init2),
                                       dropout_keep_prob=dropout_keep_prob, stn_init=stn_init3,
                                       scope='STN_3', is_training=is_training, reuse=False)
                # stned_images_2 = normalise(stned_images_2)
                # input_images = normalise(images)
                # tf.summary.image('input_img', images, 1)
                tf.summary.image('stn_img_1', stned_images_1, 1)
                tf.summary.image('stn_img_2', stned_images_2, 1)
                tf.summary.image('stn_img_3', stned_images_3, 1)
        with slim.arg_scope(vgg_arg_scope()):
            net_1, _ = vgg_16(stned_images_1, num_classes=dataset.num_classes, is_training=is_training,
                              spatial_squeeze=False,
                              scope='vgg_16_1', fc_conv_padding='SAME', reuse=None, endpoints='conv5')
            net_2, _ = vgg_16(stned_images_2, num_classes=dataset.num_classes, is_training=is_training,
                              spatial_squeeze=False,
                              scope='vgg_16_2', fc_conv_padding='SAME', reuse=False, endpoints='conv5')
            net_3, _ = vgg_16(stned_images_3, num_classes=dataset.num_classes, is_training=is_training,
                              spatial_squeeze=False,
                              scope='vgg_16_3', fc_conv_padding='SAME', reuse=False, endpoints='conv5')
        with tf.variable_scope('classification_layer'):
            logits_1 = full_rank_bilinear_pooling(net_1, dataset.num_classes, dropout_keep_prob,
                                                  is_training=is_training_classification, reuse=None,
                                                  scope='full_rank_bilinear_pooling_1')
            p1 = slim.softmax(logits_1, scope='Predictions_1')
            logits_2 = full_rank_bilinear_pooling(net_2, dataset.num_classes, dropout_keep_prob,
                                                  is_training=is_training_classification, reuse=None,
                                                  scope='full_rank_bilinear_pooling_2')
            p2 = slim.softmax(logits_2, scope='Predictions_2')
            logits_3 = full_rank_bilinear_pooling(net_3, dataset.num_classes, dropout_keep_prob,
                                                  is_training=is_training_classification, reuse=False,
                                                  scope='full_rank_bilinear_pooling_3')
            p3 = slim.softmax(logits_3, scope='Predictions_3')
            logits = logits_1 + logits_2 + logits_3
        # #get all the variables to restore from theaccuracy checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)
            # Just define the metrics to track without the loss or whatsoever
        with tf.variable_scope('Accruacy_Compute'):
            # probabilities = slim.softmax(logits, scope='Predictions')
            probabilities = p1 + p2 + p3
            predictions = tf.argmax(probabilities, 1)
            accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)

            # Compute a per-batch confusion
            batch_confusion = tf.confusion_matrix(labels, predictions,
                                                  num_classes=dataset.num_classes,
                                                  name='batch_confusion')
            # Create an accumulator variable to hold the counts
            confusion = tf.Variable(tf.zeros([dataset.num_classes, dataset.num_classes],
                                             dtype=tf.int32),
                                    name='confusion')
            # Create the update op for doing a "+=" accumulation on the batch
            confusion_update = confusion.assign(confusion + batch_confusion)

            # Cast counts to float so tf.summary.image renormalizes to [0,255]
            confusion_image = tf.reshape(tf.cast(confusion, tf.float32),
                                         [1, dataset.num_classes, dataset.num_classes, 1])
            # Combine streaming accuracy and confusion matrix updates in one op
            metrics_op = tf.group(accuracy_update, confusion_update, probabilities)

            tf.summary.image('confusion', 1 - confusion_image)
            tf.summary.scalar('accuracy', accuracy)
            # metrics_op = tf.group(accuracy_update, probabilities)

        # Create the global step and an increment op for monitoring
        global_step = slim.get_or_create_global_step()
        global_step_op = tf.assign(global_step,
                                   global_step + 1)  # no apply_gradient method so manually increasing the global_step

        # Create a evaluation step function
        def eval_step(sess, metrics_op, global_step):
            '''
            Simply takes in a session, runs the metrics op and some logging information.
            '''
            start_time = time.time()
            _, global_step_count, accuracy_value, predictions_value, labels_value = sess.run(
                [metrics_op, global_step_op, accuracy, predictions, labels])
            time_elapsed = time.time() - start_time

            # Log some information
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count,
                         accuracy_value,
                         time_elapsed)
            # print (predict_file, 'predictions: \n', predictions_value)
            # print (label_file, 'labels: \n', labels_value)
            predict_file.write('%d \n' % (np.int(predictions_value)))
            label_file.write('%d \n' % (np.int(labels_value)))
            return accuracy_value

        # Define some scalar quantities to monitor
        tf.summary.scalar('Validation_Accuracy', accuracy)
        my_summary_op = tf.summary.merge_all()

        # Get your supervisor
        sv = tf.train.Supervisor(logdir=log_eval, summary_op=None, saver=None, init_fn=restore_fn)

        # Now we are ready to run in one session
        with sv.managed_session() as sess:
            for step in range(num_steps_per_epoch * num_epochs):
                sess.run(sv.global_step)
                # print vital information every start of the epoch as always
                if step % num_batches_per_epoch == 0:
                    logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, num_epochs)
                    logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy))

                # Compute summaries every 10 steps and continue evaluating
                if step % 10 == 0:
                    eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)
                    summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries)


                # Otherwise just run as per normal
                else:
                    eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)

            # At the end of all the evaluation, show the final accuracy
            logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy))

            logging.info(
                'Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
            # predict_file.close
            # label_file.close

if __name__ == '__main__':
    run()
