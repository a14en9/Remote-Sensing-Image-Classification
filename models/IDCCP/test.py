import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import time
slim = tf.contrib.slim
from utils.read_tfrecord import *
from net.vgg import vgg_16, vgg_arg_scope
from utils.SoSF import _variable_with_orth_weight_decay, _cal_for_norm_cov
from net.resnet_v1 import resnet_v1,resnet_v1_50,resnet_arg_scope
from utils.MPNCOV import *

# ================ DATASET INFORMATION ======================
# State your log directory where you can retrieve your model
log_dir = '../../log/'

# Create a new evaluation log directory to visualize the validation process
log_eval = '../../log_eval/'

# State the dataset directory where the validation set is found
dataset_dir = '../../datasets/'

n_times = 8
# State training or not
is_training = False
# State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 1
# State the number of epochs to evaluate
num_epochs = 1
# Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)
# checkpoint_file = '../../exp/nwpu_resisc45/01/log/model.ckpt-22055'
dropout_keep_prob = 1.0
# Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'

# State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224

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

def run():
    # Create log_dir for evaluation information

    if not os.path.exists(log_eval):
        os.mkdir(log_eval)
    predict_file = open(log_eval + '/predictions.txt', 'w+')
    label_file = open(log_eval + '/labels.txt', 'w+')
    # Just construct the graph from scratch again
    with tf.Graph().as_default() as graph:

        tf.logging.set_verbosity(tf.logging.INFO)
        # Get the dataset first and load one batch of validation images and labels tensors. Set is_training as False so as to use the evaluation preprocessing
        with tf.variable_scope('Data_input'):
            dataset = get_split('validation', dataset_dir, file_pattern, file_pattern_for_counting='')
            images, raw_images, labels = load_batch(dataset, batch_size=batch_size, height=image_size,width=image_size,is_training=False)
            # tf.summary.image('raw_img', raw_images, 1)
            tf.summary.image('image', images, 1)
            d4_img = d4_transformation(images)
        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        '''
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
        

        # Create covariance pooling inference
        with tf.variable_scope('rotate_invariant_cov_pooling'):
            cov_pooling = []
            for i in range(8):
                cov_pooling_tmp = Covpool(tf.reshape(proj_conv[i], [batch_size, proj_conv_tmp.shape[1], proj_conv_tmp.shape[2], proj_conv_tmp.shape[3]]))
#                 cov_pooling_tmp = Covpool(tf.reshape(net[i], [batch_size, net_tmp.shape[1], net_tmp.shape[2], net_tmp.shape[3]]))
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
            # logits = arcface_softmax(phi_I, labels, dataset.num_classes, s=s, m=m)
            logits = slim.fully_connected(phi_I, dataset.num_classes, activation_fn=None)
        # #get all the variables to restore from theaccuracy checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Just define the metrics to track without the loss or whatsoever
        with tf.variable_scope('Accruacy_Compute'):
            probabilities = slim.softmax(logits, scope='Predictions')
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

            tf.summary.image('confusion', 1-confusion_image)
            tf.summary.scalar('accuracy', accuracy)
            # metrics_op = tf.group(accuracy_update, probabilities)

        # Create the global step and an increment op for monitoring
        global_step = slim.get_or_create_global_step()
        global_step_op = tf.assign(global_step,global_step + 1)  # no apply_gradient method so manually increasing the global_step

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
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value,
                         time_elapsed)
            #print (predict_file, 'predictions: \n', predictions_value)
            #print (label_file, 'labels: \n', labels_value)
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

