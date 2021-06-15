import numpy as np
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import time
from utils.read_tfrecord import *

slim = tf.contrib.slim
from net.vgg import vgg_16, vgg_arg_scope
from utils.SoSF import cov_pooling_operation

# ================ DATASET INFORMATION ======================
# State your log directory where you can retrieve your model
log_dir = '../../log/'

# Create a new evaluation log directory to visualize the validation process
log_eval = '../../log_eval/'

# State the dataset directory where the validation set is found
dataset_dir = '../../datasets/'

# State training or not
is_training = False
# State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 1
# State the number of epochs to evaluate
num_epochs = 1
# State the number of transformations to train
num_transformation = 12
# Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(log_dir)
# checkpoint_file = '../../exp/aid/log_05/model.ckpt-61288'
dropout_keep_prob = 1.0
# Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'

# State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 224
pad_img_size = 317  # 759 #317

def _transform(padded, num_of_transformations):
    #tiled = tf.tile(tf.expand_dims(padded, axis=4), [1, 1, 1, 1, num_of_transformations])
    rotated_tiles = []
    for ti in range(num_of_transformations):
        angle =360.0*ti/float(num_of_transformations)
        rotated_tiles.append(tf.contrib.image.rotate(padded, math.radians(angle), interpolation='BILINEAR'))
        # rotated_tiles.append(tf.image.rot90(padded))
        # tiled[:,:,:,:, ti] = rotate(tiled[:, :, :, :, ti],angle,axes=[1,2],reshape=False)
    print ('finished transforming')
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

        # First create the dataset and load one batch
        with tf.variable_scope('Data_input'):
            dataset = get_split('validation', dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='')
            images, raw_images, labels = load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size,
                                                    is_training=True)
            tf.summary.image('raw_img', raw_images, 1)
            tf.summary.image('image', images, 1)
        with tf.variable_scope('1st_granularity'):
            images_1 = tf.image.resize_bilinear(
                tf.image.resize_image_with_crop_or_pad(images, pad_img_size, pad_img_size), [image_size, image_size])
            tf.summary.image('pad_image_1', images_1, 1)
            transformed_images_1 = _transform(images_1, num_transformation)
            for i in range(0, num_transformation):
                tf.summary.image('rotate_image', transformed_images_1[i], 1)

        with tf.variable_scope('2nd_granularity'):
            images_2 = tf.image.resize_bilinear(tf.image.resize_image_with_crop_or_pad(tf.image.resize_bilinear(tf.image.central_crop(images, 0.9),
                                               [image_size, image_size]), pad_img_size,pad_img_size),[image_size,image_size])
            tf.summary.image('crop_image_2', images_2, 1)
            transformed_images_2 = _transform(images_2, num_transformation)
            for i in range(0,num_transformation):
                tf.summary.image('cropped_rotate_image_2',transformed_images_2[i], 1)
        with tf.variable_scope('3rd_granularity'):
            images_3 = tf.image.resize_bilinear(tf.image.resize_image_with_crop_or_pad(tf.image.resize_bilinear(tf.image.central_crop(images, 0.8),
                                                [image_size, image_size]), pad_img_size,pad_img_size),[image_size,image_size])
            tf.summary.image('crop_image_3', images_3, 1)
            transformed_images_3 = _transform(images_3, num_transformation)
            for i in range(0,num_transformation):
                tf.summary.image('cropped_rotate_image_3',transformed_images_3[i], 1)


        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed

        # create cnn feature inference based on vgg architecture
        with slim.arg_scope(vgg_arg_scope()):
            # with vgg_arg_scope(weight_decay=0.0005):
            net_1 = []
            for i in range(0, num_transformation):
                net_1_tmp, _ = vgg_16(transformed_images_1[i], num_classes=dataset.num_classes, is_training=is_training,
                                      spatial_squeeze=False,
                                      scope='vgg_16_1', fc_conv_padding='SAME', reuse=tf.AUTO_REUSE, endpoints='conv5')
                net_1.append(net_1_tmp)

        with slim.arg_scope(vgg_arg_scope()):
            # with vgg_arg_scope(weight_decay=0.0005):
            net_2 = []
            for i in range(0, num_transformation):
                net_2_tmp, _ = vgg_16(transformed_images_2[i], num_classes=dataset.num_classes, is_training=is_training,
                                      spatial_squeeze=False,
                                      scope='vgg_16_2', fc_conv_padding='SAME', reuse=tf.AUTO_REUSE, endpoints='conv5')
                net_2.append(net_2_tmp)

        with slim.arg_scope(vgg_arg_scope()):
            # with vgg_arg_scope(weight_decay=0.0005):
            net_3 = []
            for i in range(0, num_transformation):
                net_3_tmp, _ = vgg_16(transformed_images_3[i], num_classes=dataset.num_classes, is_training=is_training,
                                      spatial_squeeze=False,
                                      scope='vgg_16_3', fc_conv_padding='SAME', reuse=tf.AUTO_REUSE, endpoints='conv5')
                net_3.append(net_3_tmp)

        # create covariance pooling inference
        with tf.variable_scope('gaussian_cov_pooling'):
            cov_pooling_1 = []
            for i in range(0, num_transformation):
                cov_pooling_1_tmp = cov_pooling_operation(net_1[i], batch_size)
                cov_pooling_1.append(cov_pooling_1_tmp)

            cov_pooling_2 = []
            for i in range(0, num_transformation):
                cov_pooling_2_tmp = cov_pooling_operation(net_2[i], batch_size)
                cov_pooling_2.append(cov_pooling_2_tmp)

            cov_pooling_3=[]
            for i in range(0, num_transformation):
                cov_pooling_3_tmp = cov_pooling_operation(net_3[i], batch_size)
                cov_pooling_3.append(cov_pooling_3_tmp)

        # Create covariance regression inference
        with tf.variable_scope('cov_pooling_regression'):

            pool_stack_1 = tf.stack(cov_pooling_1, 3)
            ti_pooled_1_0 = tf.reduce_max(pool_stack_1, reduction_indices=[3])
            ti_pooled_1_0 = tf.nn.l2_normalize(ti_pooled_1_0,dim=[1,2])

            # index_0 = tf.argmax(pool_stack_1,axis=2)################################################
            
            pool_stack_2 = tf.stack(cov_pooling_2, 3)
            ti_pooled_2_0 = tf.reduce_max(pool_stack_2, reduction_indices=[3])
            ti_pooled_2_0 = tf.nn.l2_normalize(ti_pooled_2_0,dim=[1,2])

            pool_stack_3 = tf.stack(cov_pooling_3, 3)
            ti_pooled_3_0 = tf.reduce_max(pool_stack_3, reduction_indices=[3])
            ti_pooled_3_0 = tf.nn.l2_normalize(ti_pooled_3_0,dim=[1,2])

            ti_pooled = ti_pooled_1_0+ti_pooled_2_0+ti_pooled_3_0
            phi_I = tf.reshape(ti_pooled, [-1, ti_pooled.shape[2] * ti_pooled.shape[2]])
            phi_I = tf.divide(phi_I, tf.to_float(net_1_tmp.shape[1] * net_1_tmp.shape[2]))
            phi_I = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))
            z_l2 = tf.nn.l2_normalize(phi_I, dim=1)
            logits = slim.fully_connected(z_l2,dataset.num_classes, activation_fn=None)
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
            logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)', global_step_count, accuracy_value,
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
