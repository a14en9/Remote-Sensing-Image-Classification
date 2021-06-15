from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from net.vgg import vgg_16, vgg_arg_scope
import time
from utils.read_tfrecord import *

slim = tf.contrib.slim
from utils.SoSF import cov_pooling_operation

# ================ DATASET INFORMATION ======================

# State dataset directory where the tfrecord files are located
dataset_dir = '../../datasets/'

# State where your log file is at. If it doesn't exist, create it.
log_dir = '../../log/'

# comment the line below when fine-tuning
checkpoint_file = '../../net/vgg_16_3s.ckpt'
# comment the line below during training
# checkpoint_file = tf.train.latest_checkpoint(log_dir)

image_size = 224
pad_img_size = 317  # 759 #317
# Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = '_%s_*.tfrecord'

# ================= TRAINING INFORMATION ==================
# State training or not
is_training = False
# State the number of epochs to train
num_epochs = 100
# State your batch size
batch_size = 8
# State number of transformations
num_transformation = 12
# Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.1  # 0.0001
learning_rate_decay_factor = 0.9
num_epochs_before_decay = 10

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
            # tf.summary.image('pad_image_2', images_2, 1)
            transformed_images_2 = _transform(images_2, num_transformation)
            # transformed_images_2 = crop_image_2
            for i in range(0,num_transformation):
                tf.summary.image('cropped_rotate_image_2',transformed_images_2[i], 1)
        with tf.variable_scope('3rd_granularity'):
            images_3 = tf.image.resize_bilinear(tf.image.resize_image_with_crop_or_pad(tf.image.resize_bilinear(tf.image.central_crop(images, 0.8),
                                                [image_size, image_size]), pad_img_size,pad_img_size),[image_size,image_size])
            tf.summary.image('crop_image_3', images_3, 1)
            # images_3 = tf.image.resize_bilinear(_tf_pad(tf.image.resize_bilinear(crop_image_3,[image_size, image_size]),
            #                                             raw_image_size,raw_image_size,batch_size),[224,224])
            # tf.summary.image('pad_image_3', images_3, 1)
            transformed_images_3 = _transform(images_3, num_transformation)
            for i in range(0,num_transformation):
                tf.summary.image('cropped_rotate_image_3',transformed_images_3[i], 1)

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

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
        # Define the scopes that you want to exclude for restoration
        exclude = ['gaussian_cov_pooling', 'cov_pooling_regression']
        # exclude = ['compression_layer','dense_layers']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

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
        opt_regression = tf.train.MomentumOptimizer(lr, momentum=0.9)  # tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=0.1)

        gradients_feat = opt_feat.compute_gradients(total_loss, var_list_feat)
        gradients_regression = opt_regression.compute_gradients(total_loss, var_list_regression)

        train_opt_feat = opt_feat.apply_gradients(gradients_feat, global_step=global_step)
        train_opt_regression = opt_regression.apply_gradients(gradients_regression, global_step=global_step)
        train_op = tf.group(train_opt_regression)
        # train_op = tf.group(train_opt_feat, train_opt_regression)

        ## State the metrics that you wa[nt to predict. We get a predictions that is not one_hot_encoded.
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
                    print('Labels:\n:', labels_value)

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
