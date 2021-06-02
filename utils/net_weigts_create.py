import sys
sys.path.append('../../')
import tensorflow as tf
from scripts.read_tfrecord import *
from net.vgg import  vgg_arg_scope, vgg_16


restore_file = '../../net/vgg_16.ckpt'
save_file = '../../net/vgg_16_3s.ckpt'
dataset_dir = '../../datasets/AID_20percent/train/'
image_size = 448
batch_size = 32
file_pattern = '_%s_*.tfrecord'


def run():
    dataset = get_split('train', dataset_dir, file_pattern=file_pattern, file_pattern_for_counting='')
    images, raw_images, labels = load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size,
                                            is_training=True)
    with slim.arg_scope(vgg_arg_scope()):
        net, _ = vgg_16(images, num_classes=dataset.num_classes, is_training=False, spatial_squeeze=False,
                               scope='vgg_16', fc_conv_padding='SAME', reuse=None, endpoints='conv5')
    with slim.arg_scope(vgg_arg_scope()):
        net_1, _ = vgg_16(images, num_classes=dataset.num_classes, is_training=False, spatial_squeeze=False,
                               scope='vgg_16_1', fc_conv_padding='SAME', reuse=None, endpoints='conv5')
    with slim.arg_scope(vgg_arg_scope()):
        net_2, _ = vgg_16(images, num_classes=dataset.num_classes, is_training=False, spatial_squeeze=False,
                               scope='vgg_16_2', fc_conv_padding='SAME', reuse=None, endpoints='conv5')
    with slim.arg_scope(vgg_arg_scope()):
        net_3, _ = vgg_16(images, num_classes=dataset.num_classes, is_training=False, spatial_squeeze=False,
                        scope='vgg_16_3', fc_conv_padding='SAME', reuse=None, endpoints='conv5')


    variables = slim.get_variables_to_restore()
    l = int((len(variables)) / 4)
    variables_to_restore = variables[:l]
    variables_to_save2 = variables[l:2 * l]
    variables_to_save3 = variables[2 * l:3 * l]
    variables_to_save4 = variables[3 * l:4 * l]
    restore = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver(variables_to_save2 + variables_to_save3 + variables_to_save4)

    with tf.Session() as sess:
        restore.restore(sess, restore_file)
        for i in range(len(variables_to_restore)):
            assign_op = tf.assign(variables_to_save2[i], variables_to_restore[i])
            sess.run(assign_op)
            assign_op = tf.assign(variables_to_save3[i], variables_to_restore[i])
            sess.run(assign_op)
            assign_op = tf.assign(variables_to_save4[i], variables_to_restore[i])
            sess.run(assign_op)

        saver.save(sess, save_file)


if __name__ == '__main__':
    run()