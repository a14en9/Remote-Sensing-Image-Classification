import tensorflow as tf
import sys
import net.vgg_preprocessing as vgg_preprocessing
import os
import tensorflow.contrib.slim as slim

#============== DATASET LOADING ======================
#We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
def get_split(split_name, dataset_dir, file_pattern, file_pattern_for_counting='flowers'):
    '''
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
    set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
    Your file_pattern is very important in locating the files later. 

    INPUTS:
    - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
    - dataset_dir(str): the dataset directory where the tfrecord files are located
    - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data
    - file_pattern_for_counting(str): the string name to identify your tfrecord files for counting

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
    '''

#State the labels file and read it

    with open(dataset_dir+'labels.txt') as labels_file:  
        labels = labels_file.readlines() 

#Create a dictionary to refer each label to their string name
    labels_to_name = {}
    for line in labels:
        label, string_name = line.split(':')
        string_name = string_name[:-1] #Remove newline
        labels_to_name[int(label)] = string_name

    #State the number of classes to predict:
    num_classes = len(labels)

    #Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
    items_to_descriptions = {
        'image': 'A 3-channel RGB coloured image.',
        'label': 'A label that is as such -- 0:, 1:, 2:, 3:, 4:, ...'
    }

    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary fofile_pattern_pathr the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset


def load_batch(dataset, batch_size, height, width, is_training=True):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    #Obtain the raw image using the get methodimage_size
    raw_image, label = data_provider.get(['image', 'label'])

    raw_image = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(raw_image, 0), [256, 256]))
    # Perform the correct preprocessing for this image depending if it is training or evaluating
    # image = inception_preprocessing.preprocess_image(raw_image,height, width, is_training)
    image = vgg_preprocessing.preprocess_image(raw_image, height, width, is_training)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_bilinear(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)
    # raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)


    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels
