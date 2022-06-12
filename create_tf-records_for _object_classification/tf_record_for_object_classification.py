import numpy as np
import tensorflow as tf

import random
import os

DATA_PATH = "../train"


class TFRecordForObjectClassifiaction:
    def __init__(self, fname, op='read'):
        self.tfrecord_name = fname

    def dataset_to_tf_record(self, image_files, label_files):
        with tf.io.TFRecordWriter(path=self.tfrecord_name) as writer:
            for file_name, label_name in zip(image_files, label_files):
                img_bytes = open(file_name, 'rb').read()
                single_feature = self.image_example(img_bytes, label_name)
                writer.write(single_feature)
        return

    def shuffle_dict(self, original_dict):
        keys = []
        shuffled_dict = {}
        for k in original_dict.keys():
            keys.append(k)
        random.shuffle(keys)
        for item in keys:
            shuffled_dict[item] = original_dict[item]
        return shuffled_dict

    @staticmethod
    def get_image_attributes(image_string):
        image_shape = tf.io.decode_jpeg(image_string).shape
        return image_shape

    @staticmethod
    def image_example(image_data, label):
        image_shape = TFRecordForObjectClassifiaction.get_image_attributes(image_data)
        feature = {
            'raw_image': TFRecordForObjectClassifiaction._bytes_feature(image_data),
            'height': TFRecordForObjectClassifiaction._int64_feature(image_shape[0]),
            'width': TFRecordForObjectClassifiaction._int64_feature(image_shape[1]),
            'label': TFRecordForObjectClassifiaction._int64_feature(label),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class ParseTFRecord:
    def __init__(self, augmentataion=False, image_size=512):
        self.augment = augmentataion
        self.image_size = image_size

    def parse_image_function(self, example_proto):
        # Parse the input tf.Example proto.
        return tf.io.parse_single_example(example_proto, {
            'label': tf.io.FixedLenFeature([], tf.dtypes.int64),
            'raw_image': tf.io.FixedLenFeature([], tf.dtypes.string),
        })

    def parse_dataset(self, tf_record):
        raw_dataset = tf.data.TFRecordDataset(tf_record)
        parsed_dataset = raw_dataset.map(self.parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
        return parsed_dataset

    def process_features(self, feature):
        image_raw = feature['raw_image'].numpy()
        image_tensor_list = []
        for image in image_raw:
            image_tensor = self.load_and_preprocess_image(image, image_size=self.image_size,
                                                          data_augmnetation=self.augment)
            image_tensor_list.append(image_tensor)
        images = tf.stack(image_tensor_list, axis=0)
        labels = feature['label'].numpy()
        return images, labels

    @staticmethod
    def get_the_length_of_dataset(dataset):
        count = 0
        for i in dataset:
            count += 1
        return count

    @staticmethod
    def load_and_preprocess_image(raw_image, image_size=512, data_augmnetation=False):
        # decode image
        image_tensor = tf.io.decode_jpeg(contents=raw_image, channels=3)
        image_tensor = tf.cast(image_tensor, tf.float32)

        if data_augmnetation:
            image_tensor = tf.image.random_brightness(image_tensor, max_delta=0.5)
            image_tensor = tf.image.random_flip_left_right(image_tensor)

        image_tensor = tf.image.resize(image_tensor, [image_size, image_size])
        return image_tensor


def get_image_and_label(dataset_dir):
    image_extensions = ['jpg', 'jpeg', 'png']
    image_list = [(img_name, img_name.split('.')[0]) for img_name in os.listdir(dataset_dir) if
                  img_name.split('.')[-1] in image_extensions]
    return np.asarray(image_list)


if __name__ == '__main__':
    img_data = get_image_and_label('../train')
    data_path = "/mnt/c/Users/tandon/OneDrive - TomTom/Desktop/tomtom/Workspace/Private_repo/Create_TF_records/train"

    # Create TF record
    img_files = list(map(lambda x: os.path.join(data_path, x), img_data[:, 0]))
    img_labels = list(map(lambda x: 0 if x == 'cat' else 1, img_data[:, 1]))
    tf_record = TFRecordForObjectClassifiaction('train_image_classification.tfrecords')
    # tf_dataset.dataset_to_tf_record(img_files,img_labels)

    parse_tf_record = ParseTFRecord(augmentataion=False)
    train_dataset = parse_tf_record.parse_dataset('train_image_classification.tfrecords')
    print("Total Length of Data record:{}".format(parse_tf_record.get_the_length_of_dataset(train_dataset)))

    for features in train_dataset.batch(10):
        image, label = parse_tf_record.process_features(features)
        print("", image.numpy().shape)
        break
