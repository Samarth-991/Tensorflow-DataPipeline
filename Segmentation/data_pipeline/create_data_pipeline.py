import logging

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow_io as tfio
import tifffile
from PIL import Image,ImageOps
import os
import gc


class TFRecordForSegmentation:
    def __init__(self, fname):
        self.tf_record_filename = fname

    def serialize_images(self, image_id, image, mask,shape):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        image_bytes = image.tobytes()
        mask_bytes = mask.tobytes()
        feature_dict = {
            'image': self._bytes_feature(image_bytes),
            'mask': self._bytes_feature(mask_bytes),
            'image_id': self._bytes_feature(image_id),
            'width':self._int64_feature(shape[0]),
            'height':self._int64_feature(shape[1])
        }
        # Create a Features message using tf.train.Example.
        message_feature = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        return message_feature.SerializeToString()

    def create_tf_record(self, image_id, image, mask_image, tile_size):
        tile_count = 0
        logging.info("Creating tensorflow Record..")
        if not os.path.isfile(self.tf_record_filename):
            compress = tf.io.TFRecordOptions(compression_type="GZIP")
            with tf.io.TFRecordWriter(self.tf_record_filename, compress) as writer:
                for img, mask, id, shape in tqdm(zip(image, mask_image, image_id, tile_size)):
                    single_feature = self.serialize_images(id, image, mask, shape)
                    writer.write(single_feature)
                    tile_count = tile_count + 1
            writer.close()
        return tile_count

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    # For the mask
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class ParseTFRecord:
    def __init__(self, tile_size=512, is_compressed=False):
        self.target_img_size = tile_size
        self.is_tf_compressed = is_compressed
        self.image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.dtypes.string),
            'mask': tf.io.FixedLenFeature([], tf.dtypes.string),
            'image_id': tf.io.FixedLenFeature([], tf.dtypes.string),
            'width':tf.io.FixedLenFeature([],tf.int64),
            'height':tf.io.FixedLenFeature([],tf.int64)
        }

    def parse_dataset(self, tf_record, compressed=False):
        parsed_dataset = list()
        raw_dataset = tf.data.TFRecordDataset(tf_record, compression_type="GZIP")
        # parsed_dataset = raw_dataset.map(self._parse_image_function)
        return raw_dataset

    def process_features(self, example_proto,tile_size=512):
        width = example_proto['width'].numpy()
        height = example_proto['height'].numpy()
        image_id = example_proto['image_id'].numpy()
        image = tf.io.decode_raw(example_proto['image'],out_type='uint8')
        img_array = tf.reshape(image, (tile_size, tile_size, 3))

        mask = tf.io.decode_raw(example_proto['mask'],out_type='bool')
        mask = tf.reshape(mask, (tile_size, tile_size))

        return img_array,mask

    def _parse_image_function(self, example_proto):
        return tf.io.parse_single_example(example_proto,
                                          self.image_feature_description)


def normalize(input_image):
    """
    The colour channel sometimes occurred in the first column and sometimes in the third.
    Some images also had leading dimensions of size 1 which we remove using the squeeze function.
    We run the garbage collector to free up memory space after calling the function
    :param input_image:
    :return: image,width,height
    """
    image = tifffile.imread(input_image)
    image = tf.squeeze(image)
    if image.shape[0] == 3:
        image = tf.transpose(image, [2, 1, 0])
    image = tf.cast(image, tf.float32) / 255.0
    return image, image.shape[0], image.shape[1]


def read_mask_image(input_image):
    mask = tifffile.imread(input_image[:-1])
    mask = tf.squeeze(mask)
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask.numpy()


def generate_records(file_list):
    image_data = list()
    tf_file_name = "train_image_segmentation.tfrecords"
    tf_segmentation = TFRecordForSegmentation(tf_file_name)
    for filename in file_list:
        image, shape0, shape1 = normalize(filename)
        image_name = os.path.basename(filename)
        image_id = bytes(image_name.split('_')[0], 'utf8')
        mask = read_mask_image(filename.replace('train', 'train_labels'))
        image_data.append([image, mask, image_id, (shape0, shape1)])
        # tile_count = tf_segmentation.create_tf_record(image_id, image, mask, tile_size)
    return np.asarray(image_data)


if __name__ == '__main__':
    file_list = tf.io.gfile.glob('../dataset/train/*.tiff')
    image_data = generate_records(file_list=file_list)

    # Create TF record
    tf_record = TFRecordForSegmentation('train_image_segmentation.tfrecords')
    tile_count = tf_record.create_tf_record(image=image_data[:, 0], mask_image=image_data[:, 1],
                                            image_id=image_data[:, 2], tile_size=image_data[:, 3])
    print("total number of features written {}".format(tile_count))
