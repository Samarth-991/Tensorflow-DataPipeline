import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,concatenate
from tensorflow.keras.layers import Dropout,MaxPooling2D,Input,Conv2DTranspose
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from create_data_pipeline import ParseTFRecord as parse_dataset
from configuration import EPOCHS,BATCH_SIZE,INIT_LR


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def unet(n_filters = 16, dropout = 0.1, batchnorm = True,image_shape=(512,512,3)):
    # Contracting Path
    in_src_image = Input(shape=image_shape)
    c1 = conv2d_block(in_src_image, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[in_src_image], outputs=[outputs])
    return model

def dice_coeff(y_true, y_pred):
    # this formula adds epsilon to the numerator and denomincator to avoid a divide by 0 error
    # in case a slice has no pixels set; the relative values are important, so this addition
    # does not effect the coefficient
    _epsilon = 10 ** -7
    intersections = tf.reduce_sum(y_true * y_pred)
    unions = tf.reduce_sum(y_true + y_pred)
    dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
    return dice_scores


def dice_loss(y_true, y_pred):
    #defined as 1 minues the dice coefficient
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss



if __name__ == '__main__':
    model = unet()
    get_custom_objects().update({"dice": dice_loss})
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR),
                  loss=dice_loss,
                  metrics=[dice_coeff])
    checkpointer = ModelCheckpoint('unet-cpu.h5', verbose=1)
    # Parse
    parse_tf_record = parse_dataset(tile_size=512)
    train_image_dataset  = parse_tf_record.parse_dataset('train_image_segmentation.tfrecords',compressed=True)
    train_dataset = train_image_dataset.map(parse_tf_record._parse_image_function)
    images = []
    masks = []
    for feature in train_dataset:
        image , mask = parse_tf_record.process_features(feature)
        images.append(image)
        masks.append(mask)

    model.fit(x=images,y=masks, batch_size=BATCH_SIZE, epochs=EPOCHS,callbacks=[checkpointer])
