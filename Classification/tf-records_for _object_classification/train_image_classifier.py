import os
import math
import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
import tensorflow as tf

from configuration import IMAGE_WIDTH, IMAGE_HEIGHT, INIT_LR, EPOCHS, NUM_CLASSES, BATCH_SIZE
from create_tfrecord_for_classifier import tf_record_for_object_classification as prepare_dataset


def get_model(model_name="mobilenet", freeze=False):
    model = None
    if model_name == 'mobilenet':
        # Pre-trained model with MobileNetV2
        base_model = MobileNetV2(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False, weights='imagenet')
        base_model.trainable = False

        # Trainable classification head
        maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
        prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
        model = tf.keras.Sequential([
            base_model,
            maxpool_layer,
            prediction_layer])
    return model


def create_tf_record(path, img_data, fname='train_image_classification.tfrecords'):
    tf_record_file = os.path.join(os.path.dirname(path), fname)
    if not os.path.isfile(tf_record_file):
        img_files = list(map(lambda x: os.path.join(path, x), img_data[:, 0]))
        img_labels = list(map(lambda x: 0 if x == 'cat' else 1, img_data[:, 1]))
        tf_record = prepare_dataset.TFRecordForObjectClassifiaction(tf_record_file)
        tf_record.dataset_to_tf_record(img_files, img_labels)
    return tf_record_file


def check_for_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    return len(gpus)


def main(args):
    print("GPU Found {}".format(check_for_gpu()))
    train_data = prepare_dataset.get_image_and_label(os.path.join(args.path, 'train'))
    valid_data = prepare_dataset.get_image_and_label(os.path.join(args.path, 'valid'))

    tfrecord_train = create_tf_record(os.path.join(args.path, 'train'), train_data, fname='train.tfrecords')
    tfrecord_valid = create_tf_record(os.path.join(args.path, 'valid'), valid_data, fname='valid.tfrecords')

    # Parse tf file
    if not os.path.isfile(tfrecord_train):
        print("NO {} file found ", format(tfrecord_train))
    parse_tf_record = prepare_dataset.ParseTFRecord(augmentataion=False,image_size=224)
    train_dataset = parse_tf_record.parse_dataset(tfrecord_train)
    train_count = parse_tf_record.get_the_length_of_dataset(train_dataset)
    print("Total Length of Data record:{}".format(train_count))

    valid_dataset = parse_tf_record.parse_dataset(tfrecord_valid)
    valid_count = parse_tf_record.get_the_length_of_dataset(valid_dataset)
    print("Total Length of Data record:{}".format(valid_count))

    # Get model
    classifier_model = get_model()
    # Optimizer
    optimizer = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    # loss
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    # metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(image_batch, label_batch):
        print(label_batch)
        with tf.GradientTape() as tape:
            predictions = classifier_model(image_batch)
            loss = loss_object(y_true=label_batch, y_pred=predictions)

        gradients = tape.gradient(loss, classifier_model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, classifier_model.trainable_variables))

        train_loss.update_state(values=loss)
        train_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    # @tf.function
    def valid_step(image_batch, label_batch):
        predictions = classifier_model(image_batch, training=False)
        v_loss = loss_object(label_batch, predictions)

        valid_loss.update_state(values=v_loss)
        valid_accuracy.update_state(y_true=label_batch, y_pred=predictions)

    for epoch in range(EPOCHS):
        step = 0
        for features in train_dataset.batch(BATCH_SIZE):
            step += 1
            image, label = parse_tf_record.process_features(features)
            train_step(image, label)
            if args.verbose ==2:
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch,
                                                                                     EPOCHS,
                                                                                     step,
                                                                                     math.ceil(
                                                                                         train_count / BATCH_SIZE),
                                                                                     train_loss.result().numpy(),
                                                                                     train_accuracy.result().numpy()))
        for features in valid_dataset.batch(1):
            valid_images, valid_labels = parse_tf_record.process_features(features)
            valid_step(valid_images, valid_labels)
            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                  "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch,
                                                                      EPOCHS,
                                                                      train_loss.result().numpy(),
                                                                      train_accuracy.result().numpy(),
                                                                      valid_loss.result().numpy(),
                                                                      valid_accuracy.result().numpy()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Azure Cognitive vision services for POI verification')
    parser.add_argument('-p', '--path', default=None, required=True, help='Data path')
    parser.add_argument('-v','--verbose',default=1,required=False)
    args = parser.parse_args()
    main(args)
