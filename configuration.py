
DEVICE = "gpu"   # cpu or gpu

MODEL = 'mobilenet'
# some training parameters
EPOCHS = 50
BATCH_SIZE = 8
NUM_CLASSES = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
INIT_LR = 1e-3

save_model_dir = "{}_{}".format(MODEL,'saved_model')
save_every_n_epoch = 10
test_image_dir = ""

# VALID_SET_RATIO = 1 - TRAIN_SET_RATIO - TEST_SET_RATIO
TRAIN_SET_RATIO = 0.6
TEST_SET_RATIO = 0.2


