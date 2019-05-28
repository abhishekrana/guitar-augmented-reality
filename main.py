import os
import pudb
import glob
import logging
import datetime
from keras import backend as K

from model import *
from data import *


### LOGGER
def logger_reset(fileHandler, consoleHandler):
    rootLogger = logging.getLogger()
    rootLogger.removeHandler(fileHandler)
    rootLogger.removeHandler(consoleHandler)

def logger_init(output_dir, log_level):
    logging.disabled = True

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
    log_file_path_name = os.path.join(output_dir, 'log_' + timestamp)
    logFormatter = logging.Formatter("%(asctime)s | %(filename)20s:%(lineno)s | %(funcName)20s() | %(message)s")

    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler(log_file_path_name)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    rootLogger.setLevel(log_level)

    logging.getLogger('PIL').setLevel(logging.ERROR)

    return fileHandler, consoleHandler

if __name__ == '__main__':

    logging.debug('K.image_data_format {}'.format(K.image_data_format()))

    ### Configuration
    output_dir = 'output'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)

    train_data_dir = 'data/guitar/train'
    test_data_dir = 'data/guitar/test'

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')
    data_gen_args = {}

    save_to_dir = None
    flag_multi_class = False
    num_channels = 3
    batch_size = 1
    epochs = 50
    target_size = (256, 256)
    num_classes = 2
    model_checkpoint = ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_weights.hdf5'),
            monitor='loss',
            verbose=1,
            save_best_only=True)

    if save_to_dir:
        os.makedirs(save_to_dir, exist_ok=True)

    class_name = ''
    image_ext = '.jpg'
    train_images_list = glob.glob(os.path.join(train_data_dir, class_name, 'image', '*' + image_ext))
    num_train_images = len(train_images_list)
    test_images_list = glob.glob(os.path.join(test_data_dir, class_name, '*' + image_ext))
    num_test_images = len(test_images_list)
    logging.debug('num_train_images {}'.format(num_train_images))
    logging.debug('num_test_images {}'.format(num_test_images))


    ### Data Generators
    train_gen = trainGenerator(
            batch_size,
            train_data_dir,
            'image',
            'label',
            data_gen_args,
            save_to_dir=save_to_dir,
            target_size=target_size,
            image_color_mode = 'rgb',
            mask_color_mode = 'grayscale',
            flag_multi_class = flag_multi_class
            )

    ### Model
    input_size = (target_size[0], target_size[1], num_channels)
    model = UNet(
            input_size=input_size,
            num_classes = num_classes
            )

    ### Train
    # tensorboard_dir = os.path.join(output_dir, 'summary')
    # tensorboard_cb = keras.callbacks.TensorBoard(
    #         log_dir=tensorboard_dir, 
    #         write_graph=True)

    history = model.fit_generator(
            train_gen, 
            steps_per_epoch=num_train_images//batch_size,
            epochs=epochs,
            callbacks=[model_checkpoint]
            # callbacks=[model_checkpoint, tensorboard_cb]
            # validation_data=validation_generator,
            # validation_steps=num_validation_samples // batch_size_val,
            )
    logging.debug('history {}'.format(history))

    ### Test
    test_gen = testGenerator(
            test_data_dir,
            test_images_list,
            target_size=target_size,
            flag_multi_class = flag_multi_class,
            as_gray = False
            )
    # filenames = test_gen.filenames
    # nb_samples = len(filenames)
    results = model.predict_generator(
            test_gen,
            num_test_images,
            verbose=1
            )
    saveResult(
            output_dir, 
            results,
            flag_multi_class = flag_multi_class
            )


