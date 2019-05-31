import os
import pudb
import glob
import logging
import skimage.io as io
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
import keras

from utils import logger_init
from model import *
from data import *
from fretboard import  overlay_image_alpha, get_fretborad


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    logging.debug('K.image_data_format {}'.format(K.image_data_format()))

    ### Configuration
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    output_dir = 'output'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)

    # train_data_dir = 'data/guitar/train'
    # test_data_dir = 'data/guitar/test'

    # train_data_dir = 'data/guitar/train_1'
    # test_data_dir = 'data/guitar/test_2'

    train_data_dir = 'data/guitar/dataset_frames1_train'
    val_data_dir = 'data/guitar/dataset_frames1_val'

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_gen_args = dict(
                        rotation_range=0.2,
                        # width_shift_range=0.05,
                        # height_shift_range=0.05,
                        # shear_range=0.05,
                        # zoom_range=0.05,
                        # horizontal_flip=False,
                        fill_mode='nearest'
                        )
    data_gen_args = {}
    save_to_dir = os.path.join(output_dir, 'aug')

    flag_multi_class = False
    num_channels = 3
    # batch_size = 2
    batch_size = 1
    epochs = 100
    # target_size = (256, 256)
    # target_size = (420, 1280)
    # target_size = (416, 1280)
    target_size = (720, 1280)
    num_classes = 2
    
    if save_to_dir:
        os.makedirs(save_to_dir, exist_ok=True)

    class_name = ''
    image_ext = '.jpg'
    train_images_list = glob.glob(os.path.join(train_data_dir, class_name, 'image', '*' + image_ext))
    num_train_images = len(train_images_list)
    image_ext = '.png'
    val_images_list = glob.glob(os.path.join(val_data_dir, class_name, 'label', '*' + image_ext))
    num_val_images = len(val_images_list)
    logging.debug('num_train_images {}'.format(num_train_images))
    logging.debug('num_val_images {}'.format(num_val_images))


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
            flag_multi_class = flag_multi_class,
            shuffle = True
            )

    val_gen = trainGenerator(
            batch_size,
            val_data_dir,
            'image',
            'label',
            aug_dict={},
            save_to_dir=None,
            target_size=target_size,
            image_color_mode = 'rgb',
            mask_color_mode = 'grayscale',
            flag_multi_class = flag_multi_class,
            shuffle = False
            )



    # train_flag = True
    # test_flag = False

    train_flag = False
    test_flag = True


    if train_flag:
        ### Model
        input_size = (target_size[0], target_size[1], num_channels)
        model = UNet(
                input_size=input_size,
                num_classes = num_classes
                )

        model_checkpoint = ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_weights.hdf5'),
            # monitor='loss',
            monitor='val_loss',
            verbose=1,
            save_best_only=True)

        ### Train
        # tensorboard_dir = os.path.join(output_dir, 'summary')
        # tensorboard_cb = keras.callbacks.TensorBoard(
        #         log_dir=tensorboard_dir, 
        #         write_graph=True)

        history = model.fit_generator(
                train_gen, 
                steps_per_epoch=num_train_images//batch_size,
                epochs=epochs,
                callbacks=[model_checkpoint],
                # callbacks=[model_checkpoint, tensorboard_cb]
                validation_data=val_gen,
                validation_steps=num_val_images // batch_size,
                )
        logging.debug('history {}'.format(history))

    ### Test
    if test_flag:
        # img = io.imread(test_images_list[0])
        # h, w, c = img.shape
        # input_size = (img.shape)
        # input_size = (h-4, w, c)

        test_data_dir = 'data/guitar/dataset_frames1_val/'
        test_images_list = glob.glob(os.path.join(test_data_dir, class_name, 'image', '*' + '.jpg'))
        num_test_images = len(test_images_list)
        input_size = (target_size[0], target_size[1], num_channels)
        logging.debug('input_size {}'.format(input_size))
        model = UNet(
               # pretrained_weights='output1/checkpoints/model_weights.hdf5',
               # pretrained_weights='output_6_model1/checkpoints/model_weights.hdf5',
               # pretrained_weights='output/checkpoints/model_weights.hdf5',
               pretrained_weights='output_7_model2-val_acc_0.9928-val_loss_0.0493/checkpoints/model_weights.hdf5',
               input_size=input_size,
               num_classes = num_classes
               )

        test_gen = testGenerator(
                test_data_dir,
                test_images_list,
                target_size=target_size,
                flag_multi_class = flag_multi_class,
                as_gray = False
                )

        test_gen_data, test_gen_img_name = split_gen(test_gen)

        # filenames = test_gen.filenames
        # nb_samples = len(filenames)
        results = model.predict_generator(
                test_gen_data,
                num_test_images,
                verbose=1
                )
        saveResult(
                output_dir, 
                results,
                test_gen_img_name,
                flag_multi_class = flag_multi_class
                )


