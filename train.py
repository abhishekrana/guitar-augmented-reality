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



def testing_load_model():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    num_classes = 2
    num_channels = 3
    # target_size = (720, 1280)
    # target_size = (640, 640)
    target_size = (480, 640)
    # target_size = (64, 64)
    output_dir = 'output'
    batch_size = 1

    output_dir_test = os.path.join(output_dir, 'test')
    os.makedirs(output_dir_test, exist_ok=True)

    # test_data_dir = 'data/guitar/dataset_frames1_val/'
    # test_data_dir = 'data/guitar/dataset_frames1_val_aug_v3/'
    # test_images_list = glob.glob(os.path.join(test_data_dir, class_name, 'image', '*' + '.jpg'))
    # test_images_list = test_images_list[0:100]

    input_size = (target_size[0], target_size[1], num_channels)
    logging.debug('input_size {}'.format(input_size))
    model = UNet(
           pretrained_weights = '/media/abhishek/OS1/Abhishek/gar2/model_weights/model_weights_640x640.hdf5',
           # pretrained_weights = '/media/abhishek/OS1/Abhishek/gar2/model_weights/model_weights_1280x720_v1.hdf5',
           # pretrained_weights = 'weights/model_weights/model_weights_640x640.hdf5',
           # pretrained_weights = 'weights/model_weights/model_weights_1280x720_v1.hdf5',
           input_size=input_size,
           num_classes = num_classes
           )

    return model



def testing_predict(model, frames):
    flag_multi_class = False
    target_size = (480, 640)
    # target_size = (640, 640)
    batch_size = 1
    output_dir = 'output'
    output_dir_test = os.path.join(output_dir, 'test')

    # num_test_images = len(test_images_list)
    num_test_images = 1
    logging.debug('num_test_images {}'.format(num_test_images))

    images = []
    pred_masks = []
    for img in frames:

        # Swap channels
        img = img[...,::-1] 

        # image_name = os.path.basename(img_path_name)
        # img = io.imread(img_path_name, as_gray=False)
        # img = np.expand_dims(img, axis=0)
        img = img / 255
        img = trans.resize(img,target_size)

        # TODO
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)

        # img_rgb = io.imread(img_path_name)
        # img_rgb = trans.resize(img_rgb, target_size)
        # io.imsave(os.path.join(output_dir_test, os.path.basename(img_path_name)), (img_rgb*255).astype(np.uint8))

        
        logging.debug('Prediction Starts')
        pred_mask = model.predict(
                img,
                # batch_size=num_test_images,
                batch_size=batch_size,
                verbose=1
                )
        logging.debug('Prediction Ends')
        # logging.debug('pred_mask {}'.format(pred_mask))

        ## Save image
        # pu.db
        image_name = 'image_1.jpg'
        io.imsave(os.path.join(output_dir_test, image_name), np.squeeze((img*255).astype(np.uint8)))

        # mask_name = os.path.basename(img_path_name).split('.')[0] + '_predict.png'
        mask_name = 'mask_1.jpg'
        io.imsave(os.path.join(output_dir_test, mask_name), np.squeeze((pred_mask*255).astype(np.uint8)))
        # img = labelVisualize(num_classes,COLOR_DICT,pred_mask) if flag_multi_class else pred_mask[:,:,0]
        # io.imsave(os.path.join(output_dir_test, mask_name), img)

        # images.extend(img)
        pred_masks.extend(pred_mask)
        logging.debug('pred_mask {}'.format(pred_mask.shape))

    return np.array(pred_masks)


def testing_predict2(model, test_images_list):
    flag_multi_class = False
    target_size = (480, 640)
    batch_size = 1
    output_dir = 'output'
    output_dir_test = os.path.join(output_dir, 'test')

    num_test_images = len(test_images_list)
    logging.debug('num_test_images {}'.format(num_test_images))

    images = []
    pred_masks = []
    for img_path_name in test_images_list:
        image_name = os.path.basename(img_path_name)
        img = io.imread(img_path_name, as_gray=False)
        img = img / 255
        img = trans.resize(img,target_size)
        # TODO
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)

        img_rgb = io.imread(img_path_name)
        img_rgb = trans.resize(img_rgb, target_size)
        io.imsave(os.path.join(output_dir_test, os.path.basename(img_path_name)), (img_rgb*255).astype(np.uint8))

        pred_mask = model.predict(
                img,
                # batch_size=num_test_images,
                batch_size=batch_size,
                verbose=1
                )
        # logging.debug('pred_mask {}'.format(pred_mask))

        ## Save image
        mask_name = os.path.basename(img_path_name).split('.')[0] + '_predict.png'
        io.imsave(os.path.join(output_dir_test, mask_name), np.squeeze((pred_mask*255).astype(np.uint8)))
        # img = labelVisualize(num_classes,COLOR_DICT,pred_mask) if flag_multi_class else pred_mask[:,:,0]
        # io.imsave(os.path.join(output_dir_test, mask_name), img)

        images.extend(img)
        pred_masks.extend(pred_mask)
        logging.debug('pred_mask {}'.format(pred_mask.shape))

    return np.array(pred_masks)




if __name__ == '__main__':

    # train_flag = True
    # test_flag = False
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    train_flag = False
    test_flag = True
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    logging.debug('K.image_data_format {}'.format(K.image_data_format()))

    ### Configuration
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    output_dir = 'output_train'
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)

    # train_data_dir = 'data/guitar/train'
    # test_data_dir = 'data/guitar/test'

    # train_data_dir = 'data/guitar/train_1'
    # test_data_dir = 'data/guitar/test_2'

    # train_data_dir = 'data/guitar/dataset_frames1_train_1'
    # val_data_dir = 'data/guitar/dataset_frames1_val_1'

    # train_data_dir = 'data/guitar/dataset_frames1_train'
    # val_data_dir = 'data/guitar/dataset_frames1_val'

    # train_data_dir = 'data/guitar/dataset_frames1_train_aug_v2'
    # val_data_dir = 'data/guitar/dataset_frames1_val_aug_v2'


    # train_data_dir = 'data/guitar/dataset_frames1_train_aug_v2'
    # val_data_dir = 'data/guitar/dataset_frames1_val_aug_v3'

    # train_data_dir = 'data/guitar/dataset_frames1_train_aug_v3'
    # val_data_dir = 'data/guitar/dataset_frames1_val_aug_v3'


    train_data_dir = 'data/guitar/dataset_frames1_train_aug_v4'
    val_data_dir = 'data/guitar/dataset_frames1_val_aug_v4'


    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    data_gen_args = dict(
                        # width_shift_range=0.05,
                        # height_shift_range=0.05,
                        # width_shift_range=0.0,
                        # height_shift_range=0.0,
                        # shear_range=0.05,
                        # horizontal_flip=False,
                        rotation_range=90,
                        zoom_range=0.5, # if you specify 0.3, then the range will be [0.7, 1.3], or between 70% (zoom in) and 130% (zoom out).
                        # brightness_range=(1.0, 2.0), # BUG: don't use. brightness_range: Values less than 1.0 should darken the image (currently a BUG). Values larger than 1.0 brighten the image, e.g. [1.0, 1.5], where 1.0 has no effect on brightness.
                        fill_mode='constant'
                        )
    data_gen_args = {}
    save_to_dir = os.path.join(output_dir, 'aug')
    save_to_dir = None

    flag_multi_class = False
    num_channels = 3
    # batch_size = 2
    # batch_size = 8
    batch_size = 1
    epochs = 200
    # epochs = 1
    # target_size = (256, 256)
    # target_size = (420, 1280)
    # target_size = (416, 1280)

    # target_size = (640, 640)
    target_size = (720, 1280)
    # target_size = (480, 640)

    # target_size = (360, 640)
    # target_size = (64, 64)
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
            aug_dict=data_gen_args,
            save_to_dir=None,
            target_size=target_size,
            image_color_mode = 'rgb',
            mask_color_mode = 'grayscale',
            flag_multi_class = flag_multi_class,
            shuffle = False
            )



    
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

        num_train_images = 100
        history = model.fit_generator(
                train_gen, 
                steps_per_epoch=num_train_images//batch_size,
                epochs=epochs,
                callbacks=[model_checkpoint],
                # callbacks=[model_checkpoint, tensorboard_cb]
                validation_data=val_gen,
                validation_steps=num_val_images//batch_size,
                )
        logging.debug('history {}'.format(history))

    ### Test
    if test_flag:
        # img = io.imread(test_images_list[0])
        # h, w, c = img.shape
        # input_size = (img.shape)
        # input_size = (h-4, w, c)

        output_dir_test = os.path.join(output_dir, 'test')
        os.makedirs(output_dir_test, exist_ok=True)

        # test_data_dir = 'data/guitar/dataset_frames1_val/'
        test_data_dir = 'data/guitar/dataset_frames1_val_aug_v3/'
        test_images_list = glob.glob(os.path.join(test_data_dir, class_name, 'image', '*' + '.jpg'))
        test_images_list = test_images_list[0:100]

        num_test_images = len(test_images_list)
        input_size = (target_size[0], target_size[1], num_channels)
        logging.debug('input_size {}'.format(input_size))
        model = UNet(
               # pretrained_weights='output/checkpoints/model_weights.hdf5',
               # pretrained_weights='output1/checkpoints/model_weights.hdf5',
               # pretrained_weights='output_6_model1/checkpoints/model_weights.hdf5',
               # pretrained_weights='output/checkpoints/model_weights.hdf5',
               # pretrained_weights='output_7_model2-val_acc_0.9928-val_loss_0.0493/checkpoints/model_weights.hdf5',
               # pretrained_weights='output_8_wrong/checkpoints/model_weights.hdf5',
               # pretrained_weights='output_10_model3_aug/checkpoints/model_weights.hdf5',
               pretrained_weights=os.path.join(output_dir, 'checkpoints/model_weights.hdf5'),
               # pretrained_weights='output_10_aug/checkpoints/model_weights.hdf5', # Best, 640x640
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
                output_dir_test, 
                results,
                test_gen_img_name,
                flag_multi_class = flag_multi_class
                )


