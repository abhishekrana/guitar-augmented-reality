import os
import pudb
import glob
import logging
import skimage.io as io
import skimage.transform as trans
import cv2
import numpy as np
import time
import shutil
import pickle

from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras

from utils import logger_init
from model import UNet
# from model import *
# from data import *
# from fretboard import FretBoard
from train import testing_load_model, testing_predict, testing_predict2
from fit_rectangle import find_corners
# from homography import get_warped_image

DEBUG_FLAG = True

def model_load(target_size, output_dir):
    num_classes = 2
    num_channels = 3
    output_dir_test = os.path.join(output_dir, 'test')
    os.makedirs(output_dir_test, exist_ok=True)

    input_size = (target_size[0], target_size[1], num_channels)
    logging.debug('input_size {}'.format(input_size))
    model = UNet(
           # pretrained_weights = '/media/abhishek/OS1/Abhishek/gar2/model_weights/model_weights_640x640.hdf5',
           # pretrained_weights = '/media/abhishek/OS1/Abhishek/gar2/model_weights/model_weights_1280x720_v1.hdf5',
           pretrained_weights = 'weights/model_weights/model_weights_640x640.hdf5',
           # pretrained_weights = 'weights/model_weights/model_weights_1280x720_v1.hdf5',
           input_size=input_size,
           num_classes = num_classes
           )

    return model


def prediction(model, test_images_list, target_size, batch_size, output_dir):

    output_dir_test = os.path.join(output_dir, 'test')

    num_test_images = len(test_images_list)
    images = []
    pred_masks = []

    for img_path_name in test_images_list:
        image_name = os.path.basename(img_path_name)
        if not os.path.exists(img_path_name):
            continue
        img = io.imread(img_path_name, as_gray=False)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)

        # img_rgb = io.imread(img_path_name)
        # img_rgb = trans.resize(img_rgb, target_size)
        # io.imsave(os.path.join(output_dir_test, os.path.basename(img_path_name)), (img_rgb*255).astype(np.uint8))

        pred_mask = model.predict(
                img,
                batch_size=batch_size,
                verbose=1
                )

        ## Save image
        mask_name = os.path.basename(img_path_name).split('.')[0] + '_predict.png'
        if DEBUG_FLAG:
            io.imsave(os.path.join(output_dir_test, mask_name), np.squeeze((pred_mask*255).astype(np.uint8)))

        images.extend(img)
        pred_masks.extend(pred_mask)
        # logging.debug('pred_mask {}'.format(pred_mask.shape))

    return np.array(pred_masks)



if __name__ == '__main__':

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    # fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)
    fileHandler, consoleHandler = logger_init(output_dir, logging.INFO)
    class_name = ''
    pred_corners_file = os.path.join(output_dir, 'pred_corners.pkl')

    # pred_dir = 'data/guitar/test_3/'
    # pred_dir = os.path.join(output_dir, 'frames_pred')
    # os.makedirs(pred_dir, exist_ok=True)

    # # test_images_list = glob.glob(os.path.join(test_data_dir, class_name, 'image', '*' + '.jpg'))
    # test_images_list = glob.glob(os.path.join(test_data_dir, '*' + '.jpg'))
    # # test_images_list = test_images_list[0:50]
    # test_images_list = test_images_list[0:8]
    # logging.debug('test_images_list {}'.format(test_images_list))


    # target_size = (720, 1280)
    # target_size = (640, 640)
    target_size = (480, 640)
    # target_size = (64, 64)

    batch_size = 1


    ### Load Model
    pred_model = model_load(target_size, output_dir)
    pred_image_file = os.path.join(output_dir, 'pred_image.jpg')
    test_images_list = [pred_image_file]

    counter = 0
    images_processed_count = 0
    while(True):
        counter += 1
        logging.info('Processed [{}/{}]'.format(images_processed_count, counter))

        # if os.path.isfile(pred_corners_file):
        if not os.path.isfile(pred_image_file):
            # logging.debug('No pred_image_file {}'.format(pred_image_file))
            time.sleep(0.5)
            continue

        ### UNet Prediction ###
        # pred_masks = testing_predict2(pred_model, test_images_list)
        pred_masks = prediction(pred_model, test_images_list, target_size, batch_size, output_dir)
        logging.debug('pred_masks {}'.format(pred_masks.shape)) # (4, 640, 640, 1)

        # shutil.rmtree(pred_dir)

        ### Cornors ###
        output_dir_contour = os.path.join(output_dir, 'contours')
        os.makedirs(output_dir_contour, exist_ok=True)
        corners_list = []
        for pred_idx, pred_mask in enumerate(pred_masks):
            image_name = os.path.basename(test_images_list[pred_idx]).split('.')[0]
            corners = find_corners(output_dir_contour, np.squeeze((pred_mask*255).astype(np.uint8)), image_name)
            # TODO: Handle < 4 corners
            if corners is not None:
                corners_list.append(corners)
                logging.debug('corners {}'.format(corners))

        if len(corners_list) != 0:
            images_processed_count += 1
            logging.debug('corners_list {}'.format(len(corners_list)))
            while os.path.exists(pred_corners_file):
                time.sleep(0.1)
            with open(pred_corners_file, 'wb') as fp:
                pickle.dump(corners_list, fp)
        else:
            if os.path.isfile(pred_image_file):
                os.remove(pred_image_file)
        





