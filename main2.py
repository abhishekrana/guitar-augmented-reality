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
from main import testing
from fit_rectangle import find_corners


if __name__ == '__main__':

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)


    class_name = ''

    test_data_dir = 'data/guitar/dataset_frames1_val/'
    # test_data_dir = 'data/guitar/dataset_frames1_val_aug_v3/'
    test_images_list = glob.glob(os.path.join(test_data_dir, class_name, 'image', '*' + '.jpg'))
    test_images_list = test_images_list[0:4]


    pu.db
    ### UNet Prediction ###
    pred_masks = testing(test_images_list)
    logging.debug('pred_masks {}'.format(pred_masks.shape)) # (4, 640, 640, 1)

    ### Cornors ###
    output_dir_contour = os.path.join(output_dir, 'contours')
    os.makedirs(output_dir_contour, exist_ok=True)
    for pred_idx, pred_mask in enumerate(pred_masks):
        image_name = os.path.basename(test_images_list[pred_idx]).split('.')[0]
        corners = find_corners(output_dir_contour, np.squeeze((pred_mask*255).astype(np.uint8)), image_name)
        logging.debug('corners {}'.format(corners))

