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
import cv2

from utils import logger_init
from model import *
from data import *
from fretboard import  overlay_image_alpha, get_fretborad
from main import testing
from fit_rectangle import find_corners
from homography import get_warped_image


if __name__ == '__main__':

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)
    class_name = ''

    # test_data_dir = 'data/guitar/dataset_frames1_val/'
    test_data_dir = 'data/guitar/dataset_frames1_val_aug_v3/'
    test_images_list = glob.glob(os.path.join(test_data_dir, class_name, 'image', '*' + '.jpg'))
    test_images_list = test_images_list[0:50]


    ### UNet Prediction ###
    pred_masks = testing(test_images_list)
    logging.debug('pred_masks {}'.format(pred_masks.shape)) # (4, 640, 640, 1)


    ### Cornors ###
    output_dir_contour = os.path.join(output_dir, 'contours')
    os.makedirs(output_dir_contour, exist_ok=True)
    corners_list = []
    for pred_idx, pred_mask in enumerate(pred_masks):
        image_name = os.path.basename(test_images_list[pred_idx]).split('.')[0]
        corners_ret = find_corners(output_dir_contour, np.squeeze((pred_mask*255).astype(np.uint8)), image_name)
        # TODO: Handle < 4 corners
        if corners_ret is not None:
            corners = corners_ret
            # corners = [corners_ret[0][0], corners_ret[1][0], corners_ret[2][0], corners_ret[3][0]]
            corners_list.append(corners)
            logging.debug('corners {}'.format(corners))
        else:
            # TODO: hardcoding 
            logging.error('corners_ret {}'.format(corners_ret))
            corners = [[0,0], [0,0], [0,0], [0,0]]
            corners_list.append(corners)
    
    if len(corners_list) == 0:
        exit(0)
    # corners = array([
    #    [[567, 382]],
    #    [[260, 412]],
    #    [[262, 473]],
    #    [[567, 428]]], dtype=int32)

    ### Template Wrap ###
    im_template = get_fretborad()

    # TODO: Don't read again
    for idx, im_dst_path in enumerate(test_images_list):
        im_dst_name = os.path.basename(im_dst_path)
        im_dst = cv2.imread(im_dst_path)

        try:
            template_coords = corners_list[idx]
        except:
            logging.error('template_coords {}'.format(template_coords))
            continue

        #TODO: Check coordinate order for correctness
        im_warp = get_warped_image(im_template, im_dst, template_coords)
        if im_warp is not None:
            overlay_image_alpha(im_dst, im_warp[:, :, 0:3], (0, 0), im_warp[:, :, 3]/10)
            cv2.imwrite(os.path.join(output_dir, im_dst_name + '_overlay_jpg'), im_dst)



