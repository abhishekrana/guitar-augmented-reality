import os
import pudb
import glob
import numpy as np
import cv2
import logging
import random
import shutil

from utils import logger_init




if __name__ == '__main__':
    random.seed(7)

    # https://tlcguitargoods.com/en/howto-fret-calculator
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)

    dataset_dir = 'dataset/data_dataset_voc_frames1/'
    images_dir =  os.path.join(dataset_dir, 'JPEGImages')
    labels_dir =  os.path.join(dataset_dir, 'SegmentationClassPNG')

    images_list = glob.glob(os.path.join(images_dir, '*' + '.jpg'))
    images_list.sort()
    logging.debug('images_list {}'.format(len(images_list)))

    images_indices = np.arange(len(images_list))
    random.shuffle(images_indices)

    num_train_images = 30
    train_indices = images_indices[:num_train_images]
    val_indices = images_indices[num_train_images:]

    train_images = np.array(images_list)[train_indices]
    val_images = np.array(images_list)[val_indices]

    train_dir = 'data/guitar/dataset_frames1_train'
    val_dir = 'data/guitar/dataset_frames1_val'
    os.makedirs(os.path.join(train_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'label'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'label'), exist_ok=True)

    for train_image in train_images:
        shutil.copy(train_image, os.path.join(train_dir, 'image'))
        train_label = os.path.join(labels_dir, os.path.basename(train_image).split('.')[0] + '.png')
        shutil.copy(train_label, os.path.join(train_dir, 'label'))

    for val_image in val_images:
        shutil.copy(val_image, os.path.join(val_dir, 'image'))
        val_label = os.path.join(labels_dir, os.path.basename(val_image).split('.')[0] + '.png')
        shutil.copy(val_label, os.path.join(val_dir, 'label'))




