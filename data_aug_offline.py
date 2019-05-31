# https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline

from PIL import Image
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import itertools
import pudb
import logging
import Augmentor
import random

from utils import logger_init


def save_augmented_images_and_masks(augmented_images_masks, output_dir_aug, mask_in=None, prefix=None):
    img_dir=os.path.join(output_dir_aug, 'image')
    mask_dir=os.path.join(output_dir_aug, 'label')
    mask2_dir=os.path.join(output_dir_aug, 'label2')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask2_dir, exist_ok=True)

    for img_mask in augmented_images_masks:
        img = img_mask[0]

        if isinstance(mask_in, np.ndarray):
            mask = mask_in
        else:
            mask = img_mask[1]

        hash_str = str(random.getrandbits(64))
        if prefix:
            hash_str = prefix + '-' + hash_str

        img_rgb = Image.fromarray((img).astype(np.uint8))
        img_rgb.save(os.path.join(img_dir, 'img_' + hash_str + '.jpg'))

        img_gs = Image.fromarray((mask).astype(np.uint8))
        img_gs.save(os.path.join(mask_dir, 'mask_' + hash_str + '.png'))

        img_gs2rgb = Image.new('RGB', img_gs.size)
        img_gs2 = Image.fromarray((mask*255).astype(np.uint8))
        img_gs2rgb.paste(img_gs2)
        img_gs2rgb.save(os.path.join(mask2_dir, 'mask_' + hash_str + '.jpg'))


if __name__ == '__main__':
    seed = 7
    random.seed(seed)

    output_dir = 'output'
    output_dir_aug = os.path.join(output_dir, 'aug')
    output_dir_aug2 = os.path.join(output_dir, 'aug2')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_aug, exist_ok=True)
    os.makedirs(output_dir_aug2, exist_ok=True)

    filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)

    train_path = 'data/guitar/dataset_frames1_train_1'
    # train_path = 'data/guitar/dataset_frames1_train'
    # train_path = 'data/guitar/dataset_frames1_val'
    image_folder = 'image'
    mask_folder = 'label'


    images_list = glob.glob(os.path.join(train_path, image_folder, '*' + '.jpg'))
    labels_list = []
    for image_path in images_list:
        labels_list.append(os.path.join(train_path, mask_folder, os.path.basename(image_path).split('.')[0] + '.png'))
    collated_images_and_masks = list(zip(images_list, labels_list))
    images = [[np.asarray(Image.open(y)) for y in x] for x in collated_images_and_masks]
    y = len(images_list)*[1] # TODO: hardcoding from class 1

    p = Augmentor.DataPipeline(images, y)
    # p.set_seed(seed)
    # p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    # p.flip_top_bottom(0.5)
    # p.zoom_random(1, percentage_area=0.5)

    ### Train
    p.skew(0.25, magnitude=0.2) # To skew or tilt an image either left, right, forwards, or backwards and 8 cornors
    p.rotate(0.25, max_left_rotation=15, max_right_rotation=15) # You could combine this with a resize operation, 
    p.crop_random(0.25, percentage_area=0.7)
    p.zoom_random(0.25, percentage_area=0.8)
    augmented_images_masks, ys = p.sample(100)

    ### Val
    # p.rotate(0.25, max_left_rotation=10, max_right_rotation=10) # You could combine this with a resize operation, 
    # p.crop_random(0.25, percentage_area=0.9)
    # p.zoom_random(0.25, percentage_area=0.9)
    # p.random_erasing(0.5, rectangle_area=0.5)
    # augmented_images_masks, ys = p.sample(50)

    save_augmented_images_and_masks(augmented_images_masks, output_dir_aug)


    images_list = glob.glob(os.path.join(output_dir_aug, 'image', '*' + '.jpg'))
    labels_list = []
    for image_path in images_list:
        labels_list.append(os.path.join(output_dir_aug, 'label', 'mask_' + os.path.basename(image_path).split('.')[0].split('img_')[1] + '.png'))

    for image_path, mask_path in zip(images_list, labels_list):
        image = [np.asarray(Image.open(image_path))]
        mask = np.asarray(Image.open(mask_path))
        p2 = Augmentor.DataPipeline(image)
        # p2.set_seed(seed)

        ### Train
        ## p2.random_erasing(1.0, rectangle_area=0.9) # not working, TODO: fix
        p2.random_brightness(0.5, min_factor=0.1, max_factor=2.0)
        p2.random_contrast(0.5, min_factor=0.1, max_factor=2.0) # 1.0: original image
        augmented_images_masks_2 = p2.sample(4)

        ### Val
        ## p2.random_erasing(1.0, rectangle_area=0.9) # not working, TODO: fix
        # p2.random_brightness(0.5, min_factor=0.5, max_factor=1.5)
        # p2.random_contrast(0.5, min_factor=0.5, max_factor=1.5) # 1.0: original image
        # augmented_images_masks_2 = p2.sample(2)

        for idx, aum_2 in enumerate(augmented_images_masks_2):
            save_augmented_images_and_masks(np.array([[aum_2]]), output_dir_aug2, mask, 
                    prefix = os.path.basename(image_path).split('.')[0].split('img_')[1] + '_' + str(idx))


