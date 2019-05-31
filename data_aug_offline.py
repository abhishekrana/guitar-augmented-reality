from PIL import Image
from PIL import ImagingPalette
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

if __name__ == '__main__':
    # https://tlcguitargoods.com/en/howto-fret-calculator
    output_dir = 'output'
    output_dir_aug = os.path.join(output_dir, 'aug')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_aug, exist_ok=True)

    filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)
    pu.db




# def trainGenerator2(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
#                     mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
#                     flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1,
#                     class_mode=None, shuffle=False
#                     ):
 
    train_path = 'data/guitar/dataset_frames1_train_1'
    image_folder = 'image'
    mask_folder = 'label'

    images_list = glob.glob(os.path.join(train_path, image_folder, '*' + '.jpg'))

    labels_list = []
    for image_path in images_list:
        labels_list.append(os.path.join(train_path, mask_folder, os.path.basename(image_path).split('.')[0] + '.png'))

    images = [np.asarray(Image.open(x)) for x in images_list] # 0-255
    labels = [np.asarray(Image.open(x)) for x in labels_list] # 0-1
    pu.db

    # images = np.array(images)
    # labels = np.array(labels)

    p = Augmentor.DataPipeline(images, labels)
    # p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p.flip_top_bottom(0.5)
    # p.zoom_random(1, percentage_area=0.5)

    # TODO: Batch size should be always 1
    imgs, masks = p.sample(4)

    imgs = np.array(imgs)
    masks = np.array(masks)

    # img = np.squeeze(img)
    # mask = np.squeeze(mask)

    for img, mask in zip(imgs, masks):
        hash_str = str(random.getrandbits(16))
        img_rgb = Image.fromarray((img).astype(np.uint8))
        img_rgb.save(os.path.join('output', 'aug', 'img-' + hash_str + '.jpg'))

        img_gs = Image.fromarray((mask).astype(np.uint8))
        img_gs.quantize(palette=ImagingPalette)
        pu.db
        img_gs.save(os.path.join('output', 'aug', 'mask-' + hash_str + '.png'))




    # img, mask = adjustData(img,mask,flag_multi_class,num_class)
    # img_rgb = Image.fromarray((img * 255).astype(np.uint8))
    # img_rgb.save(os.path.join('output', 'aug', 'img-' + hash_str + '.jpg'))

    # img_gs = Image.fromarray((mask * 255).astype(np.uint8))
    # img_gs.save(os.path.join('output', 'aug', 'mask-' + hash_str + '.jpg'))

    # yield (img, mask)

    ### images, labels = next(g)
    # for img, mask in g:
    #     img, mask = adjustData(img,mask,flag_multi_class,num_class)

    #     hash_srt = str(random.getrandbits(16))
    #     img_rgb = Image.fromarray((img * 255).astype(numpy.uint8))
    #     img_rbg.save(os.path.join('output', 'aug', 'aug_image-' + hash_str + '.jpg'))


        # yield (img, mask)



