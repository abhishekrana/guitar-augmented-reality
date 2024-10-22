from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
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

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask[mask == 38.0] = 1
        # mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        #new_mask = np.zeros(mask.shape + (num_class,))
        ## for i in range(num_class):
        #for idx, i in enumerate([0.0, 38.0]):
        #    #for one pixel in the image, find the class in mask and convert it into one-hot vector
        #    #index = np.where(mask == i)
        #    #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
        #    #new_mask[index_mask] = 1
        #    new_mask[mask == i, idx] = 1
        ## TODO:
        ## new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        #mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        # Hardcoding
        # mask = mask /255
        # print('mask', np.unique(mask, return_counts=True))
        mask[mask == 38.0] = 1

        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # np.unique(mask, return_counts=True)
        # mask[mask > 0.5] = 1
        # mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator2(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1,
                    class_mode=None, shuffle=False
                    ):
 
    # images_dir = os.path.join(train_path, image_folder)
    # labels_dir = os.path.join(train_path, mask_folder)
    # logging.debug('images_dir {}'.format(images_dir))
    # logging.debug('labels_dir {}'.format(labels_dir))

    images_list = glob.glob(os.path.join(train_path, image_folder, '*' + '.jpg'))
    # labels = glob.glob(os.path.join(images_dir, 'label', '*' + '.png'))

    # images = []
    # labels = []
    # for image_path in images_list:
    #     image = load_img(image_path) # target_size= # PIL
    #     x = img_to_array(image)
    #     images.append(x)

    #     label_path = os.path.join(train_path, mask_folder, os.path.basename(image_path).split('.')[0] + '.png')
    #     label = load_img(label_path) # target_size=
    #     y = img_to_array(label)
    #     labels.append(y)


        # labels.append([os.path.join(train_path, mask_folder, os.path.basename(image_path).split('.')[0] + '.png')])

    labels_list = []
    for image_path in images_list:
        labels_list.append(os.path.join(train_path, mask_folder, os.path.basename(image_path).split('.')[0] + '.png'))

    images = [np.asarray(Image.open(x)) for x in images_list]
    labels = [np.asarray(Image.open(x)) for x in labels_list]

    # images = np.array(images)
    # labels = np.array(labels)

    p = Augmentor.DataPipeline(images, labels)
    # p.rotate(1, max_left_rotation=5, max_right_rotation=5)
    # p.flip_top_bottom(0.5)
    p.zoom_random(1, percentage_area=0.5)

    # TODO: Batch size should be always 1
    img, mask = p.sample(1)
    # g = p.keras_generator(batch_size=1)

    img = np.array(img)
    mask = np.array(mask)

    img = np.squeeze(img)
    mask = np.squeeze(mask)

    img, mask = adjustData(img,mask,flag_multi_class,num_class)
    hash_str = str(random.getrandbits(16))

    img_rgb = Image.fromarray((img * 255).astype(np.uint8))
    img_rgb.save(os.path.join('output', 'aug', 'img-' + hash_str + '.jpg'))

    img_gs = Image.fromarray((mask * 255).astype(np.uint8))
    img_gs.save(os.path.join('output', 'aug', 'mask-' + hash_str + '.jpg'))

    # yield (img, mask)

    ### images, labels = next(g)
    # for img, mask in g:
    #     img, mask = adjustData(img,mask,flag_multi_class,num_class)

    #     hash_srt = str(random.getrandbits(16))
    #     img_rgb = Image.fromarray((img * 255).astype(numpy.uint8))
    #     img_rbg.save(os.path.join('output', 'aug', 'aug_image-' + hash_str + '.jpg'))


        # yield (img, mask)

    return None


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1,
                    class_mode=None, shuffle=False
                    ):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    # class_mode: One of "categorical", "binary", "sparse", "input", or None. Default: "categorical". 

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = class_mode,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix = image_save_prefix,
        seed = seed,
        shuffle = shuffle
        )
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = class_mode,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed,
        shuffle = shuffle
        )
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)


def testGenerator(test_path, test_images_list,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for img_path_name in test_images_list:
        # img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = io.imread(img_path_name, as_gray=as_gray)

        img = img / 255
        img = trans.resize(img,target_size)
        # TODO
        # img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)


        img_rgb = io.imread(img_path_name)
        img_rgb = trans.resize(img_rgb, target_size)
        io.imsave(os.path.join('output', os.path.basename(img_path_name)), img_rgb)

        yield img, img_path_name


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult(save_path,npyfile, test_gen_img_name, flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img_name = next(test_gen_img_name)
        img_name = os.path.basename(img_name).split('.')[0] + '_predict.png'
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path, img_name), img)
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)


def split_gen(gen):
    gen_a, gen_b = itertools.tee(gen, 2)
    return (a for a, b in gen_a), (b for a, b in gen_b)
