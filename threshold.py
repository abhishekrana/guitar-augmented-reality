import os
import pudb
import glob
import skimage.io as io
import numpy as np
import cv2
import logging
from utils import logger_init


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)


    ## Read
    # img_path = 'assets/leaf.jpeg'
    img_path = 'data/guitar/test/2019-05-28-085835_1.jpg'
    img_name = os.path.basename(img_path).split('.')[0]
    img_gbr = cv2.imread(img_path)
    cv2.imwrite(os.path.join(output_dir, img_name + '_0' + '.jpg'), img_gbr)


    hsv_img = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(output_dir, img_name + '_1' + '.jpg'), hsv_img)

    green_low = np.array([45 , 100, 50] )
    green_high = np.array([75, 255, 255])
    curr_mask = cv2.inRange(hsv_img, green_low, green_high)
    hsv_img[curr_mask > 0] = ([75,255,200])
    cv2.imwrite(os.path.join(output_dir, img_name + '_2' + '.jpg'), hsv_img)


    ## converting the HSV image to Gray inorder to be able to apply 
    ## contouring
    RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(os.path.join(output_dir, img_name + '_3' + '.jpg'), gray)

    ret, threshold = cv2.threshold(gray, 90, 255, 0)
    cv2.imwrite(os.path.join(output_dir, img_name + '_4' + '.jpg'), threshold)

    contours, hierarchy =  cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_gbr, contours, -1, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(output_dir, img_name + '_5' + '.jpg'), img_gbr)



#     ## convert to hsv
#     # hsv = cv2.cvtColor(img_dst, cv2.COLOR_BGR2HSV)

#     ## mask of green (36,25,25) ~ (86, 255,255)
#     # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
#     mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))

#     ## slice the green
#     imask = mask>0
#     green = np.zeros_like(img_dst, np.uint8)
#     green[imask] = img_dst[imask]

#     ## save 
#     cv2.imwrite(os.path.join(output_dir, '', green)

    
