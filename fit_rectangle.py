# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

import os
import pudb
import glob
import skimage.io as io
import numpy as np
import cv2
import logging
from utils import logger_init


def find_corners(output_dir_contour, img,image_name):

    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 1)
    cnt = contours[0]
    M = cv2.moments(cnt)
    logging.debug('M {}'.format(M))

    # Centroids
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    # area = cv2.contourArea(cnt)
    # perimeter = cv2.arcLength(cnt,True)

    # # Convex hull
    # # hull = cv2.convexHull(points[, hull[, clockwise[, returnPoints]]
    # hull = cv2.convexHull(cnt)

    # Contour Approximation
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    logging.debug('approx {}'.format(approx))
    for apx in approx:
        cv2.circle(img, center=(apx[0][0], apx[0][1]), radius=1, color=(200, 200, 200), thickness=10)

    # Rotated rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(127,127,255),2)


    valid_corners = []
    if len(approx) == 4:
        valid_corners = approx

    elif len(approx) > 4:
        a_idx_closest = []
        for b in box:
            a_idx_list = []
            for a_idx, a in enumerate(approx):
                bx, by = b[0], b[1]
                ax, ay = a[0][0], a[0][1]
                dist = np.square(bx - ax) + np.square(by - ay)
                a_idx_list.append(dist)
            a_idx_closest.append(np.argmin(a_idx_list))

        for a_idx in a_idx_closest:
            valid_corners.append(approx[a_idx])
        valid_corners = np.array(valid_corners)
        logging.debug('valid_corners {}'.format(valid_corners))

        for apx in valid_corners:
            cv2.circle(img, center=(apx[0][0], apx[0][1]), radius=3, color=(250, 250, 250), thickness=15)

    else:
        valid_corners = None

    cv2.imwrite(os.path.join(output_dir_contour, image_name + '_contour' + '.jpg'), img)

    return valid_corners


if __name__ == '__main__':
    # https://tlcguitargoods.com/en/howto-fret-calculator
    output_dir = 'output'
    output_dir_contour = os.path.join(output_dir, 'contours')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_contour, exist_ok=True)
    filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)


    images_list = glob.glob(os.path.join(output_dir, 'test', '*' + '.png'))
    pu.db
    for image_path in images_list:
        image_name = os.path.basename(image_path).split('.')[0]
        img = cv2.imread(image_path ,0) # (640, 640)
        corners = find_corners(output_dir_contour, img, image_name)
        logging.debug('corners {}'.format(corners))

