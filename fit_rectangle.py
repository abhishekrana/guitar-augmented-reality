# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html

import os
import pudb
import glob
import skimage.io as io
import numpy as np
import cv2
import logging
from utils import logger_init

def corners_arrange(corners, cx, cy):
    # corners = np.array([[[ 492,  476]],
    #        [[ 497,  566]],
    #        [[1149,  583]],
    #        [[1157,  535]]])
    # cx = 804
    # cy = 535

    corners_l = []
    corners_r = []
    for v in corners:
        x, y = v[0][0], v[0][1]
        # print('x, y: {}, {}'.format(x, y))
        if x < cx:
            corners_l.append([x,y])
        else:
            corners_r.append([x,y])


    try:
        top_left = None
        bottom_left = None
        if corners_l[0][0] < corners_l[1][0]:
            top_left = corners_l[0]
            bottom_left = corners_l[1]
        else:
            top_left = corners_l[1]
            bottom_left = corners_l[0]

        top_right = None
        bottom_right = None
        if corners_r[0][1] < corners_r[1][1]:
            top_right = corners_r[0]
            bottom_right = corners_r[1]
        else:
            top_right = corners_r[1]
            bottom_right = corners_r[0]
    except:
        logging.error('corners {}'.format(corners))
        logging.error('cx {}'.format(cx))
        logging.error('cy {}'.format(cy))
        return None

    corners_cw = [top_left, top_right, bottom_right, bottom_left]
    print('corners_cw', corners_cw)
    return corners_cw



def find_corners(output_dir_contour, img,image_name):
    # pu.db

    ret,thresh = cv2.threshold(img,127,255,0)
    # contours,hierarchy = cv2.findContours(thresh, 1, 2)
    contours, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    cnt = contours[0]
    M = cv2.moments(cnt)
    logging.debug('M {}'.format(M))

    # Centroids
    if M['m00'] == 0:
        return None

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.circle(img, center=(cx, cy), radius=5, color=(100, 100, 100), thickness=10)

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
        cv2.circle(img, center=(apx[0][0], apx[0][1]), radius=4, color=(150, 150, 150), thickness=10)

    # Rotated rectangle
    rect = cv2.minAreaRect(cnt)
    # Point center_of_rect = (r.br() + r.tl())*0.5;
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
            cv2.circle(img, center=(apx[0][0], apx[0][1]), radius=3, color=(200, 200, 200), thickness=10)

    else:
        return None

    # pu.db
    valid_corners_arranged =  corners_arrange(valid_corners, cx, cy)

    cv2.imwrite(os.path.join(output_dir_contour, image_name + '_contour' + '.jpg'), img)

    return valid_corners_arranged


if __name__ == '__main__':
    # https://tlcguitargoods.com/en/howto-fret-calculator
    output_dir = 'output'
    output_dir_contour = os.path.join(output_dir, 'contours')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_contour, exist_ok=True)
    filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)


    images_list = glob.glob(os.path.join(output_dir, 'test', '*' + '.png'))
    for image_path in images_list:
        image_name = os.path.basename(image_path).split('.')[0]
        img = cv2.imread(image_path ,0) # (640, 640)
        corners = find_corners(output_dir_contour, img, image_name)
        logging.debug('corners {}'.format(corners))

