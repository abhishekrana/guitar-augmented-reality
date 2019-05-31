import os
import pudb
import glob
import skimage.io as io
import numpy as np
import cv2
import logging
from utils import logger_init


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def get_fretborad():
    
    # Measurement in mm
    fb_scale = 1.0
    fb_h = 42.0
    # fb_w = 323.85 # 12th fret
    fb_w = 455.138 # 21st fret
    # fb_w = 465.945 # 22st fret

    ### Strings
    fb_num_string = 6
    fb_string_dist = fb_h/fb_num_string
    fb_string_dist_offset = fb_string_dist/2
    fb_string_dict = {}
    fb_string_dict['s1'] = fb_string_dist*6.0 - fb_string_dist_offset
    fb_string_dict['s2'] = fb_string_dist*5.0 - fb_string_dist_offset
    fb_string_dict['s3'] = fb_string_dist*4.0 - fb_string_dist_offset
    fb_string_dict['s4'] = fb_string_dist*3.0 - fb_string_dist_offset
    fb_string_dict['s5'] = fb_string_dist*2.0 - fb_string_dist_offset
    fb_string_dict['s6'] = fb_string_dist*1.0 - fb_string_dist_offset

    ### Frets (Standard Fender Stratocaster)
    # fb_fret_dict = {}
    # fb_fret_dict['f0'] = 0.0
    # fb_fret_dict['f1'] = 36.353
    # fb_fret_dict['f2'] = 70.665
    # fb_fret_dict['f3'] = 103.051
    # fb_fret_dict['f4'] = 133.620
    # fb_fret_dict['f5'] = 162.473
    # fb_fret_dict['f6'] = 189.707
    # fb_fret_dict['f7'] = 215.412
    # fb_fret_dict['f8'] = 239.675
    # fb_fret_dict['f9'] = 262.575
    # fb_fret_dict['f10'] = 284.191
    # fb_fret_dict['f11'] = 304.593
    # fb_fret_dict['f12'] = 323.850
    # fb_fret_dict['f13'] = 342.026
    # fb_fret_dict['f14'] = 359.182
    # fb_fret_dict['f15'] = 375.376
    # fb_fret_dict['f16'] = 390.660
    # fb_fret_dict['f17'] = 405.087
    # fb_fret_dict['f18'] = 418.703
    # fb_fret_dict['f19'] = 431.556
    # fb_fret_dict['f20'] = 443.687
    # fb_fret_dict['f21'] = 455.138

    ### Frets (Fender Bullet Strat - Custom v1)
    # fb_fret_dict = {}
    # fb_fret_dict['f0'] = 0.0
    # fb_fret_dict['f1'] = 47.41
    # fb_fret_dict['f2'] = 88.05
    # fb_fret_dict['f3'] = 125.31
    # fb_fret_dict['f4'] = 159.00
    # fb_fret_dict['f5'] = 189.48
    # fb_fret_dict['f6'] = 217.59
    # fb_fret_dict['f7'] = 242.99
    # fb_fret_dict['f8'] = 266.19
    # fb_fret_dict['f9'] = 288.21
    # fb_fret_dict['f10'] = 308.19
    # fb_fret_dict['f11'] = 326.98
    # fb_fret_dict['f12'] = 344.93
    # fb_fret_dict['f13'] = 360.85
    # fb_fret_dict['f14'] = 375.58
    # fb_fret_dict['f15'] = 389.64
    # fb_fret_dict['f16'] = 402.34
    # fb_fret_dict['f17'] = 414.36
    # fb_fret_dict['f18'] = 425.53
    # fb_fret_dict['f19'] = 436.03
    # fb_fret_dict['f20'] = 445.52
    # fb_fret_dict['f21'] = 455.138


    ### Frets (Fender Bullet Strat - Custom v2)
    fb_fret_dict = {}
    fb_fret_dict['f0'] = 0.0
    fb_fret_dict['f1'] = 48
    fb_fret_dict['f2'] = 90
    fb_fret_dict['f3'] = 126
    fb_fret_dict['f4'] = 161
    fb_fret_dict['f5'] = 192
    fb_fret_dict['f6'] = 220
    fb_fret_dict['f7'] = 245
    fb_fret_dict['f8'] = 270
    fb_fret_dict['f9'] = 292
    fb_fret_dict['f10'] = 312.5
    fb_fret_dict['f11'] = 329
    fb_fret_dict['f12'] = 348
    fb_fret_dict['f13'] = 364
    fb_fret_dict['f14'] = 379
    fb_fret_dict['f15'] = 392
    fb_fret_dict['f16'] = 405
    fb_fret_dict['f17'] = 416
    fb_fret_dict['f18'] = 427
    fb_fret_dict['f19'] = 437
    fb_fret_dict['f20'] = 446
    fb_fret_dict['f21'] = 455


    # img_overlay = img_dst.copy()
    # 4th channel is alpha
    # fb_scale = 2.0
    img_overlay = np.zeros((int(fb_h*fb_scale), int(fb_w*fb_scale), 4), np.uint8)
    # fb_scale = 1.0
    img_overlay[:,:,0:3] = 255
    img_overlay[:,:,3] = 3 # Alpha channel values: [0:10]

    ### FretBoard
    x, y, w, h = int(0*fb_scale), int(0*fb_scale), int(fb_w*fb_scale), int(fb_h*fb_scale)
    logging.debug('x,y,w,h: {} {} {} {}'.format(x, y, w, h))
    cv2.rectangle(img_overlay, (x, y), (x+w, y+h), (0, 200, 0), 0)  # A filled rectangle
    # alpha = 0.4
    # img_overlay = cv2.addWeighted(img_overlay, alpha, img_overlay, 1 - alpha, 0)
    # img_overlay = cv2.addWeighted(src1=overlay, alpha=1, src2=img_dst, beta=0.5, gamma=0)

    ### Strings
    for string_name, string_height in fb_string_dict.items():
        x1, y1, x2, y2 = int(0*fb_scale), int(string_height*fb_scale), int(fb_w*fb_scale), int(string_height*fb_scale)
        cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 1)

    for fret_name, fret_width in fb_fret_dict.items():
        x1, y1, x2, y2 = int(fret_width*fb_scale), int(0*fb_scale), int(fret_width*fb_scale), int(fb_h*fb_scale)
        cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 1)

    # Flip around axis
    cv2.flip(img_overlay, 1, img_overlay)

    return img_overlay

if __name__ == '__main__':
    # https://tlcguitargoods.com/en/howto-fret-calculator
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)

    img_dst = cv2.imread('data/guitar/test/2019-05-28-085835_1.jpg')
    fb_overlay = get_fretborad()
    overlay_image_alpha(img_dst,
                    fb_overlay[:, :, 0:3],
                    (0, 0),
                    fb_overlay[:, :, 3]/10)
    cv2.imwrite(os.path.join(output_dir, 'img_overlay_dst.jpg'), img_dst)
