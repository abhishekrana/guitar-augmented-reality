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
    fb_w = 465.945 # 21st fret

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

    ### Frets
    fb_fret_dict = {}
    fb_fret_dict['f0'] = 0.0
    fb_fret_dict['f1'] = 36.353
    fb_fret_dict['f2'] = 70.665
    fb_fret_dict['f3'] = 103.051
    fb_fret_dict['f4'] = 133.620
    fb_fret_dict['f5'] = 162.473
    fb_fret_dict['f6'] = 189.707
    fb_fret_dict['f7'] = 215.412
    fb_fret_dict['f8'] = 239.675
    fb_fret_dict['f9'] = 262.575
    fb_fret_dict['f10'] = 284.191
    fb_fret_dict['f11'] = 304.593
    fb_fret_dict['f12'] = 323.850
    fb_fret_dict['f13'] = 342.026
    fb_fret_dict['f14'] = 359.182
    fb_fret_dict['f15'] = 375.376
    fb_fret_dict['f16'] = 390.660
    fb_fret_dict['f17'] = 405.087
    fb_fret_dict['f18'] = 418.703
    fb_fret_dict['f19'] = 431.556
    fb_fret_dict['f20'] = 443.687
    fb_fret_dict['f21'] = 455.138
    fb_fret_dict['f22'] = 465.945


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
        cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 2)

    for fret_name, fret_width in fb_fret_dict.items():
        x1, y1, x2, y2 = int(fret_width*fb_scale), int(0*fb_scale), int(fret_width*fb_scale), int(fb_h*fb_scale)
        cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 2)

    # Flip around axis
    cv2.flip(img_overlay, 1, img_overlay)

    return img_overlay

if __name__ == '__main__':
    # https://tlcguitargoods.com/en/howto-fret-calculator
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)

    img_dst = cv2.imread('data/guitar/test/2019-05-28-085835_1.jpg')
    fb_overlay = get_fretborad()
    overlay_image_alpha(img_dst,
                    fb_overlay[:, :, 0:3],
                    (0, 0),
                    fb_overlay[:, :, 3]/10)
    cv2.imwrite(os.path.join(output_dir, 'img_overlay_dst.jpg'), img_dst)
