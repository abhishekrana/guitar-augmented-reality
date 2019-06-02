import os
import pudb
import glob
import skimage.io as io
import numpy as np
import cv2
import logging
from collections import OrderedDict
from utils import logger_init


class FretBoard():

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)

        self.setup_fretboard()
        self.tab_got()


    def overlay_image_alpha(self, img, img_overlay, pos, alpha_mask):
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

    def setup_fretboard(self):
        
        # Measurement in mm
        self.fb_scale = 1.0
        self.fb_h = 42.0
        # self.fb_w = 323.85 # 12th fret
        self.fb_w = 455.138 # 21st fret
        # self.fb_w = 465.945 # 22st fret

        ### Strings
        fb_num_string = 6
        fb_string_dist = self.fb_h/fb_num_string
        fb_string_dist_offset = fb_string_dist/2
        self.fb_string_dict = {}
        self.fb_string_dict['s1'] = fb_string_dist*6.0 - fb_string_dist_offset
        self.fb_string_dict['s2'] = fb_string_dist*5.0 - fb_string_dist_offset
        self.fb_string_dict['s3'] = fb_string_dist*4.0 - fb_string_dist_offset
        self.fb_string_dict['s4'] = fb_string_dist*3.0 - fb_string_dist_offset
        self.fb_string_dict['s5'] = fb_string_dist*2.0 - fb_string_dist_offset
        self.fb_string_dict['s6'] = fb_string_dist*1.0 - fb_string_dist_offset

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
        # fb_fret_dict = {}
        # fb_fret_dict['f0'] = 0.0
        # fb_fret_dict['f1'] = 48
        # fb_fret_dict['f2'] = 90
        # fb_fret_dict['f3'] = 126
        # fb_fret_dict['f4'] = 161
        # fb_fret_dict['f5'] = 192
        # fb_fret_dict['f6'] = 220
        # fb_fret_dict['f7'] = 245
        # fb_fret_dict['f8'] = 270
        # fb_fret_dict['f9'] = 292
        # fb_fret_dict['f10'] = 312.5
        # fb_fret_dict['f11'] = 329
        # fb_fret_dict['f12'] = 348
        # fb_fret_dict['f13'] = 364
        # fb_fret_dict['f14'] = 379
        # fb_fret_dict['f15'] = 392
        # fb_fret_dict['f16'] = 405
        # fb_fret_dict['f17'] = 416
        # fb_fret_dict['f18'] = 427
        # fb_fret_dict['f19'] = 437
        # fb_fret_dict['f20'] = 446
        # fb_fret_dict['f21'] = 455


        ### Frets (Fender Bullet Strat - Custom v2)
        # Mantaining order of insertion by using OrderedDict
        self.fb_fret_dict = OrderedDict([
            ('f0', 0.0),    ('f1', 48.0),   ('f2', 90.0),   ('f3', 126.0),  ('f4', 161.0),   ('f5', 192.0),  
            ('f6', 220.0),  ('f7', 245.0),  ('f8', 270.0),  ('f9', 292.0),  ('f10', 312.5),  ('f11', 329.0), 
            ('f12', 348.0), ('f13', 364.0), ('f14', 379.0), ('f15', 392.0), ('f16', 405.0),  ('f17', 416.0), 
            ('f18', 427.0), ('f19', 437.0), ('f20', 446.0), ('f21', 455.0)
            ])

        self.fb_string_fret_dict = OrderedDict([
            # String 1
            ('s1f1', (int((self.fb_fret_dict['f1'] + self.fb_fret_dict['f0'])/2), int(self.fb_string_dict['s1']))),
            ('s1f2', (int((self.fb_fret_dict['f2'] + self.fb_fret_dict['f1'])/2), int(self.fb_string_dict['s1']))),
            ('s1f3', (int((self.fb_fret_dict['f3'] + self.fb_fret_dict['f2'])/2), int(self.fb_string_dict['s1']))),
            ('s1f4', (int((self.fb_fret_dict['f4'] + self.fb_fret_dict['f3'])/2), int(self.fb_string_dict['s1']))),
            ('s1f5', (int((self.fb_fret_dict['f5'] + self.fb_fret_dict['f4'])/2), int(self.fb_string_dict['s1']))),
            ('s1f6', (int((self.fb_fret_dict['f6'] + self.fb_fret_dict['f5'])/2), int(self.fb_string_dict['s1']))),
            ('s1f7', (int((self.fb_fret_dict['f7'] + self.fb_fret_dict['f6'])/2), int(self.fb_string_dict['s1']))),
            ('s1f8', (int((self.fb_fret_dict['f8'] + self.fb_fret_dict['f7'])/2), int(self.fb_string_dict['s1']))),
            ('s1f9', (int((self.fb_fret_dict['f9'] + self.fb_fret_dict['f8'])/2), int(self.fb_string_dict['s1']))),
            ('s1f10', (int((self.fb_fret_dict['f10'] + self.fb_fret_dict['f9'])/2), int(self.fb_string_dict['s1']))),
            ('s1f11', (int((self.fb_fret_dict['f11'] + self.fb_fret_dict['f10'])/2), int(self.fb_string_dict['s1']))),
            ('s1f12', (int((self.fb_fret_dict['f12'] + self.fb_fret_dict['f11'])/2), int(self.fb_string_dict['s1']))),
            # String 2
            ('s2f1', (int((self.fb_fret_dict['f1'] + self.fb_fret_dict['f0'])/2), int(self.fb_string_dict['s2']))),
            ('s2f2', (int((self.fb_fret_dict['f2'] + self.fb_fret_dict['f1'])/2), int(self.fb_string_dict['s2']))),
            ('s2f3', (int((self.fb_fret_dict['f3'] + self.fb_fret_dict['f2'])/2), int(self.fb_string_dict['s2']))),
            ('s2f4', (int((self.fb_fret_dict['f4'] + self.fb_fret_dict['f3'])/2), int(self.fb_string_dict['s2']))),
            ('s2f5', (int((self.fb_fret_dict['f5'] + self.fb_fret_dict['f4'])/2), int(self.fb_string_dict['s2']))),
            ('s2f6', (int((self.fb_fret_dict['f6'] + self.fb_fret_dict['f5'])/2), int(self.fb_string_dict['s2']))),
            ('s2f7', (int((self.fb_fret_dict['f7'] + self.fb_fret_dict['f6'])/2), int(self.fb_string_dict['s2']))),
            ('s2f8', (int((self.fb_fret_dict['f8'] + self.fb_fret_dict['f7'])/2), int(self.fb_string_dict['s2']))),
            ('s2f9', (int((self.fb_fret_dict['f9'] + self.fb_fret_dict['f8'])/2), int(self.fb_string_dict['s2']))),
            ('s2f10', (int((self.fb_fret_dict['f10'] + self.fb_fret_dict['f9'])/2), int(self.fb_string_dict['s2']))),
            ('s2f11', (int((self.fb_fret_dict['f11'] + self.fb_fret_dict['f10'])/2), int(self.fb_string_dict['s2']))),
            ('s2f12', (int((self.fb_fret_dict['f12'] + self.fb_fret_dict['f11'])/2), int(self.fb_string_dict['s2']))),
            # String 3
            ('s3f1', (int((self.fb_fret_dict['f1'] + self.fb_fret_dict['f0'])/2), int(self.fb_string_dict['s3']))),
            ('s3f2', (int((self.fb_fret_dict['f2'] + self.fb_fret_dict['f1'])/2), int(self.fb_string_dict['s3']))),
            ('s3f3', (int((self.fb_fret_dict['f3'] + self.fb_fret_dict['f2'])/2), int(self.fb_string_dict['s3']))),
            ('s3f4', (int((self.fb_fret_dict['f4'] + self.fb_fret_dict['f3'])/2), int(self.fb_string_dict['s3']))),
            ('s3f5', (int((self.fb_fret_dict['f5'] + self.fb_fret_dict['f4'])/2), int(self.fb_string_dict['s3']))),
            ('s3f6', (int((self.fb_fret_dict['f6'] + self.fb_fret_dict['f5'])/2), int(self.fb_string_dict['s3']))),
            ('s3f7', (int((self.fb_fret_dict['f7'] + self.fb_fret_dict['f6'])/2), int(self.fb_string_dict['s3']))),
            ('s3f8', (int((self.fb_fret_dict['f8'] + self.fb_fret_dict['f7'])/2), int(self.fb_string_dict['s3']))),
            ('s3f9', (int((self.fb_fret_dict['f9'] + self.fb_fret_dict['f8'])/2), int(self.fb_string_dict['s3']))),
            ('s3f10', (int((self.fb_fret_dict['f10'] + self.fb_fret_dict['f9'])/2), int(self.fb_string_dict['s3']))),
            ('s3f11', (int((self.fb_fret_dict['f11'] + self.fb_fret_dict['f10'])/2), int(self.fb_string_dict['s3']))),
            ('s3f12', (int((self.fb_fret_dict['f12'] + self.fb_fret_dict['f11'])/2), int(self.fb_string_dict['s3']))),
            # String 4
            ('s4f1', (int((self.fb_fret_dict['f1'] + self.fb_fret_dict['f0'])/2), int(self.fb_string_dict['s4']))),
            ('s4f2', (int((self.fb_fret_dict['f2'] + self.fb_fret_dict['f1'])/2), int(self.fb_string_dict['s4']))),
            ('s4f3', (int((self.fb_fret_dict['f3'] + self.fb_fret_dict['f2'])/2), int(self.fb_string_dict['s4']))),
            ('s4f4', (int((self.fb_fret_dict['f4'] + self.fb_fret_dict['f3'])/2), int(self.fb_string_dict['s4']))),
            ('s4f5', (int((self.fb_fret_dict['f5'] + self.fb_fret_dict['f4'])/2), int(self.fb_string_dict['s4']))),
            ('s4f6', (int((self.fb_fret_dict['f6'] + self.fb_fret_dict['f5'])/2), int(self.fb_string_dict['s4']))),
            ('s4f7', (int((self.fb_fret_dict['f7'] + self.fb_fret_dict['f6'])/2), int(self.fb_string_dict['s4']))),
            ('s4f8', (int((self.fb_fret_dict['f8'] + self.fb_fret_dict['f7'])/2), int(self.fb_string_dict['s4']))),
            ('s4f9', (int((self.fb_fret_dict['f9'] + self.fb_fret_dict['f8'])/2), int(self.fb_string_dict['s4']))),
            ('s4f10', (int((self.fb_fret_dict['f10'] + self.fb_fret_dict['f9'])/2), int(self.fb_string_dict['s4']))),
            ('s4f11', (int((self.fb_fret_dict['f11'] + self.fb_fret_dict['f10'])/2), int(self.fb_string_dict['s4']))),
            ('s4f12', (int((self.fb_fret_dict['f12'] + self.fb_fret_dict['f11'])/2), int(self.fb_string_dict['s4']))),
            # String 5
            ('s5f1', (int((self.fb_fret_dict['f1'] + self.fb_fret_dict['f0'])/2), int(self.fb_string_dict['s5']))),
            ('s5f2', (int((self.fb_fret_dict['f2'] + self.fb_fret_dict['f1'])/2), int(self.fb_string_dict['s5']))),
            ('s5f3', (int((self.fb_fret_dict['f3'] + self.fb_fret_dict['f2'])/2), int(self.fb_string_dict['s5']))),
            ('s5f4', (int((self.fb_fret_dict['f4'] + self.fb_fret_dict['f3'])/2), int(self.fb_string_dict['s5']))),
            ('s5f5', (int((self.fb_fret_dict['f5'] + self.fb_fret_dict['f4'])/2), int(self.fb_string_dict['s5']))),
            ('s5f6', (int((self.fb_fret_dict['f6'] + self.fb_fret_dict['f5'])/2), int(self.fb_string_dict['s5']))),
            ('s5f7', (int((self.fb_fret_dict['f7'] + self.fb_fret_dict['f6'])/2), int(self.fb_string_dict['s5']))),
            ('s5f8', (int((self.fb_fret_dict['f8'] + self.fb_fret_dict['f7'])/2), int(self.fb_string_dict['s5']))),
            ('s5f9', (int((self.fb_fret_dict['f9'] + self.fb_fret_dict['f8'])/2), int(self.fb_string_dict['s5']))),
            ('s5f10', (int((self.fb_fret_dict['f10'] + self.fb_fret_dict['f9'])/2), int(self.fb_string_dict['s5']))),
            ('s5f11', (int((self.fb_fret_dict['f11'] + self.fb_fret_dict['f10'])/2), int(self.fb_string_dict['s5']))),
            ('s5f12', (int((self.fb_fret_dict['f12'] + self.fb_fret_dict['f11'])/2), int(self.fb_string_dict['s5']))),
            # String 6
            ('s6f1', (int((self.fb_fret_dict['f1'] + self.fb_fret_dict['f0'])/2), int(self.fb_string_dict['s6']))),
            ('s6f2', (int((self.fb_fret_dict['f2'] + self.fb_fret_dict['f1'])/2), int(self.fb_string_dict['s6']))),
            ('s6f3', (int((self.fb_fret_dict['f3'] + self.fb_fret_dict['f2'])/2), int(self.fb_string_dict['s6']))),
            ('s6f4', (int((self.fb_fret_dict['f4'] + self.fb_fret_dict['f3'])/2), int(self.fb_string_dict['s6']))),
            ('s6f5', (int((self.fb_fret_dict['f5'] + self.fb_fret_dict['f4'])/2), int(self.fb_string_dict['s6']))),
            ('s6f6', (int((self.fb_fret_dict['f6'] + self.fb_fret_dict['f5'])/2), int(self.fb_string_dict['s6']))),
            ('s6f7', (int((self.fb_fret_dict['f7'] + self.fb_fret_dict['f6'])/2), int(self.fb_string_dict['s6']))),
            ('s6f8', (int((self.fb_fret_dict['f8'] + self.fb_fret_dict['f7'])/2), int(self.fb_string_dict['s6']))),
            ('s6f9', (int((self.fb_fret_dict['f9'] + self.fb_fret_dict['f8'])/2), int(self.fb_string_dict['s6']))),
            ('s6f10', (int((self.fb_fret_dict['f10'] + self.fb_fret_dict['f9'])/2), int(self.fb_string_dict['s6']))),
            ('s6f11', (int((self.fb_fret_dict['f11'] + self.fb_fret_dict['f10'])/2), int(self.fb_string_dict['s6']))),
            ('s6f12', (int((self.fb_fret_dict['f12'] + self.fb_fret_dict['f11'])/2), int(self.fb_string_dict['s6']))),
            ])
        logging.debug('fb_string_fret_dict {}'.format(self.fb_string_fret_dict))
         

    def get_fretboard_overlay_basic(self):
        # img_overlay = img_dst.copy()
        # 4th channel is alpha
        # fb_scale = 2.0
        img_overlay = np.zeros((int(self.fb_h*self.fb_scale), int(self.fb_w*self.fb_scale), 4), np.uint8)
        # self.fb_scale = 1.0
        img_overlay[:,:,0:3] = 255
        img_overlay[:,:,3] = 3 # Alpha channel values: [0:10]

        ### FretBoard
        x, y, w, h = int(0*self.fb_scale), int(0*self.fb_scale), int(self.fb_w*self.fb_scale), int(self.fb_h*self.fb_scale)
        cv2.rectangle(img_overlay, (x, y), (x+w, y+h), (0, 200, 0), 0)  # A filled rectangle


        ### Strings
        for string_name, string_height in self.fb_string_dict.items():
            x1, y1, x2, y2 = int(0*self.fb_scale), int(string_height*self.fb_scale), int(self.fb_w*self.fb_scale), int(string_height*self.fb_scale)
            cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 1)

        for fret_name, fret_width in self.fb_fret_dict.items():
            x1, y1, x2, y2 = int(fret_width*self.fb_scale), int(0*self.fb_scale), int(fret_width*self.fb_scale), int(self.fb_h*self.fb_scale)
            cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 1)

        # cv2.flip(img_overlay, 1, img_overlay)

        return img_overlay


    def get_fretboard_overlay(self):
        # img_overlay = img_dst.copy()
        # 4th channel is alpha
        # fb_scale = 2.0
        img_overlay = np.zeros((int(self.fb_h*self.fb_scale), int(self.fb_w*self.fb_scale), 4), np.uint8)
        # self.fb_scale = 1.0
        img_overlay[:,:,0:3] = 255
        img_overlay[:,:,3] = 7 # Alpha channel values: [0:10]

        ### FretBoard
        x, y, w, h = int(0*self.fb_scale), int(0*self.fb_scale), int(self.fb_w*self.fb_scale), int(self.fb_h*self.fb_scale)
        logging.debug('x,y,w,h: {} {} {} {}'.format(x, y, w, h))
        cv2.rectangle(img_overlay, (x, y), (x+w, y+h), (0, 200, 0), 0)  # A filled rectangle
        # alpha = 0.4
        # img_overlay = cv2.addWeighted(img_overlay, alpha, img_overlay, 1 - alpha, 0)
        # img_overlay = cv2.addWeighted(src1=overlay, alpha=1, src2=img_dst, beta=0.5, gamma=0)

        ### Strings
        for string_name, string_height in self.fb_string_dict.items():
            x1, y1, x2, y2 = int(0*self.fb_scale), int(string_height*self.fb_scale), int(self.fb_w*self.fb_scale), int(string_height*self.fb_scale)
            cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 1)

        for fret_name, fret_width in self.fb_fret_dict.items():
            x1, y1, x2, y2 = int(fret_width*self.fb_scale), int(0*self.fb_scale), int(fret_width*self.fb_scale), int(self.fb_h*self.fb_scale)
            cv2.line(img_overlay, (x1, y1), (x2, y2), (0,255,0), 1)

        # for string_fret_name, string_pos in self.fb_string_fret_dict.items():
        #     logging.debug('string_fret_name {}'.format(string_fret_name))
        #     logging.debug('string_pos {}'.format(string_pos))
        #     cv2.circle(img_overlay, center=(string_pos), radius=3, color=(255, 0, 0), thickness=1)


        ## Flip around axis
        # cv2.flip(img_overlay, 1, img_overlay)


        return img_overlay

    
    def add_fretboard_overlay_notes(self, img_overlay, notes):
        logging.debug('notes {}'.format(notes))
        notes_pos = []
        for note in notes:
            logging.debug('note {}'.format(note))
            # logging.debug('string_fret_name {}'.format(string_fret_name))
            # logging.debug('string_pos {}'.format(string_pos))
            if note is not None:
                string_pos = self.fb_string_fret_dict[note[0]]
                cv2.circle(img_overlay, center=(string_pos), radius=3, color=(255, 255, 0), thickness=2)

                # TODO: Flipping notes for display
                notes_pos.append([float(self.fb_w - string_pos[0]), float(string_pos[1])])

            else:
                notes_pos.append(None)

        return notes_pos


    def update_fretboard_overlay_got(self, img_overlay, progression_time):

        if progression_time < len(self.progression) and self.progression[progression_time] is not None:
            notes_pos = self.add_fretboard_overlay_notes(img_overlay, notes=self.progression[progression_time])
            return notes_pos
        else:
            return None

        # return img_overlay


    def tab_got(self):
        self.progression = [
                [['s1f1']],
                [['s2f1']],
                None,
                [['s4f5']],
                None,
                [['s5f3']],
                None,
                [['s4f6']],
                None,
                [['s4f5']],
                ]


if __name__ == '__main__':

    # https://tlcguitargoods.com/en/howto-fret-calculator
    output_dir = 'output'
    
    fb = FretBoard(output_dir)
    fb_overlay = fb.get_fretboard_overlay()

    progression_time = 1
    fb.update_fretboard_overlay_got(fb_overlay, progression_time)
    cv2.flip(fb_overlay, 1, fb_overlay)

    img_dst = cv2.imread('data/guitar/test/2019-05-28-085835_1.jpg')
    fb.overlay_image_alpha(img_dst,
                    fb_overlay[:, :, 0:3],
                    (0, 0),
                    fb_overlay[:, :, 3]/10)
    cv2.imwrite(os.path.join(output_dir, 'img_overlay_dst.jpg'), img_dst)


