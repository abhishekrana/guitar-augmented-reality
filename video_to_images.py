import cv2
import os
import pudb
import logging

from utils import logger_init

if __name__ == '__main__':

    output_dir = 'output'
    output_dir_frames = os.path.join(output_dir, 'frames')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_frames, exist_ok=True)

    filehandler, consolehandler = logger_init(output_dir, logging.DEBUG)

    # video_path = 'assets/2019-05-28-081643.webm'
    # video_path = 'assets/2019-05-28-081812.webm'
    # video_path = 'assets/2019-05-28-081845.webm'
    video_path = 'assets/2019-05-28-085718.webm'
    video_name = os.path.basename(video_path).split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
        logging.debug('count {}'.format(count))
        if count%10 == 0:
            cv2.imwrite(os.path.join(output_dir_frames, '{}_frame{}.jpg'.format(video_name, count)), image)
        success,image = vidcap.read()
        logging.debug('Read a new frame: {}'.format(success))
        count += 1
