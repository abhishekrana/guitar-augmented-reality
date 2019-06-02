import os
import pudb
import glob
import logging
import skimage.io as io
import cv2
import numpy as np
import time
import pickle

# from keras import backend as K
# from keras.backend.tensorflow_backend import set_session
# import tensorflow as tf
# import keras

from utils import logger_init
# from model import *
# from data import *
from fretboard import FretBoard
# from train import testing_load_model, testing_predict, testing_predict2
# from fit_rectangle import find_corners
from homography import get_warped_image

DEBUG_FLAG = False

if __name__ == '__main__':

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    # fileHandler, consoleHandler = logger_init(output_dir, logging.DEBUG)
    fileHandler, consoleHandler = logger_init(output_dir, logging.INFO)
    class_name = ''
    pred_corners_file = os.path.join(output_dir, 'pred_corners.pkl')
    pred_image_file = os.path.join(output_dir, 'pred_image.jpg')

    # test_data_dir = 'data/guitar/dataset_frames1_val/'
    # test_data_dir = 'data/guitar/dataset_frames1_val_aug_v3/'
    test_data_dir = 'data/guitar/test_3/'
    # test_images_list = glob.glob(os.path.join(test_data_dir, class_name, 'image', '*' + '.jpg'))
    test_images_list = glob.glob(os.path.join(test_data_dir, '*' + '.jpg'))
    # test_images_list = test_images_list[0:50]
    test_images_list = test_images_list[0:8]
    logging.debug('test_images_list {}'.format(test_images_list))

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('data/guitar/videos/2019-06-02-225112.webm')
     
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
      print("Unable to read camera feed")
     
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print('frame_width', frame_width)
    print('frame_height', frame_height)

    # frame_width = int(640)
    # frame_height = int(640)
     
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(os.path.join(output_dir, 'outpy.avi'), cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    print('out', out)
 
    # testing_model = testing_load_model(test_images_list)

    frames_count = 0
    pred_count = 0
    # corners_list = [[[0,0], [0,0], [0,0], [0,0]]]
    corners_list = None

    mask_path_name = 'output/test/pred_image_predict.png'
    pred_image = None

    while(True):
        ret, frame = cap.read()

        if ret == True: 
            frames_count += 1
            print('pred/frames: [{}/{}]'.format(pred_count, frames_count))

            # time.sleep(0.05)
            time.sleep(0.01)


            if not os.path.isfile(pred_image_file):
                # Swap channels
                io.imsave(pred_image_file, (frame[...,::-1]).astype(np.uint8))
                print('Saving image pred_image_file {}'.format(pred_image_file))


            if os.path.isfile(pred_corners_file):
                try:
                    with open(pred_corners_file, 'rb') as f:
                        corners_list = pickle.load(f)
                        # logging.debug('corners_list {}'.format(corners_list))
                except:
                    logging.error('Exception')
                    os.remove(pred_corners_file)
                    os.remove(pred_image_file)
                    continue

                if os.path.exists(mask_path_name):
                    pred_image = cv2.imread(mask_path_name)
                os.remove(pred_corners_file)
                os.remove(pred_image_file)
                pred_count += 1


            if corners_list is not None:

                fb = FretBoard(output_dir)

                im_dst_name = 'final_out.jpg'

                # im_dst = cv2.imread(im_dst_path)
                # TODO: swap channels
                im_dst = frame

                template_coords = corners_list

                ### Template Wrap ###
                im_template = fb.get_fretboard_overlay()
                im_template_basic = fb.get_fretboard_overlay_basic()

                # Display fretboard on top-left
                notes_pos = fb.update_fretboard_overlay_got(im_template, progression_time=frames_count)
                if notes_pos is not None:
                    for ps in notes_pos:
                        if ps is not None:
                            cv2.circle(im_dst, center=(int(ps[0]), int(ps[1])), radius=2, color=(0, 255, 0), thickness=5)
                cv2.flip(im_template, 1, im_template)
                fb.overlay_image_alpha(im_dst, im_template[:, :, 0:3], (0, 0), im_template[:, :, 3]/10)

                # notes_pos already flipped
                notes_pos_basic = fb.update_fretboard_overlay_got(im_template_basic, progression_time=frames_count)
                cv2.flip(im_template_basic, 1, im_template_basic)

                
                #TODO: Check coordinate order for correctness
                im_warp = get_warped_image(im_template_basic, im_dst, template_coords, notes_pos_basic)
                if im_warp is not None:
                    fb.overlay_image_alpha(im_dst, im_warp[:, :, 0:3], (0, 0), im_warp[:, :, 3]/10)

                    if DEBUG_FLAG:
                        cv2.imwrite(os.path.join(output_dir, im_dst_name + '_' + str(frames_count) + '_overlay.jpg'), im_dst)






            # Write the frame into the file 'output.avi'
            # out.write(frame)


            
            # Display the resulting frame    
            cv2.imshow('frame',frame)

            if pred_image is not None:
                cv2.imshow('prediction', pred_image)

            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break 


    # When everything done, release the video capture and video write objects
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows() 



    # ### Cornors ###
    # output_dir_contour = os.path.join(output_dir, 'contours')
    # os.makedirs(output_dir_contour, exist_ok=True)
    # corners_list = []
    # for pred_idx, pred_mask in enumerate(pred_masks):
    #     image_name = os.path.basename(test_images_list[pred_idx]).split('.')[0]
    #     corners_ret = find_corners(output_dir_contour, np.squeeze((pred_mask*255).astype(np.uint8)), image_name)
    #     # TODO: Handle < 4 corners
    #     if corners_ret is not None:
    #         corners = corners_ret
    #         # corners = [corners_ret[0][0], corners_ret[1][0], corners_ret[2][0], corners_ret[3][0]]
    #         corners_list.append(corners)
    #         logging.debug('corners {}'.format(corners))
    #     else:
    #         # TODO: hardcoding 
    #         logging.error('corners_ret {}'.format(corners_ret))
    #         corners = [[0,0], [0,0], [0,0], [0,0]]
    #         corners_list.append(corners)
    
    # if len(corners_list) == 0:
    #     # continue
    #     exit(0)


    ## corners_list = np.array([
    ##    [[567, 382]],
    ##    [[260, 412]],
    ##    [[262, 473]],
    ##    [[567, 428]]], dtype=int32)

    #corners_list = [
    #        # [[437, 154], [1014, 90], [1025, 136], [435, 216]],
    #        [[522, 463], [1130, 429], [1133, 482], [527, 535]],
    #        ]
 
    #fb = FretBoard(output_dir)


    ### TODO: Don't read again
    ##for idx, im_dst_path in enumerate(test_images_list):
    ##    im_dst_name = os.path.basename(im_dst_path)
    ##    im_dst = cv2.imread(im_dst_path)

    ##    try:
    ##        template_coords = corners_list[idx]
    ##    except:
    ##        logging.error('template_coords {}'.format(template_coords))
    ##        continue

    ##    #TODO: Check coordinate order for correctness
    ##    im_warp = get_warped_image(im_template, im_dst, template_coords)
    ##    if im_warp is not None:
    ##        fb.overlay_image_alpha(im_dst, im_warp[:, :, 0:3], (0, 0), im_warp[:, :, 3]/10)
    ##        cv2.imwrite(os.path.join(output_dir, im_dst_name + '_overlay_jpg'), im_dst)


    #num_frames = 50
    #test_images_list = num_frames*[test_images_list]
    #corners_list = num_frames*corners_list

    #for idx, im_dst_path in enumerate(test_images_list):
    #    im_dst_name = os.path.basename(im_dst_path).split('.')[0]
    #    im_dst = cv2.imread(im_dst_path)

    #    try:
    #        template_coords = corners_list[idx]
    #    except:
    #        logging.error('template_coords {}'.format(template_coords))
    #        continue

    #    ### Template Wrap ###
    #    im_template = fb.get_fretboard_overlay()
    #    im_template_basic = fb.get_fretboard_overlay_basic()

    #    # Display fretboard on top-left
    #    notes_pos = fb.update_fretboard_overlay_got(im_template, progression_time=idx)
    #    if notes_pos is not None:
    #        for ps in notes_pos:
    #            if ps is not None:
    #                cv2.circle(im_dst, center=(int(ps[0]), int(ps[1])), radius=2, color=(0, 255, 0), thickness=5)
    #    cv2.flip(im_template, 1, im_template)
    #    fb.overlay_image_alpha(im_dst, im_template[:, :, 0:3], (0, 0), im_template[:, :, 3]/10)

    #    # notes_pos already flipped
    #    notes_pos_basic = fb.update_fretboard_overlay_got(im_template_basic, progression_time=idx)
    #    cv2.flip(im_template_basic, 1, im_template_basic)

        
    #    #TODO: Check coordinate order for correctness
    #    im_warp = get_warped_image(im_template_basic, im_dst, template_coords, notes_pos_basic)
    #    if im_warp is not None:
    #        fb.overlay_image_alpha(im_dst, im_warp[:, :, 0:3], (0, 0), im_warp[:, :, 3]/10)
    #        cv2.imwrite(os.path.join(output_dir, im_dst_name + '_' + str(idx) + '_overlay.jpg'), im_dst)



