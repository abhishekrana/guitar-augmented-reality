import os
import cv2
import numpy as np
import pudb
from fretboard import  overlay_image_alpha, get_fretborad
import logging

"""
OpenCV: 
    Origin: top left
    X-axis: left to right
    Y-axis: up to down

    cv2.circle(im_src, center, radius, color[, thickness[, lineType[, shift]]])

"""

# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_warped_image(im_template, im_dst, template_coords):
    output_dir = 'output'

 
    # Read source image.
    # im_template = cv2.imread('data/template/template_1.png')
    im_template = get_fretborad()
    h_template, w_template, c_template = im_template.shape
    pts_src = np.array([[0, 0], [w_template, 0], [w_template, h_template],[0, h_template]])
    for ps in pts_src:
        cv2.circle(im_template, center=(ps[0], ps[1]), radius=1, color=(0,255,0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, 'img_template.jpg'), im_template)
    # cv2.imshow("Source Image", im_template)


    # im_dst = cv2.imread('data/guitar/test/2019-05-28-085835_1.jpg')
    # pts_dst = np.array([[427, 157], [1014, 90], [1025, 136], [420, 218]])
    # pts_dst = np.array([[437, 154], [1014, 90], [1025, 136], [435, 216]])
    pts_dst = np.array(template_coords)
    for ps in pts_dst:
        cv2.circle(im_dst, center=(ps[0], ps[1]), radius=1, color=(0,255,0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, 'img_dst.jpg'), im_dst)
    # cv2.imshow("Destination Image", im_dst)

    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    print('h', h)
    print('status', status)
     
    # Warp source image to destination based on homography
    try:
        im_warp = cv2.warpPerspective(im_template, h, (im_dst.shape[1],im_dst.shape[0]))
        cv2.imwrite(os.path.join(output_dir, 'img_template_warp.jpg'), im_warp)
    except:
        logging.error('im_template {}'.format(im_template))
        logging.error('h {}'.format(h))
        logging.error('im_dst {}'.format(im_dst.shape))
        return None
        # logging.error('im_warp {}'.format(im_warp))
    # cv2.imshow("Warped Source Image", im_warp)
     

    # Setting alpha=1, beta=1, gamma=0 gives direct overlay of two images
    # im_overlay = cv2.addWeighted(src1=im_template, alpha=1, src2=im_dst, beta=1, gamma=0, dst=im_dst)
    # im_overlay = cv2.addWeighted(src1=im_warp, alpha=1, src2=im_dst, beta=0.5, gamma=0, dst=im_dst)

    return im_warp




if __name__ == '__main__' :

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    im_template = get_fretborad()
    im_dsts = [
            'data/guitar/test/2019-05-28-085835_1.jpg',
            'data/guitar/test/2019-05-28-085835_2.jpg',
            'data/guitar/test/2019-05-28-085835_5.jpg',
            'data/guitar/test/2019-05-28-085835_6.jpg',
            'data/guitar/test/final.jpg',
            ]

    template_coords_list = [
            [[437, 154], [1014, 90], [1025, 136], [435, 216]],
            [[448, 162], [1051, 170], [1057, 226], [441, 230]],
            [[444, 151], [1014, 62], [1029, 103], [440, 209]],
            [[438, 149], [1040, 104], [1050, 157], [438, 220]],
            [[78, 364], [1228, 362], [1234, 484], [69, 490]]
            ]
            # [[1057, 226], [1051, 170], [448, 162], [441, 230]],

    for idx, im_dst_path in enumerate(im_dsts):
        im_dst_name = os.path.basename(im_dst_path)
        im_dst = cv2.imread(im_dst_path)

        template_coords = template_coords_list[idx]
        im_warp = get_warped_image(im_template, im_dst, template_coords)
        overlay_image_alpha(im_dst, im_warp[:, :, 0:3], (0, 0), im_warp[:, :, 3]/10)
        cv2.imwrite(os.path.join(output_dir, im_dst_name + '_overlay_jpg'), im_dst)



    # cv2.imwrite(os.path.join(output_dir, 'img_overlay.jpg'), im_overlay)
    # cv2.imshow("Overlay Image", im_overlay)
 

    # overlay = image.copy()
    # output = image.copy()
    # cv2.rectangle(overlay, (420, 205), (595, 385),
    #         (0, 0, 255), -1)


    # cv2.waitKey(0)

