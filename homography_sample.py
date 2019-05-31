import cv2
import numpy as np
import pudb

"""
OpenCV: 
    Origin: top left
    X-axis: left to right
    Y-axis: up to down

    cv2.circle(im_src, center, radius, color[, thickness[, lineType[, shift]]])

"""
 
if __name__ == '__main__' :
    # pu.db
 
    # Read source image.
    im_src = cv2.imread('temp/book1.jpg')
    # Four corners of the book in source image
    # pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])
    # pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])

    cv2.circle(im_src, center=(153, 220), radius=1, color=(0,255,0), thickness=2)
    cv2.circle(im_src, center=(340, 236), radius=1, color=(0,255,0), thickness=2)
    cv2.circle(im_src, center=(340, 372), radius=1, color=(0,255,0), thickness=2)
    cv2.circle(im_src, center=(135, 357), radius=1, color=(0,255,0), thickness=2)
    pts_src = np.array([[153, 220], [340, 236], [340, 372],[135, 357]])
    cv2.imshow("Source Image", im_src)

    im_dst = cv2.imread('temp/book2.jpg')
    cv2.circle(im_dst, center=(216, 250), radius=1, color=(0,255,0), thickness=2)
    cv2.circle(im_dst, center=(333, 322), radius=1, color=(0,255,0), thickness=2)
    cv2.circle(im_dst, center=(268, 404), radius=1, color=(0,255,0), thickness=2)
    cv2.circle(im_dst, center=(145, 317), radius=1, color=(0,255,0), thickness=2)
    pts_dst = np.array([[216, 250], [333, 322], [268, 404],[145, 317]])
    cv2.imshow("Destination Image", im_dst)
 
 
    # # Read destination image.
    # im_dst = cv2.imread('temp/book1.jpg')
    # # Four corners of the book in destination image.
    # pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])
 
    # # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    print('h', h)
    print('status', status)
     
    # # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
     
    # # Display images
    # cv2.imshow("Source Image", im_src)
    # cv2.imshow("Destination Image", im_dst)
    # cv2.imshow("Warped Source Image", im_out)
 
    im_overlay = cv2.addWeighted(src1=im_out, alpha=1, src2=im_dst, beta=1, gamma=0)
    cv2.imshow("Overlay Image", im_overlay)
 
    cv2.waitKey(0)
