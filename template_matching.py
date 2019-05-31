"""
Limitations of Template Matching:
    Pattern occurrences have to preserve the orientation of the reference pattern image(template)
    As a result, it does not work for rotated or scaled versions of the template as a change in shape/size/shear etc. of object w.r.t. template will give a false match.
    The method is inefficient when calculating the pattern correlation image for medium to large images as the process is time consuming.

https://www.geeksforgeeks.org/template-matching-using-opencv-in-python/
"""
import os
import cv2 
import numpy as np 
import pudb
  
output_dir = 'output'

# Read the main image 
img_rgb = cv2.imread(os.path.join(output_dir, 'img_dst.jpg')) 
  
# Convert it to grayscale 
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) 
  
# Read the template 
# template = cv2.imread('template',0) 
# img_template = cv2.imread(os.path.join(output_dir, 'img_template_warp.jpg'), 0) 
img_template = cv2.imread(os.path.join(output_dir, 'img_template_warp_1.jpg'), 0) 
# img_template = cv2.imread(os.path.join(output_dir, 't1.jpg'), 0) 
  
# Store width and heigth of template in w and h 
w, h = img_template.shape[::-1] 
  
# Perform match operations. 
res = cv2.matchTemplate(img_gray,img_template,cv2.TM_CCOEFF_NORMED) 
print('res', res)
  
# Specify a threshold 
# threshold = 0.9
threshold = 0.2
  
# Store the coordinates of matched area in a numpy array 
loc = np.where( res >= threshold)  
print('loc', loc)
  
# Draw a rectangle around the matched region. 
for pt in zip(*loc[::-1]): 
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
  
# Show the final image with the matched area. 
# cv2.imshow('Detected',img_rgb) 
cv2.imwrite(os.path.join(output_dir, 'img_template_warp_match.jpg'), img_rgb)

