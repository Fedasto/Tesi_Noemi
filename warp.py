import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


# collect all frames
imgs=sorted(glob.glob('/Users/noemi/Documents/Uni/TESI/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])


### get first frame and frame dimensions ###
im = cv.imread(imgs[0])[450:2150,700:2350]
width = im.shape[1]
height = im.shape[0]


### set starting points ###
# points from undistorted image, in pixels
start_pt = np.array([[0, 0],
                     [width, 0],
                     [0, height],
                     [width, height]], np.float32)
    
       
### perform perspective transformation on all frames ###
im_rot = []

# first frame is unaltered
im_rot.append(im)
cv.imwrite('frames_warp/frame_'+str(12760)+'_warp.jpg', im)

for i in range(1,len(imgs)):
    # new points, in pixels
    end_pt = np.array([[0+i, 0+i],
                       [width-2*i, 0+2*i],
                       [0+2*i, height-2*i],
                       [width-i, height-i]], np.float32)
    # construct perspective transformation matrix
    P = cv.getPerspectiveTransform(start_pt, end_pt)
    # perform transformation
    im_rot.append(cv.warpPerspective(im, P, (width, height)))
    cv.imwrite('frames_warp/frame_'+str(12760+i)+'_warp.jpg', im_rot[i])
    # show frame
    plt.figure(figsize=(10,10))
    plt.imshow(cv.cvtColor(im_rot[i], cv.COLOR_BGR2RGB))
    plt.show()
       