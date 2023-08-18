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


### set up rotation parameters ###
center = (im.shape[1]/2, im.shape[0]/2)
angle = 0.2   # deg
scale = 1


### perform rotation on all frames ###
im_rot = []

# first frame is unaltered
im_rot.append(im)
cv.imwrite('frames_rot/frame_'+str(12760)+'_rot.jpg', im)

for i in range(1,len(imgs)):
    # construct the rotation matrix
    # applying the negative of the angle to rotate clockwise
    R = cv.getRotationMatrix2D(center, -angle*i, scale)
    M = np.concatenate( (R, np.array([[0,0,1]])) )
    # perform rotation using the previous frame
    im_rot.append(cv.warpPerspective(im, M, (width, height)))
    cv.imwrite('frames_rot/frame_'+str(12760+i)+'_rot.jpg', im_rot[i])
    # show frame
    plt.figure(figsize=(10,10))
    plt.imshow(cv.cvtColor(im_rot[i], cv.COLOR_BGR2RGB))
    plt.show()