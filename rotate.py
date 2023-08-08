import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

resize = False

# collect all frames
imgs=sorted(glob.glob('/Users/noemi/Documents/Uni/TESI/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])


### test ###
# showing the first frame
im = cv.imread(imgs[0])[450:2150,800:2250]
plt.figure(figsize=(10,10))
plt.imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB))
plt.show()
width = im.shape[1]
height = im.shape[0]


### set up rotation parameters ###
center = (im.shape[1]/2, im.shape[0]/2)
angle = 30   # deg
scale = 1


### construct the rotation matrix ###
# applying the negative of the angle to rotate clockwise
M = cv.getRotationMatrix2D(center, -angle, scale)


### fit the whole image ###
if resize:
    # compute new bounding dimensions
    alpha = np.abs(M[0, 0])   # scale * cos(angle)
    beta = np.abs(M[0, 1])   # scale * sin(angle)
    width = int(im.shape[0] * beta + im.shape[1] * alpha)
    height = int(im.shape[0] * alpha + im.shape[1] * beta)
    # compute center translation to adjust the rotation matrix
    # it works only if the centre of rotation is the centre of the image
    M[0, 2] += (width / 2) - im.shape[1]/2
    M[1, 2] += (height / 2) - im.shape[0]/2


### perform rotation on all frames ###
for i in range(len(imgs)):
    temp = cv.imread(imgs[i])[450:2150,800:2250]
    temp_rot = cv.warpAffine(temp, M, (width, height))
    cv.imwrite('frames_rot'+str(angle)+'/frame_'+str(12760+i)+'_rot'+str(angle)+'.jpg', temp_rot)
    # test: showing the first frame
    if i==0:
       plt.figure(figsize=(10,10))
       plt.imshow(cv.cvtColor(temp_rot, cv.COLOR_BGR2RGB))
       plt.show() 
       