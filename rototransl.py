import cv2 as cv
import matplotlib.pyplot as plt
import glob
import re

# collect all frames
imgs=sorted(glob.glob('/Users/noemi/Documents/Uni/TESI/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])


### test ###
# showing the first frame
im = cv.imread(imgs[0])[450:2150,800:2250]
plt.figure(figsize=(10,10))
plt.imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB))
plt.show()


### set up translation parameters ###
tx = 100   # x-axes translation in pixel
ty = 50   # y-axes translation in pixel


### set up rotation parameters ###
center = (im.shape[1]/2, im.shape[0]/2)
angle = 30   # deg
scale = 1


### construct the transformation matrix ###
# construct the rotation matrix
# applying the negative of the angle to rotate clockwise
M = cv.getRotationMatrix2D(center, -angle, scale)
# combine with translation
M[0, 2] += tx
M[1, 2] += ty


### perform transformation on all frames ###
for i in range(len(imgs)):
    temp = cv.imread(imgs[i])[450:2150,800:2250]
    temp_rot = cv.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))
    cv.imwrite('frames_rt/frame_'+str(12760+i)+'_rt.jpg', temp_rot)
    # test: showing the first frame
    if i==0:
       plt.figure(figsize=(10,10))
       plt.imshow(cv.cvtColor(temp_rot, cv.COLOR_BGR2RGB))
       plt.show() 
       