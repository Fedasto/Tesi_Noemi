import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
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


### check all targets are still inside the image boundaries ###
if tx > 173 or tx < -274 or ty > 265 or ty < -211:
    print('Target out of boundaries!')
    sys.exit()


### construct the transformation matrix ###
# applying the negative of the angle to rotate clockwise
M = np.array([[1, 0, tx], [0, 1, ty]], np.float32)


### perform translation on all frames ###
for i in range(len(imgs)):
    temp = cv.imread(imgs[i])[450:2150,800:2250]
    temp_transl = cv.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))
    cv.imwrite('frames_transl/frame_'+str(12760+i)+'_transl.jpg', temp_transl)
    # test: showing the first frame
    if i==0:
       plt.figure(figsize=(10,10))
       plt.imshow(cv.cvtColor(temp_transl, cv.COLOR_BGR2RGB))
       plt.show() 
       