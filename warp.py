import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

affine = True
perspective = False

# collect all frames
imgs=sorted(glob.glob('/Users/noemi/Documents/Uni/TESI/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])


### test ###
# showing the first frame
im = cv.imread(imgs[0])[450:2150,800:2250]
plt.figure(figsize=(10,10))
plt.imshow(cv.cvtColor(im,cv.COLOR_BGR2RGB))
plt.show()


if affine:
### set up affine transformation matrix ###
    # M matrix will act as follows
    # M = [[a, b, c], [d, e, f]]
    # new_x = a*x + b*y + c
    # new_y = d*x + e*y + f
    M = np.array([[0.8, -0.4, 500],
                  [1, 0.6, -400]], dtype='double')
    
    
### perform affine transformation on all frames ###
    for i in range(len(imgs)):
        temp = cv.imread(imgs[i])[450:2150,800:2250]
        temp_warp = cv.warpAffine(temp, M, (temp.shape[1], temp.shape[0]))
        cv.imwrite('frames_warp/frame_'+str(12760+i)+'_warp.jpg', temp_warp)
        # test: showing the first frame
        if i==0:
           plt.figure(figsize=(10,10))
           plt.imshow(cv.cvtColor(temp_warp, cv.COLOR_BGR2RGB))
           plt.show()


if perspective:
### set up perspective transformation matrix ###
    # P matrix will act as follows
    # P = [[a, b, c], [d, e, f], [g, h, i]]
    # new_x*t = a*x + b*y + c
    # new_y*t = d*x + e*y + f
    # t = g*x + h*y + i
    P = np.array([[0.8, -0.4, 500],
                  [1, 0.6, -400],
                  [0, 0, 1]], dtype = 'double')
    
           
### perform perspective transformation on all frames ###
    for i in range(len(imgs)):
        temp = cv.imread(imgs[i])[450:2150,800:2250]
        temp_warp = cv.warpPerspective(temp, P, (temp.shape[1], temp.shape[0]))
        cv.imwrite('frames_warp/frame_'+str(12760+i)+'_warp.jpg', temp_warp)
        # test: showing the first frame
        if i==0:
           plt.figure(figsize=(10,10))
           plt.imshow(cv.cvtColor(temp_warp, cv.COLOR_BGR2RGB))
           plt.show() 
       