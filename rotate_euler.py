import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import re


### FUNCTIONS ##################################################################################################

### transforms euler angles into a rotation matrix ###
def euler2matrix(angle) :
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angle[0]), -np.sin(angle[0])],
                    [0, np.sin(angle[0]), np.cos(angle[0])]])
    Ry = np.array([[np.cos(angle[1]), 0, np.sin(angle[1])],
                    [0, 1, 0],
                    [-np.sin(angle[1]), 0, np.cos(angle[1])]])
    Rz = np.array([[np.cos(angle[2]), -np.sin(angle[2]), 0],
                    [np.sin(angle[2]), np.cos(angle[2]), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    return R

### transforms a rotation matrix into the corrisponding euler angles ###
def matrix2euler(R):
    oz = np.arctan2(R[1][0], R[0][0])
    oy = np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2))
    ox = np.arctan2(R[2][1], R[2][2])
    return (ox, oy, oz)

################################################################################################################

# collect all frames
imgs=sorted(glob.glob('/Users/noemi/Documents/Uni/TESI/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])


### get first frame and frame dimensions ###
im = cv.imread(imgs[0])[450:2150,700:2350]
width = im.shape[1]
height = im.shape[0]


### set up rotation parameters ###
roll = 0   # rad
pitch = 0   # rad
yaw = np.pi/(96*16)   # rad
euler = np.array([roll, pitch, yaw], np.float32)


### test: compute corresponding rvec ###
M = euler2matrix(euler)
rvec = cv.Rodrigues(M)[0]
print('Test\nrvec corresponding to the first rotation:\n', rvec)


### perform rotation on all frames ###
im_rot = []

# first frame is unaltered
im_rot.append(im)
cv.imwrite('frames_rot/frame_'+str(12760)+'_rot.jpg', im)

for i in range(1,len(imgs)):
    # construct the rotation matrix
    M = euler2matrix(euler*i)
    # perform rotation
    im_rot.append(cv.warpPerspective(im, M, (width, height)))
    cv.imwrite('frames_rot/frame_'+str(12760+i)+'_rot.jpg', im_rot[i])
    # show frame
    plt.figure(figsize=(10,10))
    plt.imshow(cv.cvtColor(im_rot[i], cv.COLOR_BGR2RGB))
    plt.show()
        