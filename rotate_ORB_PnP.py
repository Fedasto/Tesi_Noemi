import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import re
import glob
from  tqdm import tqdm
import skimage.color as skc
from scipy.ndimage import gaussian_filter
from photutils.centroids import centroid_com
import copy

################################################################################################################

points_3D = np.array([[  52.70329516,    2.09686929,   -1.30861781], #1
 [  80.55151081,   24.01953222,    0.77764659], #2
 [  66.70776813,  -23.2429411 ,   -3.20539092], #3
 [  46.33564945,  -40.37986811,   -3.07979647], #4
 [  -3.80981663,  -42.16976835,   -0.43974115], #6
 [  35.67897195,  -70.31805264,   -3.53668899], #7
 [  52.05945531,  -82.23929068,   -4.3062447 ], #8
 [  52.7141591 , -110.67903031,   -5.13688207], #9
 [ 126.98066486,  -91.76024083,   -7.81632531], #10
 [ 125.73203857,  -45.09735894,   -5.34239778], #11
 [ 112.27144477,  -46.04376631,   -5.35795376], #12
 [  76.58937056,  -44.2835237 ,   -4.09601365], #13
 [  -6.48342586, -127.00695482,   -0.28027338], #14
 [   8.81218279, -163.3081136 ,   -0.47710596], #15
 [ -25.51178011, -186.9776997 ,    1.28399635], #16
 [ -75.76876986, -176.00107678,    5.64441157], #17
 [  32.75262929, -168.97270725,   -4.93633206], #18
 [  51.2955913 , -151.78804288,   -7.33932007]]) #19

camera_matrix = np.array([[2.70035329e+03, 0., 1.83595278e+03],
                          [0., 2.70571803e+03, 1.04960056e+03],
                          [0., 0., 1.]], dtype = "double")
dist_coeff = np.array(([-0.011935952], [0.03064728],  [-0.00067055], [ -0.00512621], [-0.11974069]))

targets = np.array([[1281,957], #frame 12760
[1369,1114],
[1184,1043],
[1118,996],
[1106,677],
[984,899],
[932,998],
[795,1019],
[888,1430],
[1099,1373],
[1093,1300],
[1098,1108],
[717,670],
[520,771],
[376,548],
[450,215],
[477,929],
[572,1038]], dtype='double')

path = '/Users/noemi/Documents/Uni/TESI/'

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

def delta_E(image_1_rgb,color_target,sigma,dmax):

    Lab1=cv.cvtColor((image_1_rgb/255).astype('float32'), cv.COLOR_RGB2LAB)
    Lab2= skc.rgb2lab(color_target.reshape(1, 1, 3))

    deltae=skc.deltaE_ciede2000(Lab1,Lab2)

    deltae=gaussian_filter(deltae,3)

    minDeltaE = np.min(deltae)

    if minDeltaE <= 25:
        fimage=dmax*np.exp(-(deltae-minDeltaE)**2/(2*(sigma**2))) # sigma=2 genrally used in color space
        fimage[fimage<0.65]=0
    else:
        fimage = np.ones_like(deltae)*0 #np.nan
    
    return fimage,minDeltaE

### transforms a rotation matrix into the corrisponding euler angles ###
def matrix2euler(R):
    oz = np.arctan2(R[1][0], R[0][0])
    oy = np.arctan2(-R[2][0], np.sqrt(R[0][0] ** 2 + R[1][0] ** 2))
    ox = np.arctan2(R[2][1], R[2][2])
    return (ox, oy, oz)

### ORB ########################################################################################################

# collect all frames
imgs=sorted(glob.glob('/Users/noemi/Documents/Uni/TESI/Distorsioni/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])


### get first frame and frame dimensions ###
im = cv.imread(imgs[0])[450:2150, 700:2350]
width = im.shape[1]
height = im.shape[0]


### set up rotation parameters ###
roll = np.zeros(95)   # rad
pitch = np.zeros(95)   # rad
yaw = np.random.rand(95)   # rad
euler = np.float32([roll, pitch, yaw])

'''### test: compute corresponding rvec ###
M = euler2matrix(euler[0])
rvec = cv.Rodrigues(M)[0]
print('\nFirst frame rvec:\n', rvec)'''


### perform rotation on all frames ###
im_rot = []

# first frame is unaltered
im_rot.append(im)
cv.imwrite('frames_rot/frame_'+str(12760)+'_rot.jpg', im)

i=0
for e in euler:
    i += 1
    # construct the rotation matrix
    M = euler2matrix(e)
    if i == len(imgs)-1:
        print('\nLast frame angles (check):\n', np.rad2deg(e))
    # perform rotation
    im_rot.append(cv.warpPerspective(im, M, (width, height)))
    cv.imwrite('frames_rot/frame_'+str(12760+i)+'_rot.jpg', im_rot[i])
    # show frame
    plt.figure(figsize=(10,10))
    plt.imshow(cv.cvtColor(im_rot[i], cv.COLOR_BGR2RGB))
    plt.show()
    
#################    

imgs=sorted(glob.glob(path+'Distorsioni/frames_rot/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

img_guess=cv.imread(imgs[0], cv.IMREAD_GRAYSCALE)

print('\n Searching target position')

C=[]
homography=[]
minDE=[]
target_color=np.array([0.310, 0.558, 0.811])

orb = cv.ORB_create()
count = 1
q=0

box_width = 20
box_height = 20

height, width = img_guess.shape 

o = open('targets.txt','w')

for im in tqdm(imgs[0+1:96]):
    q+=1
    temp = cv.imread(im)
    temp=cv.cvtColor(temp,cv.COLOR_BGR2RGB)
    img_new = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)

    kp1, des1 = orb.detectAndCompute(img_guess,None)
    kp2, des2 = orb.detectAndCompute(img_new,None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    dmatches = sorted(matches, key = lambda x:x.distance)

    src_pts  = np.float32([kp1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
    dst_pts  = np.float32([kp2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

    ## find homography matrix and do perspective transform
    M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS, 5.0)
    homography.append(M)

    targets = cv.perspectiveTransform(targets.reshape(-1,1,2), M)
    
    res = targets.reshape(len(targets),2)
    
    t=0
    for item in res:
        t+=1
        if item[0] > width or item[1] > height:
            pass

        else:
            center = np.nan
            fact = 1
            while not np.all(np.isfinite(center)):

                box = temp[
                    int(item[1]-fact*box_height):int(item[1]+fact*box_height),
                    int(item[0]-fact*box_width):int(item[0]+fact*box_width)
                    ] 
                
                mask_box,min_delta_e=delta_E(box,target_color,2,1)
                
                minDE.append(min_delta_e)

                center = centroid_com(mask_box) 
            
                center[1] += int(item[1]-fact*box_height)
                center[0] += int(item[0]-fact*box_width)
                fact *= 2

            C.append([center[0],center[1]]) #(x,y)

            cv.drawMarker(
                temp, (int(center[0]), int(center[1])),(0,255,0), markerSize=20, markerType=cv.MARKER_SQUARE, thickness=2, line_type=cv.LINE_AA)
    
    targets=np.array(C[-len(targets):]).reshape(-1,2)
    
    print('\nframe #', 12760+q, ':\n', np.array2string(targets, separator=", "), file=o)

    img_guess = copy.copy(img_new)
    count += 1

o.close()

plt.figure(figsize=(10,10))
plt.imshow(temp)

plt.savefig(path+'SolvePnP/target_last_frame.png')

targetList=np.array(C)

points_2D=np.zeros([95,len(targets),2])

for i in range(95):
    points_2D[i,:,:] = (targetList[i*len(targets):len(targets)+(i*len(targets))])

rvec = []
tvec = []

j=0
for pt2D in points_2D:
    j += 1
    
    if j == 1:
        ### get rtvecs from 2D and 3D points ###
        res, rvec_PnP, tvec_PnP = cv.solvePnP(points_3D, pt2D, camera_matrix, dist_coeff, flags=0)
        rvec.append(rvec_PnP)
        tvec.append(tvec_PnP)
    else:
        ### get rtvecs from 2D and 3D points ###
        res, rvec_PnP, tvec_PnP = cv.solvePnP(points_3D, pt2D, camera_matrix, dist_coeff, rvec=rvec[-1], tvec=tvec[-1], useExtrinsicGuess=True, flags=0)
        rvec.append(rvec_PnP)
        tvec.append(tvec_PnP)
        
print(rvec)
        
rvec = np.array(rvec)
tvec = np.array(tvec)
        
plt.plot(rvec[:,2])
plt.show()

plt.plot(tvec[:,2])
plt.show()
    