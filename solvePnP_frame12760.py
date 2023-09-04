import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# from "points3D_maker.py"
# target points in enu coordinates
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

# target points in 2D plane
points_2D = np.array([[1281,957], #frame 12760
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

camera_matrix = np.array([[2.70035329e+03, 0., 1.83595278e+03],
                          [  0., 2.70571803e+03, 1.04960056e+03],
                          [  0., 0., 1.]], dtype = "double")

dist_coeff = np.array(([-0.011935952], [0.03064728],  [-0.00067055], [ -0.00512621], [-0.11974069]))

################################################################################################################

### get rtvecs from 2D and 3D points ###
res, rvec_PnP, tvec_PnP, inlier_PnP = cv.solvePnPRansac(points_3D, points_2D, camera_matrix, dist_coeff, reprojectionError=6)
print('\nsolved rvec:', rvec_PnP)
print('\nsolved tvec:', tvec_PnP)
print('\n#targets used:', len(inlier_PnP))
print(inlier_PnP)


### computing the number of targets used depending on the reprojection error ###
rvec = []
tvec = []
n_targets = []
err = np.arange(0.5, 25, 0.5)

for i in range(len(err)):
    res, rvec_prov, tvec_prov, inlier = cv.solvePnPRansac(points_3D, points_2D, camera_matrix, dist_coeff, reprojectionError=err[i])
    rvec.append(rvec_prov)
    tvec.append(tvec_prov)
    #print('\nprojection error =', err[i])
    #print('rvec:', rvec[i])
    #print('tvec:', tvec[i])
    if np.any(inlier == None):
        n_targets.append(0)
    else:
        n_targets.append(len(inlier))

plt.plot(err, n_targets)
plt.xlabel('reprojection error')
plt.ylabel('# targets')
plt.title('Used targets')
plt.show()
