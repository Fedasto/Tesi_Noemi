import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

ransac = False

################################################################################################################

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
points_2D = np.array([[1093.68635122, 1159.90314016],
 [1155.3674533 , 1330.83036335],
 [ 982.9347403 , 1226.14899651],
 [ 929.48122728, 1171.79419391],
 [ 971.41193047,  854.83904582],
 [ 813.13781306, 1051.69590548],
 [ 743.96668232, 1138.51451433],
 [ 605.77489415, 1135.0711942 ],
 [ 626.20934847, 1557.96069885],
 [ 843.40618395, 1535.85229066],
 [ 848.00444883, 1464.51276958],
 [ 888.84917317, 1276.32897952],
 [ 590.56343306,  779.99841344],
 [ 380.50675659,  845.03761068],
 [ 273.75868233,  600.47572546],
 [ 404.84142855,  285.16992229],
 [ 307.59686113,  993.12886788],
 [ 383.39036773, 1114.61626369]]
, dtype='double')

camera_matrix = np.array([[2.70035329e+03, 0., 1.83595278e+03],
                          [  0., 2.70571803e+03, 1.04960056e+03],
                          [  0., 0., 1.]], dtype = "double")

dist_coeff = np.array(([-0.011935952], [0.03064728],  [-0.00067055], [ -0.00512621], [-0.11974069]))

### FUNCTIONS ##################################################################################################

### transforms a rotation matrix into the corrisponding euler angles ###
def matrix2euler(R):
    oz = np.arctan2(R[1][0], R[0][0])
    oy = np.arctan2(-R[2][0], np.sqrt(R[0][0] ** 2 + R[1][0] ** 2))
    ox = np.arctan2(R[2][1], R[2][2])
    return (ox, oy, oz)

################################################################################################################

# plt 2D projected points
plt.scatter(points_2D[:,0], points_2D[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Targets position projected in 2D')
#plt.savefig(path+'GPS target coords/targets_proj2D.png')
plt.show()

if ransac:
    ### get rtvecs from 2D and 3D points ###
    res, rvec_PnP, tvec_PnP, inlier_PnP = cv.solvePnPRansac(points_3D, points_2D, camera_matrix, dist_coeff, reprojectionError=5)
    print('\nsolved rvec:', rvec_PnP)
    print('\nsolved tvec:', tvec_PnP)
    print('\n#targets used:', len(inlier_PnP))
    
    print('\nres:', res)
    print(np.linalg.norm(np.rad2deg(rvec_PnP)))
    
    
    ### compute euler angles ###
    M = cv.Rodrigues(rvec_PnP)[0]
    ox, oy, oz = matrix2euler(M)
    roll = np.rad2deg(oz)
    pitch = np.rad2deg(oy)
    yaw = np.rad2deg(ox)
    print('\nEuler angles:')
    print('roll: ', roll)
    print('pitch: ', pitch)
    print('yaw: ', yaw)
    
    
    ### compute t_enu ###
    t_enu = np.dot(M, tvec_PnP)
    print('\nDrone position:', t_enu)
    
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


else:
    ### get rtvecs from 2D and 3D points ###
    res, rvec_PnP, tvec_PnP = cv.solvePnP(points_3D, points_2D, camera_matrix, dist_coeff)
    print('\nsolved rvec:', rvec_PnP)
    print('\nsolved tvec:', tvec_PnP)
    
    print('\nres:', res)
    print(np.rad2deg(np.linalg.norm(rvec_PnP)))
    
    
    ### compute euler angles ###
    M = cv.Rodrigues(rvec_PnP)[0]
    ox, oy, oz = matrix2euler(M)
    roll = np.rad2deg(oz)
    pitch = np.rad2deg(oy)
    yaw = np.rad2deg(ox)
    print('\nEuler angles:')
    print('roll: ', roll)
    print('pitch: ', pitch)
    print('yaw: ', yaw)
    
    
    ### compute t_enu ###
    t_enu = np.dot(M, tvec_PnP)
    print('\nDrone position:', t_enu)

