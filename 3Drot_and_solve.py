import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as skc
from scipy.ndimage import gaussian_filter

### use camera_matrix and dist_coeff ###
camera_settings = False

### visualize target plots ###
plot_3D = False
plot_2D = False

### number of rotations ###
# as if they were frames
N = 96

################################################################################################################

pt3D_in = np.array([[  52.70329516,    2.09686929,   -1.30861781], #1
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

if camera_settings:
    camera_matrix = np.array([[2.70035329e+03, 0., 1.83595278e+03],
                              [0., 2.70571803e+03, 1.04960056e+03],
                              [0., 0., 1.]], dtype = "double")
    dist_coeff = np.array(([-0.011935952], [0.03064728],  [-0.00067055], [ -0.00512621], [-0.11974069]))
else:
    camera_matrix = np.array([[1., 0., 0.],
                              [0., 1., 0.],
                              [0, 0., 1.]], dtype = "double")
    dist_coeff = np.zeros(5)

path = '/Users/noemi/Documents/Uni/TESI/'

### FUNCTIONS ##################################################################################################

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

#%%
### 3D ROTATION ################################################################################################

### set up rotation parameters ###
roll = np.deg2rad(10) / (N-1)   # rad
pitch = np.deg2rad(7) / (N-1)   # rad
yaw = np.deg2rad(150) / (N-1)   # rad
euler = np.array([roll, pitch, yaw], np.float32)


### rotate 3D coordinates ###
points_3D = []
for i in range(N):
    
    # construct the rotation matrix
    R = euler2matrix(euler*i)
    if i == N-1:
        print('\nLast frame angles (check):\n', np.rad2deg(matrix2euler(R)))
        
    # perform rotation
    temp_3D = []
    for j in range(len(pt3D_in)):
        temp_3D.append(np.dot(R, pt3D_in[j]))
    points_3D.append(np.array(temp_3D))
    
    # show 3D rotated points
    if plot_3D:
        ax = plt.figure(figsize=(5.5,5.5)).add_subplot(projection='3d')
        ax.scatter(points_3D[i][:,0], points_3D[i][:,1], points_3D[i][:,2])
        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('Up')
        ax.set_xlim(-170,170)
        ax.set_ylim(-170,170)
        ax.set_zlim(-100, 100)
        ax.set_title('Targets position in ENU coordinates')
        #plt.savefig(path+'GPS target coords/targets_ENU3D.png')
        plt.show()
    
#%%    
### 2D PROJECTION ##############################################################################################

### set up drone parameters ###
# drone euler angles
D_roll = np.deg2rad(30)   # rad
D_pitch = np.deg2rad(12)   # rad
D_yaw = np.deg2rad(180)  # rad
D_euler = np.array([D_roll, D_pitch, D_yaw], np.float32)

# drone t_enu 
D_t_enu = np.array([101.37471736, -105.49861596, 500]) # centered


### drone rotation vector ###
D_R = euler2matrix(D_euler)
D_rvec = cv.Rodrigues(D_R)[0]
print('\nDrone angles (check):\n', np.rad2deg(matrix2euler(D_R)))


### drone translation vector ###
D_tvec = np.dot(D_R, D_t_enu)


### project points in 2D ###  
points_2D = []
for i in range(N):
    temp_2D = []
    for j in range(len(pt3D_in)):  
        pt = cv.projectPoints(points_3D[i][j], D_rvec, D_tvec, camera_matrix, dist_coeff)[0]
        temp_2D.append(pt.squeeze())  
    points_2D.append(np.array(temp_2D))
    
    # show 2D projected points
    if plot_2D:
        plt.scatter(points_2D[i][:,0], points_2D[i][:,1])
        plt.xlabel('x')
        plt.ylabel('y')
        if camera_settings:
            plt.xlim(1500,3550)
            plt.ylim(-700,1600)
        else:
            plt.xlim(-1.5, 0)
            plt.ylim(0, 2.5)
        plt.title('Targets position projected in 2D')
        #plt.savefig(path+'GPS target coords/targets_proj2D.png')
        plt.show()

#%%
### SOLVE PnP ##################################################################################################

rvec = []
tvec = []
roll = []
pitch = []
yaw = []
t_enu = []


### solve all the frames ###
for i in range(N):
    
    # get rtvecs from 2D and 3D points
    res, rvec_PnP, tvec_PnP = cv.solvePnP(points_3D[i], points_2D[i], camera_matrix, dist_coeff)
    rvec.append(rvec_PnP.squeeze())
    tvec.append(tvec_PnP.squeeze())
    
    # compute euler angles
    M = cv.Rodrigues(rvec_PnP)[0]
    ox, oy, oz = matrix2euler(M)
    roll.append(np.rad2deg(ox))
    pitch.append(np.rad2deg(oy))
    yaw.append(np.rad2deg(oz))
    
    # compute t_enu
    t_enu.append(np.dot(tvec_PnP.squeeze(), M))
    
rvec = np.array(rvec)
tvec = np.array(tvec)
t_enu = np.array(t_enu)

#%%
### RESULTS ####################################################################################################

### rvec ###
plt.figure(figsize=(30,10))
plt.suptitle('Solved rvec', fontsize='20')
# roll
plt.subplot(131)
plt.plot(rvec[:,0], color='crimson')
plt.plot(np.full(N, D_rvec[0]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('(rad)')
plt.title('Roll')
# pitch
plt.subplot(132)
plt.plot(rvec[:,1], color='coral')
plt.plot(np.full(N, D_rvec[1]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('(rad)')
plt.title('Pitch')
#yaw
plt.subplot(133)
plt.plot(rvec[:,2], color='orange')
plt.plot(np.full(N, D_rvec[2]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('(rad)')
plt.title('Yaw')
# save and show
#plt.savefig(path+'SolvePnP/outputs/euler1_re5.png')
plt.show()


### tvec ###
plt.figure(figsize=(30,10))
plt.suptitle('Solved tvec', fontsize='20')
# x
plt.subplot(131)
plt.plot(tvec[:,0], color='c')
plt.plot(np.full(N, D_tvec[0]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.title('Roll')
# y
plt.subplot(132)
plt.plot(tvec[:,1], color='royalblue')
plt.plot(np.full(N, D_tvec[1]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.title('Pitch')
# z
plt.subplot(133)
plt.plot(tvec[:,2], color='mediumpurple')
plt.plot(np.full(N, D_tvec[2]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.title('Yaw')
# save and show
#plt.savefig(path+'SolvePnP/outputs/euler1_re5.png')
plt.show()


'''### rotation ###
plt.figure(figsize=(30,10))
plt.suptitle('Solved euler angles', fontsize='20')
# roll
plt.subplot(131)
plt.plot(roll, color='crimson')
plt.plot(np.full(N, np.rad2deg(D_roll)), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('angle (deg)')
plt.title('Roll')
# pitch
plt.subplot(132)
plt.plot(pitch, color='coral')
plt.plot(np.full(N, np.rad2deg(D_pitch)), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('angle (deg)')
plt.title('Pitch')
#yaw
plt.subplot(133)
plt.plot(yaw, color='orange')
plt.plot(np.full(N, np.rad2deg(D_yaw)), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('angle (deg)')
plt.title('Yaw')
# save and show
#plt.savefig(path+'SolvePnP/outputs/euler1_re5.png')
plt.show()'''


'''### translation ###
plt.figure(figsize=(30,10))
plt.suptitle('Drone position in ENU coordinates', fontsize='20')
# east
plt.subplot(131)
plt.plot(t_enu[:,0], color='c')
plt.plot(np.full(N, D_t_enu[0]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('(m)')
plt.title('East')
# north
plt.subplot(132)
plt.plot(t_enu[:,1], color='royalblue')
plt.plot(np.full(N, D_t_enu[1]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('(m)')
plt.title('North')
# up
plt.subplot(133)
plt.plot(t_enu[:,2], color='mediumpurple')
plt.plot(np.full(N, D_t_enu[2]), color='black', linestyle='dashed')
plt.xlabel('frame')
plt.ylabel('(m)')
plt.title('Up')
#plt.savefig(path+'SolvePnP/outputs/euler1_re5.png')
plt.show()'''
