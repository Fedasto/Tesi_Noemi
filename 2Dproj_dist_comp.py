import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as skc
from scipy.ndimage import gaussian_filter

################################################################################################################

pt3D_orig = np.array([[  52.70329516,    2.09686929,   -1.30861781], #1
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

path = '/Users/noemi/Documents/Uni/TESI/'
outpath = '/Users/noemi/Documents/Uni/TESI/SolvePnP/outputs/'

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
### CENTRE TRANSLATION #########################################################################################
# in order to rotate the image around the centre, we translate the centre in (0,0,u)

### finding the centre ###
ec = pt3D_orig[15,0] + ((pt3D_orig[8,0] - pt3D_orig[15,0])/2)
nc = pt3D_orig[14,1] + ((pt3D_orig[1,1] - pt3D_orig[14,1])/2)


### transalte ###
pt3D_in = pt3D_orig - [ec, nc, 0]

#%%
### 3D ROTATION ################################################################################################

### set up rotation parameters ###
yaw = np.deg2rad(45)   # rad
pitch = 0   # rad
roll = 0   # rad
euler = np.array([yaw, pitch, roll], np.float32)


### set up variations ###
#angle = np.deg2rad(np.arange(-10,10, 0.4))   # pitch & roll
angle = np.deg2rad(np.arange(0,360))   # yaw
N = len(angle)
eu_var = np.array([np.zeros(N), np.zeros(N), angle], np.float32)


### rotate 3D coordinates ###
points_3D = []
for i in range(N):
    
    # construct the rotation matrix
    R = euler2matrix(euler + eu_var[:,i])
    if i == N-1:
        print('\nLast frame angles (check):\n', np.rad2deg(matrix2euler(R)))
        
    # perform rotation
    temp_3D = []
    for j in range(len(pt3D_in)):
        temp_3D.append(np.dot(R, pt3D_in[j]))
    points_3D.append(np.array(temp_3D))
    
#%%    
### 2D PROJECTION ##############################################################################################

### set up drone parameters ###
# drone euler angles
D_roll = 0   # rad
D_pitch = 0   # rad
D_yaw = 0  # rad
D_euler = np.array([D_roll, D_pitch, D_yaw], np.float32)

# drone t_enu 
D_up = np.sqrt(500**2 + 500**2)
D_t_enu = np.array([0, 0, D_up]) # centered


### drone rotation vector ###
D_R = euler2matrix(D_euler)
D_rvec = cv.Rodrigues(D_R)[0]
print('\nDrone angles (check):\n', np.rad2deg(matrix2euler(D_R)))


### drone translation vector ###
D_tvec = np.dot(D_R, D_t_enu)


### project points in 2D ###  
points_2D = []
points_dist = []
for i in range(N):
    temp_2D = []
    temp_dist = []
    for j in range(len(pt3D_in)):  
        pt = cv.projectPoints(points_3D[i][j], D_rvec, D_tvec, camera_matrix, np.zeros(5))[0]
        d = cv.projectPoints(points_3D[i][j], D_rvec, D_tvec, camera_matrix, dist_coeff)[0]
        temp_2D.append(pt.squeeze())
        temp_dist.append(d.squeeze())
    points_2D.append(np.array(temp_2D))
    points_dist.append(np.array(temp_dist))
    
    # show 2D projected points
    fig, ax = plt.subplots(figsize=(5,5))
    l2 = ax.scatter(points_dist[i][:,0], points_dist[i][:,1], color='r', s=95)
    l1 = ax.scatter(points_2D[i][:,0], points_2D[i][:,1], color='lightsteelblue', s=80)
    ax.set_xlabel('x (px)')
    ax.set_ylabel('y (px)')
    ax.set_xlim(1250,2400)
    ax.set_ylim(475,1625)
    ax.set_title('Targets position projected in 2D')
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #ax.legend((l1,l2), ('Without\ndistortions', 'With camera\ndistorsions'), loc='center left', shadow=True, bbox_to_anchor=(1, 0.5))
    plt.savefig(outpath+'PROJ/roll_nolab/roll_nolab_frame'+str(i)+'.png', bbox_inches='tight')
    plt.show()
