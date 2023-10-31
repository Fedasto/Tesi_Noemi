import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

### use camera_matrix and dist_coeff ###
camera_settings = True
var = 0

### use ExtrinsicGuess in solcePnP ###
ExtrinsicGuess = True

### visualize target plots ###
plot_3D = False
plot_2D = False

### save plots? ####
saveplots = False

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

if camera_settings:
    camera_matrix = np.array([[2.70035329e+03, 0., 1.83595278e+03],
                              [0., 2.70571803e+03, 1.04960056e+03],
                              [0., 0., 1.]], dtype = "double")
    dist_coeff = np.array(([-0.011935952], [0.03064728],  [-0.00067055], [ -0.00512621], [-0.11974069]))
    dist_var = dist_coeff + var * dist_coeff
else:
    camera_matrix = np.array([[2.70035329e+03, 0., 1.83595278e+03],
                              [0., 2.70571803e+03, 1.04960056e+03],
                              [0., 0., 1.]], dtype = "double")
    dist_coeff = np.zeros(5)
    dist_var = np.zeros(5)

path = '/Users/noemi/Documents/Uni/TESI/'
outpath = '/Users/noemi/Documents/Uni/TESI/SolvePnP/outputs/SAME/'
#name = 'd_roll'

### FUNCTIONS ##################################################################################################

### transforms a rotation matrix into the corrisponding euler angles ###
def matrix2euler(R):
    oz = np.arctan2(R[1][0], R[0][0])
    oy = np.arctan2(-R[2][0], np.sqrt(R[0][0] ** 2 + R[1][0] ** 2))
    ox = np.arctan2(R[2][1], R[2][2])
    return (ox, oy, oz)

### transforms euler angles into a rotation matrix ###
def euler2matrix(angle):
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

### root mean square ###
def RMS(angle, value):
    asum = np.sum((angle - value)**2)
    rms = np.sqrt(asum/len(angle))
    return rms

### discrepancy ###
def discr(angle, value):
    d = abs(angle - value)
    mean_d = np.mean(d)
    return mean_d

#%%
### CENTRE TRANSLATION #########################################################################################
# in order to rotate the image around the centre, we translate the centre in (0,0,u)

### finding the centre ###
ec = pt3D_orig[15,0] + ((pt3D_orig[8,0] - pt3D_orig[15,0])/2)
nc = pt3D_orig[14,1] + ((pt3D_orig[1,1] - pt3D_orig[14,1])/2)


### transalte ###
pt3D_in = pt3D_orig - [ec, nc, 0]

# show translation
fig, ax = plt.subplots(figsize=(10,5))
l1 = ax.scatter(pt3D_orig[:,0], pt3D_orig[:,1], color='orange', s=80)
l2 = ax.scatter(pt3D_in[:,0], pt3D_in[:,1], color='royalblue', s=80)
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_title('Targets position in ENU coordinates')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend((l1,l2), ('Targets\noriginal\ncoordinates', 'Targets\ncentered\nin (0,0)'), loc='center left', shadow=True, bbox_to_anchor=(1, 0.5))
plt.savefig(path+'GPS target coords/targets_ENU2D.png')
plt.show()

#%%
### 3D ROTATION ################################################################################################

### set up rotation parameters ###
yaw = np.deg2rad(45)   # rad
pitch = 0   # rad
roll = 0   # rad
euler = np.array([yaw, pitch, roll], np.float32)


### set up variations ###
angle = np.deg2rad(np.arange(-10,10, 0.4))   # pitch & yaw
#angle = np.deg2rad(np.arange(0,360))   # roll
N = len(angle)
eu_var = np.array([np.zeros(N), angle, np.zeros(N)], np.float32)


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
    
    # show 3D rotated points
    if plot_3D:
        ax = plt.figure(figsize=(5,5)).add_subplot(projection='3d')
        ax.scatter(points_3D[i][:,0], points_3D[i][:,1], points_3D[i][:,2], color='royalblue')
        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('Up')
        ax.set_xlim(-170,170)   # yaw
        ax.set_ylim(-170,170)   # yaw
        ax.set_zlim(-100, 100)   # yaw
        ax.set_title('Targets position in ENU coordinates')
        if i==0 :
            plt.savefig(path+'GPS target coords/targets_ENU3D.png')
        plt.savefig(path+'SolvePnP/outputs/3D_rot/pitch3D/pitch3D_frame'+str(i)+'.png', bbox_inches='tight')
        plt.show()
    
#%%    
### 2D PROJECTION ##############################################################################################

### set up drone parameters ###
# drone euler angles
D_yaw = 0   # rad
D_pitch = 0   # rad
D_roll = 0  # rad
D_euler = np.array([D_yaw, D_pitch, D_roll], np.float32)

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
        plt.xlim(1250,2400)
        plt.ylim(450,1650)
        plt.title('Targets position projected in 2D')
        #plt.savefig(path+'GPS target coords/targets_proj2D.png')
        plt.show()

#%%
### SOLVE PnP ##################################################################################################

rvec = []
tvec = []
yaw = []
pitch = []
roll = []
t_enu = []


### solve all the frames ###
for i in range(N):
    # get rtvecs from 2D and 3D points
    if i == 0:
        res, rvec_PnP, tvec_PnP = cv.solvePnP(points_3D[i], points_2D[i], camera_matrix, dist_var, flags=0)
    else:
        res, rvec_PnP, tvec_PnP = cv.solvePnP(points_3D[i], points_2D[i], camera_matrix, dist_var, np.array(rvec)[i-1], np.array(tvec)[i-1], useExtrinsicGuess=ExtrinsicGuess, flags=0)
    
    rvec_PnP = rvec_PnP.squeeze()
    tvec_PnP = tvec_PnP.squeeze()
    tvec.append(list(tvec_PnP))
    rvec.append(list(rvec_PnP))
    
    # compute euler angles
    M = cv.Rodrigues(rvec_PnP)[0]
    ox, oy, oz = matrix2euler(M)
    yaw.append(np.rad2deg(ox))
    pitch.append(np.rad2deg(oy))
    roll.append(np.rad2deg(oz))
    
    # compute t_enu
    t_enu.append(np.dot(tvec_PnP.squeeze(), M))
    
rvec = np.array(rvec)
tvec = np.array(tvec)
yaw = np.array(yaw)
pitch = np.array(pitch)
roll = np.array(roll)
t_enu = np.array(t_enu)


### compute differences between input & output ###
# write in txt
if saveplots:
    o = open(outpath+name+'/'+str(name)+'.txt','w')
    print('DISCREPANCY', file=o)
    print('rvec:\t', discr(rvec[:,0], D_rvec[0]), '\t', discr(rvec[:,1], D_rvec[1]), '\t', discr(rvec[:,2], D_rvec[2]), '\n', file=o)
    print('tvec:\t', discr(tvec[:,0], D_tvec[0]), '\t', discr(tvec[:,1], D_tvec[1]), '\t', discr(tvec[:,2], D_tvec[2]), '\n', file=o)
    print('rot:\t', discr(yaw, D_yaw), '\t', discr(pitch, D_pitch), '\t', discr(roll, D_roll), '\n', file=o)
    print('transl:\t', discr(t_enu[:,0], D_t_enu[0]), '\t', discr(t_enu[:,1], D_t_enu[1]), '\t', discr(t_enu[:,2], D_t_enu[2]), '\n', file=o)
    print('ROOT MEAN SQUARE', file=o)
    print('rvec:\t', RMS(rvec[:,0], D_rvec[0]), '\t', RMS(rvec[:,1], D_rvec[1]), '\t', RMS(rvec[:,2], D_rvec[2]), '\n', file=o)
    print('tvec:\t', RMS(tvec[:,0], D_tvec[0]), '\t', RMS(tvec[:,1], D_tvec[1]), '\t', RMS(tvec[:,2], D_tvec[2]), '\n', file=o)
    print('rot:\t', RMS(yaw, D_yaw), '\t', RMS(pitch, D_pitch), '\t', RMS(roll, D_roll), '\n', file=o)
    print('transl:\t', RMS(t_enu[:,0], D_t_enu[0]), '\t', RMS(t_enu[:,1], D_t_enu[1]), '\t', RMS(t_enu[:,2], D_t_enu[2]), '\n', file=o)
    o.close()

#%%
### RESULTS ####################################################################################################

### rvec ###
plt.figure(figsize=(12,4))
plt.suptitle('Solved rvec', fontsize='14', y=1.03)
#yaw
plt.subplot(131)
plt.plot(np.rad2deg(angle), rvec[:,2], color='orange')
plt.plot(np.rad2deg(angle), np.full(N, D_rvec[2]), color='black', linestyle='dashed')
plt.xlabel('drone position variation (deg)')
plt.ylabel('(rad)')
plt.title('Yaw', y=1.05)
# pitch
plt.subplot(132)
plt.plot(np.rad2deg(angle), rvec[:,1], color='coral')
plt.plot(np.rad2deg(angle), np.full(N, D_rvec[1]), color='black', linestyle='dashed')
plt.xlabel('drone position variation (deg)')
plt.ylabel('(rad)')
plt.title('Pitch', y=1.05)
# roll
plt.subplot(133)
plt.plot(np.rad2deg(angle), rvec[:,0], color='crimson')
plt.plot(np.rad2deg(angle), np.full(N, D_rvec[0]), color='black', linestyle='dashed')
plt.xlabel('drone position variation (deg)')
plt.ylabel('(rad)')
plt.title('Roll', y=1.05)
# save and show
if saveplots:
    plt.savefig(outpath+name+'/rvec_'+str(name)+'.png')
plt.show()


### tvec ###
plt.figure(figsize=(12,4))
plt.suptitle('Solved tvec', fontsize='14', y=1.03)
# x
plt.subplot(131)
plt.plot(np.rad2deg(angle), tvec[:,0], color='c')
plt.plot(np.rad2deg(angle), np.full(N, D_tvec[0]), color='black', linestyle='dashed')
plt.xlabel('drone position variation (deg)')
plt.title('x', y=1.05)
# y
plt.subplot(132)
plt.plot(np.rad2deg(angle), tvec[:,1], color='royalblue')
plt.plot(np.rad2deg(angle), np.full(N, D_tvec[1]), color='black', linestyle='dashed')
plt.xlabel('drone position variation (deg)')
plt.title('y', y=1.05)
# z
plt.subplot(133)
plt.plot(np.rad2deg(angle), tvec[:,2], color='mediumpurple')
plt.plot(np.rad2deg(angle), np.full(N, D_tvec[2]), color='black', linestyle='dashed')
#plt.ylim(D_tvec[2]-7*10**(-12), D_tvec[2]+7*10**(-12))
plt.xlabel('drone position variation (deg)')
plt.title('z', y=1.05)
# save and show
if saveplots:
    plt.savefig(outpath+name+'/tvec_'+str(name)+'.png', bbox_inches='tight')
plt.show()


### rotation ###
plt.figure(figsize=(12,4))
plt.suptitle('Solved euler angles', fontsize='14', y=1.03)
#yaw
plt.subplot(131)
plt.plot(np.rad2deg(angle), yaw, color='orange')
plt.plot(np.rad2deg(angle), np.full(N, np.rad2deg(D_yaw)), color='black', linestyle='dashed')
#plt.xlabel('drone position variation (deg)')
plt.ylabel('(deg)')
plt.title('Yaw', y=1.05)
# pitch
plt.subplot(132)
plt.plot(np.rad2deg(angle), pitch, color='coral')
plt.plot(np.rad2deg(angle), np.full(N, np.rad2deg(D_pitch)), color='black', linestyle='dashed')
plt.xlabel('drone position variation (deg)')
#plt.ylabel('(deg)')
plt.title('Pitch', y=1.05)
# roll
plt.subplot(133)
plt.plot(np.rad2deg(angle), roll, color='crimson')
plt.plot(np.rad2deg(angle), np.full(N, np.rad2deg(D_roll)), color='black', linestyle='dashed')
#plt.xlabel('drone position variation (deg)')
#plt.ylabel('(deg)')
plt.title('Roll', y=1.05)
# save and show
if saveplots:
    plt.savefig(outpath+name+'/rot_'+str(name)+'.png', bbox_inches='tight')
plt.show()


### translation ###
plt.figure(figsize=(12,4))
plt.suptitle('Drone position in ENU coordinates', fontsize='14', y=1.03)
# east
plt.subplot(131)
plt.plot(np.rad2deg(angle), t_enu[:,0], color='c')
plt.plot(np.rad2deg(angle), np.full(N, D_t_enu[0]), color='black', linestyle='dashed')
#plt.xlabel('drone position variation (deg)')
plt.ylabel('(m)')
plt.title('East', y=1.05)
# north
plt.subplot(132)
plt.plot(np.rad2deg(angle), t_enu[:,1], color='royalblue')
plt.plot(np.rad2deg(angle), np.full(N, D_t_enu[1]), color='black', linestyle='dashed')
plt.xlabel('drone position variation (deg)')
#plt.ylabel('(m)')
plt.title('North', y=1.05)
# up
plt.subplot(133)
plt.plot(np.rad2deg(angle), t_enu[:,2], color='mediumpurple')
plt.plot(np.rad2deg(angle), np.full(N, D_t_enu[2]), color='black', linestyle='dashed')
#plt.ylim(D_t_enu[2]-7*10**(-12), D_t_enu[2]+7*10**(-12))
#plt.xlabel('drone position variation (deg)')
#plt.ylabel('(m)')
plt.title('Up', y=1.05)
# save and show
if saveplots:
    plt.savefig(outpath+name+'/transl_'+str(name)+'.png', bbox_inches='tight')
plt.show()
