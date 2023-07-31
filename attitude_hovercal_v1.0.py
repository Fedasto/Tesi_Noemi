import glob 
import copy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from photutils.centroids import centroid_com, centroid_2dg
from  tqdm import tqdm 
from math import cos, radians, sin, sqrt
from astropy import units as u
import time
import scipy.optimize as so
from scipy.spatial.transform import Rotation as Rtt
import re
from astropy.table import Table
import skimage.color as skc
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter
import csv
from scipy.interpolate import interp1d
import sys

########################################################################################################################################

    # CONTROL PARAMS

########################################################################################################################################

videocapture=False
get_init_coord=False
ORB=False
savefig=True
bootstrap=False

inpath='C:/Users/PC/Desktop/C0003/'
outpath='C:/Users/PC/Desktop/C0003/output/'


########################################################################################################################################

    # PARAMETERS AND COSTANTS

########################################################################################################################################

gps_fields = [['RTKdata:Lon_P', 'RTKdata:Lat_P', 'RTKdata:Hmsl_P'],
              ['RTKdata:Lon_S', 'RTKdata:Lat_S', 'RTKdata:Hmsl_S'],
              ['GPS:Long', 'GPS:Lat', 'GPS:heightMSL'],
              ['IMU_ATTI(0):Longitude', 'IMU_ATTI(0):Latitude', 'IMU_ATTI(0):alti:D'],
              ['IMU_ATTI(1):Longitude', 'IMU_ATTI(1):Latitude', 'IMU_ATTI(1):alti:D'],
              ['IMU_ATTI(2):Longitude', 'IMU_ATTI(2):Latitude', 'IMU_ATTI(2):alti:D'],
              ['IMUCalcs(0):Long:C', 'IMUCalcs(0):Lat:C', 'IMUCalcs(0):height:C'],
              ['IMUCalcs(1):Long:C', 'IMUCalcs(1):Lat:C', 'IMUCalcs(1):height:C'],
              ['IMUCalcs(2):Long:C', 'IMUCalcs(2):Lat:C', 'IMUCalcs(2):height:C'],
              ]

# define target position as GPS coordinate and image space coordinate
gpsArr = np.array([[-67.78673410,-22.95972985,5134.2145], #1
[-67.78646277,-22.95953205,5136.3011], #2
[-67.78659765,-22.95995848,5132.3179], #3
[-67.78679614,-22.96011310,5132.4434], #4
[-67.78728472,-22.96012925,5135.0833], #6
[-67.78689997,-22.96038322,5131.9867], #7
[-67.78674037,-22.96049078,5131.2174], #8
[-67.78673399,-22.96074738,5130.3872], #9
[-67.78601039,-22.96057668,5127.7085], #10
[-67.78602256,-22.96015566,5130.1819], #11
[-67.78615371,-22.96016420,5130.1661], #12
[-67.78650137,-22.96014832,5131.4275], #13
[-67.78731077,-22.96089470,5135.2439], #14
[-67.78716174,-22.96122223,5135.0479], #15
[-67.78749617,-22.96143579,5136.8097], #16
[-67.78798584,-22.96133675,5141.1702], #17
[-67.78692848,-22.96127334,5130.5889], #18
[-67.78674781,-22.96111829,5128.1856] #19
                     ])

 
targets = np.array([[1138,888], #frame 8000
[1224,1044],
[1044,971],
[985,927],
[973,606],
[851,829],
[800,926],
[666,946],
[754,1398],
[961,1297],
[953,1227],
[961,1039],
[595,594],
[399,692],
[253,466],
[330,126],
[351,851],
[442,963]], dtype='double')  



# frame numbers

start_frame = 8000  #min: 4.40
end_frame = 16860 #min: 9.21

video_path = 'C:/Users/PC/Desktop/C0003/'
video_name = 'C0003.MP4'

# CLASS1 	= [-67.78730697,-22.95970907,  5136.77211111] 
# CLASS2= [-67.787224, -22.95978516,  5136.793125] 
# RTK_base=[-67.78721531, -22.95963352, 5139.414]
# POI=[ -67.7872489166,-22.9597481944, 5143.6] 

POI = [-67.78724760, -22.95974877, 5135.5229]

# sony_vid_vert()

camera_matrix = np.array([[2.70571803e+03,   0.,         1.04960056e+03 ],
                            [  0.,         2.70035329e+03, 1.83595278e+03],
                            [  0.,           0.,           1.        ]], dtype = 'double')

dist_coeff = np.array(([-0.0404975], [0.19781178], [-0.00250576], [-0.00050738], [-0.41519871]))

time_fields = ['Clock:offsetTime', 'GPS:Time', 'GPS:Date']
frame_rate = 29.97002997002997

gps_csv_path='C:/Users/PC/Desktop/C0003/Flight_GPSdata/FLY036.csv'

# https://sites.google.com/uc.cl/hover-cal/field-testing/toco/site-tests-february-2023


########################################################################################################################################

    # LIBRARY

########################################################################################################################################

def ellipsoid(model='WGS84'):

    """
    Return the major and minor semiaxis given an ellipsoid model.
    Returns:
    - a: astropy.Quantity, semiaxis major in meter
    - b: astropy.Quantity, semiaxis minor in meter
    """

    if model == 'WGS84':
        a = 6378137.0*u.meter         
        b = 6356752.31424518*u.meter 

    return a, b

def _check_quantity(val, unit):

    if isinstance(val, u.Quantity):
        return val.to(unit)
    else:
        return val * unit
    
def lonlat2ecef(lon, lat, alt, ell='WGS84', deg=True, uncertainty=False, \
                delta_lon=0, delta_lat=0, delta_alt=0):
    """
    convert geodetic coordinates to ECEF 
    Parameters
    ----------
    lon : float, array (numpy or Quantity)
          geodetic longitude, if numpy float or array the unit is determined by the deg parameter
    lat : float, array (numpy or Quantity)
          geodetic latitude, if numpy float or array the unit is determined by the deg parameter 
    alt : float, array (numpy or Quantity)
          altitude above geodetic ellipsoid, If numpy float or array, it is considered in meters
    ell : string, optional
          reference ellipsoid
    deg : bool, optional
          if azimuth and elevation are not astropy quantities, if True set them to degrees 
    ell : string, optional
          reference ellipsoid
    uncertainty : bool, optional
                  if True computes the uncertainties associated in ECEF coordinates
    delta_lon : float, array (numpy or Quantity)
                delta geodetic longitude, if numpy float or array the unit is determined by 
                the deg parameter
    delta_lat : float, array (numpy or Quantity)
                delta geodetic latitude, if numpy float or array the unit is determined by 
                the deg parameter 
    delta_alt : float, array (numpy or Quantity)
                delta altitude above geodetic ellipsoid, if numpy float or array the unit is 
                determined by the deg parameter
    Returns
    -------
    x : Quantity
        target x ECEF coordinate
    y : Quantity
        target y ECEF coordinate
    z : Quantity
        target z ECEF coordinate
    """

    if deg:
        lonlat_unit = u.deg
    else:
        lonlat_unit = u.rad

    lon = _check_quantity(lon, lonlat_unit)
    lat = _check_quantity(lat, lonlat_unit)
    alt = _check_quantity(alt, u.meter)
    
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    a, b = ellipsoid(ell)
        
    N = a**2/np.sqrt(a**2*cos_lat**2 + b**2*sin_lat**2)

    x = (N + alt)*cos_lat*cos_lon
    y = (N + alt)*cos_lat*sin_lon
    z = (N*(b/a)**2 + alt)*sin_lat
    
    if uncertainty:

        delta_lat = _check_quantity(delta_lat, lonlat_unit)
        delta_lon = _check_quantity(delta_lon, lonlat_unit)
        delta_alt = _check_quantity(delta_alt, u.meter)
        
        delta_x, delta_y, delta_z = _lonlat2ecef_error(cos_lon, cos_lat, sin_lon, sin_lat, \
                                                       alt, a, b, delta_lat, delta_lon, delta_alt)
    
    else:
        delta_x, delta_y, delta_z = np.zeros_like(x)*u.meter, \
            np.zeros_like(y)*u.meter, \
            np.zeros_like(z)*u.meter      
    
    return x, y, z, delta_x, delta_y, delta_z

def ecef2enu(ecef_obs_x, ecef_obs_y, ecef_obs_z, \
             ecef_target_x, ecef_target_y, ecef_target_z, \
             delta_obs_x, delta_obs_y, delta_obs_z, \
             delta_target_x, delta_target_y, delta_target_z, \
             lon, lat, deg=True):
    
    """
    Return the position relative to a refence point in a ENU system. If one of delta_obs or delta_target 
    are different from zeros then uncertainties is calculated automatically. 
    The array returns as EAST, NORTH, UP
    Parameters
    ----------
    ecef_obs_x : float or array, numpy or Quantity
                 x ECEF coordinates array of the observer, if numpy float or array the unit is meter
    ecef_obs_y : float or array, numpy or Quantity
                 y ECEF coordinates array of the observer, if numpy float or array the unit is meter
    ecef_obs_z : float or array, numpy or Quantity
                 z ECEF coordinates array of the observer, if numpy float or array the unit is meter
    ecef_target_x : float or array, numpy or Quantity
                    x ECEF coordinates array of the target, if numpy float or array the unit is meter
    ecef_target_y : float or array, numpy or Quantity
                    y ECEF coordinates array of the target, if numpy float or array the unit is meter
    ecef_target_z : float or array, numpy or Quantity
                    z ECEF coordinates array of the target, if numpy float or array the unit is meter
    delta_obs_x : numpy or Quantity, numpy or Quantity
                  x ECEF Coordinates error of the observer, if numpy array the unit is meter
    delta_obs_y : numpy or Quantity, numpy or Quantity
                  x ECEF Coordinates error of the observer, if numpy array the unit is meter
    delta_obs_z : numpy or Quantity, numpy or Quantity
                  x ECEF Coordinates error of the observer, if numpy array the unit is meter
    delta_target_x : numpy or Quantity, numpy or Quantity
                     x ECEF Coordinates error of the target, if numpy array the unit is meter
    delta_target_y : numpy or Quantity, numpy or Quantity
                     y ECEF Coordinates error of the target, if numpy array the unit is meter
    delta_target_z : numpy or Quantity, numpy or Quantity
                     z ECEF Coordinates error of the target, if numpy array the unit is meter
    lon : float, array (numpy or Quantity)
          geodetic longitude, if numpy float or array the unit is determined by the deg parameter
    lat : float, array (numpy or Quantity)
          geodetic latitude, if numpy float or array the unit is determined by the deg parameter 
    deg : bool, optional
          if azimuth and elevation are not astropy quantities, if True set them to degrees
    Returns
    --------
    E : Quantity
        East ENU coordinate error
    N : float
        North ENU coordinate error
    U : Quantity
        Up ENU coordinate error
    delta_E : Quantity
              East ENU coordinate error
    delta_N : float
              North ENU coordinate error
    delta_U : Quantity
              Up ENU coordinate error
    """
    
    if deg:
        lonlat_unit = u.deg
    else:
        lonlat_unit = u.rad

    lon = _check_quantity(lon, lonlat_unit)
    lat = _check_quantity(lat, lonlat_unit)

    ecef_obs_x = _check_quantity(ecef_obs_x, u.meter)
    ecef_obs_y = _check_quantity(ecef_obs_y, u.meter)
    ecef_obs_z = _check_quantity(ecef_obs_z, u.meter)

    ecef_target_x = _check_quantity(ecef_target_x, u.meter)
    ecef_target_y = _check_quantity(ecef_target_y, u.meter)
    ecef_target_z = _check_quantity(ecef_target_z, u.meter)

    delta_obs_x = _check_quantity(delta_obs_x, u.meter)
    delta_obs_y = _check_quantity(delta_obs_y, u.meter)
    delta_obs_z = _check_quantity(delta_obs_z, u.meter)

    delta_target_x = _check_quantity(delta_target_x, u.meter)
    delta_target_y = _check_quantity(delta_target_y, u.meter)
    delta_target_z = _check_quantity(delta_target_z, u.meter)
    
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    mat = np.array([[-sin_lon, cos_lon, 0],
                    [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
                    [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]])

    ecef_target = np.vstack((ecef_target_x, ecef_target_y, ecef_target_z))
    ecef_obs = np.vstack((ecef_obs_x, ecef_obs_y, ecef_obs_z))
 
    diff = ecef_target-ecef_obs
    enu = np.matmul(mat, diff)

    delta_obs = np.vstack((delta_obs_x, delta_obs_y, delta_obs_z))
    delta_target = np.vstack((delta_target_x, delta_target_y, delta_target_z))

    if np.any(delta_obs != 0) or np.any(delta_target != 0):
        delta_diff = np.sqrt(delta_obs**2+delta_target**2)
        delta_enu = np.matmul(mat, delta_diff)
    else:
        delta_enu = np.zeros_like(enu)
        delta_enu = _check_quantity(delta_enu, u.meter)
    
    return enu[0], enu[1], enu[2], delta_enu[0], delta_enu[1], delta_enu[2]

def lonlat2enu(lonlat_target, lonlat_ccords):
    '''
    Send coords in lon, lat, height
    lonlat_target: base or poi coordinates
    lonlat_ccords: array of coordinates
    
    returns the e,n,u coordinate array
    '''
    poi_x, poi_y, poi_z, _, _, _ = lonlat2ecef(lonlat_target[0], lonlat_target[1], lonlat_target[2])
    
    enu_arr = np.zeros(lonlat_ccords.shape)
    
    x, y, z, _, _, _ = lonlat2ecef(lonlat_ccords[:,0], lonlat_ccords[:,1], lonlat_ccords[:,2])
    #enu_arr[:,0], enu_arr[:,1], enu_arr[:,2], _, _, _ = ecef2enu(x, y, z, poi_x, poi_y, poi_z, 0,0,0,0,0,0,lonlat_target[0], lonlat_target[1])
    enu_arr[:,0], enu_arr[:,1], enu_arr[:,2], _, _, _ = ecef2enu(poi_x, poi_y, poi_z, x, y, z, 0,0,0,0,0,0,lonlat_target[0], lonlat_target[1])
    
    return enu_arr 

def matrix2euler(R):
    '''Return equivalent rotations (in radians) around x, y, and z axes given a rotation matrix'''
    oz = np.arctan2(R[1][0], R[0][0])
    oy = np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2))
    ox = np.arctan2(R[2][1], R[2][2])
    return (ox, oy, oz)

def delta_E(image_1_rgb,color_target,sigma,dmax):

    #color_img= np.zeros_like(image_1_rgb)+color
    

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

class droneData(object):



    def __init__(self, csv_file, fields=None, time_zone=0):
        if fields == None: fields = gps_fields[0]
        self.load(csv_file, fields=fields, time_zone=time_zone)
        self.test = 1

    def load(self, fname, skip=100, fields=gps_fields[0], time_zone=0):
        if np.ndim(fields) > 1: fields = flatten(fields)
        self.fields = fields
        fields = time_fields + fields

        csvfile = open(fname, encoding="utf-8")
        reader = csv.DictReader(csvfile)
        data = []

        for i, row in enumerate(reader):
            if i < skip:
                continue
            # Current loading, have to alter this to change which data is loaded form teh giant csv files
            one_row = [row[fld] for fld in fields]
            data.append(one_row)

        # Loads as strings, this converts to floats
        flight_arr = np.array(data)
        flight_arr[flight_arr == ''] = 'NaN'
        flight_arr = flight_arr.astype(float)
        flight_arr = flight_arr[np.isnan(flight_arr).sum(axis=1) == 0]
        flight_arr = flight_arr[(flight_arr == 0).sum(axis=1) == 0]

        # Add ctime information
        gps_secs = GPSDateTime2ctime(flight_arr[:,2], flight_arr[:, 1], time_zone=time_zone)

        tags = np.where(np.diff(gps_secs) > 0.5)[0] + 1
        clock = flight_arr[:, 0] # Time from the internal IMU clock

        # Find clock difference between gps tics and clocks
        error = gps_secs - np.mean(gps_secs[tags]) - (clock - np.mean(clock[tags]))

        # Find the largest gps time tick compared to the internal IMU clock
        dt0 = error[tags].max()

        # Correct the GPS time adding the fractions of a second from the IMU clock
        ctime = gps_secs - (error - dt0)

        # Find refresh data frames
        dtags = np.where(np.diff(flight_arr[:, 3]) != 0)[0]+1

        # Store ctime
        self.ct = ctime[dtags]

        # Store data in array
        self.data = flight_arr[dtags, 3:]

        # Store auxiliary time data
        self.timedata = flight_arr[dtags, :3]

    def plot(self, index, ylabel=None, title=None, new=True):
        if new: plt.figure()
        plt.plot(self.ct, self.data[:,index])
        plt.xlabel("ctime [s]")
        plt.ylabel(ylabel)
        plt.title(title)

def GPSDateTime2ctime(ymd, hms, time_zone=-4):
    YYYY = (ymd - ymd % 10000) / 10000
    mm = ymd - YYYY * 10000
    MM = (mm - mm % 100) / 100
    DD = mm % 100
    hh, mm, ss = HHMMSS2hms(hms)
    ctime = []
    for i in range(len(YYYY)):
        date = (int(YYYY[i]), int(MM[i]), int(DD[i]),
                int(hh[i]), int(mm[i]), int(ss[i]), 0, 0, 0)
        ctime.append(time.mktime(date) - time_zone*3600)
    return np.array(ctime)

def HHMMSS2hms(hms):
    HH = (hms - hms % 10000) / 10000
    mm = hms - HH * 10000
    MM = (mm - mm % 100) / 100
    SS = mm % 100
    return HH, MM, SS

def correlate_normalized(vec1, vec2, remove_mean=True):
    assert len(vec1) == len(vec2)
    v1 = vec1.copy()
    v2 = vec2.copy()
    if remove_mean:
        v1 -= v1.mean()
        v2 -= v2.mean()
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    corr = np.dot(v1, v2) / norm1 / norm2
    return corr

def rtvec2enu(rvec, tvec):
    euler = []
    enu = []
    for rv, tv in zip(rvec, tvec):
        euler.append(rotvec2euler(rv))
        enu.append(tvec2enu(rv, tv))
    return np.array(euler), np.array(enu)

def rotvec2euler(rvecs):
    R = cv.Rodrigues(rvecs)[0]
    ox, oy, oz = matrix2euler(R)
    euler_deg = [ox * 180 / np.pi, oy * 180 / np.pi, oz * 180 / np.pi]
    return euler_deg

def tvec2enu(rotation_vector, translation_vector):
    Rt = cv.Rodrigues(rotation_vector)[0]
    pos = -np.dot(Rt.T,translation_vector)
    return pos

def get_attitude_single(rvec, t_enu, points_2D, points_3D, cam_matrix, dist_coeff):
    res = so.minimize(proj_residuals, rvec,
                      args=(t_enu, points_2D, points_3D, cam_matrix, dist_coeff),method='L-BFGS-B')#, method='L-BFGS-B', options={'gtol': 1e-6, 'disp': False}
    return res

# def proj_residuals(rvec, t_enu, points_2D, points_3D, cam_matrix):
#     R = cv.Rodrigues(rvec)[0]
#     dx_enu = points_3D - t_enu[np.newaxis, :]
#     x_cam = np.dot(R, dx_enu.T).T
#     proj = (np.dot(cam_matrix, x_cam.T).T / x_cam[:, 2][:, np.newaxis])[:,:2]
#     res = proj - np.flip(np.array(points_2D), 1)
#     return np.sum(res**2)

def proj_residuals(rvec, t_enu, points_2D, points_3D, cam_matrix, dist_coeff):
    proj= cv.projectPoints(points_3D, rvec, t_enu, camera_matrix, dist_coeff)[0]
    proj = proj.reshape(proj.shape[0],proj.shape[-1])
    res = proj - np.array(points_2D)
    return np.sum(res**2)

def yaw_pitch_roll(Xt, R, trc=None, Xc=None):
    """
    Obtains yaw, pitch and roll from camera-POI line of sight.
    The camera position and orientation are parametrized by the transformation: Xc = R Xw + t,
    where Xw are the world coordinates (ENU) of a point and Xc are its corresponding camera coordinates
    Params:
    Xt: POI (telescope) coordinates in world system
    R: camera rotation matrix
    trc: camera translation vector in camera system
    Xc: camera coordinates in world system
    """
    # Get camera coordinates
    if Xc is None:
        Trc = Transform(R=R, tr=trc)
        Xc = ~Trc * np.zeros(3)

    # Obtain director vectors for line of sight system (0)
    Z0 = (Xt - Xc)
    Z0 /= np.linalg.norm(Z0)
    X0 = np.cross(np.array([0, 0, 1]), Z0)
    X0 /= np.linalg.norm(X0)
    Y0 = np.cross(Z0, X0)
    Y0 /= np.linalg.norm(Y0)

    # Define line of sight system (0) in world coordinates
    S0w = np.array([X0, Y0, Z0])
    # Define camera system (c) in world coordinates, based in the rotation matrix
    Scw = R
    # Obtain the camera system (c) in line of sight coordinates (0) by projecting to it
    Sc0 = np.dot(Scw, S0w.T)

    # Horizontal roll: defined as the rotation around Z0 that puts the axis Xcw perpendicular to Y0
    roll_h = np.arctan2(-Sc0[0, 1], Sc0[0, 0])
    # Vertical roll: defined as the rotation around Z0 that puts the axis Ycw perpendicular to X0
    roll_v = np.arctan2(Sc0[1, 0], Sc0[1, 1])
    # Pitch: defined as the rotation around X0 that puts the axis Zcw perpendicular to Y0
    pitch = np.arctan2(Sc0[2, 1], Sc0[2, 2])
    # Yaw: defined as the rotation around Y0 that puts the axis Zcw perpendicular to X0
    yaw = np.arctan2(Sc0[2, 0], Sc0[2, 2])

    return yaw, pitch, roll_v, roll_h



########################################################################################################################################

    # GET FRAMES

########################################################################################################################################

if videocapture:

    fname=video_path + video_name
    frame_count=0
    cap = cv.VideoCapture(fname)
    while cap.isOpened():
        frame_count+=1
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv.imwrite(outpath+'frames/frame_'+str(frame_count)+'.jpg', frame)
        if frame_count%100==0:
            print('frame %1.0f of frames 23670' % frame_count )

        if cv.waitKey(1) == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

# collect the frames with the glob.glob function
imgs=sorted(glob.glob('C:/Users/PC/Desktop/C0003/frames/'+'*.jpg'), key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

if get_init_coord:

    # first frame target coordinates

    im = cv.imread(imgs[start_frame])[450:2150,800:2250]
    plt.figure(figsize=(10,10))
    plt.title('Write the initial target coordinates:')
    plt.imshow(im)
    plt.show()


########################################################################################################################################

    # SET UP MAIN FUNCTION

########################################################################################################################################

zeroPoint = POI

points_3D = lonlat2enu(zeroPoint, gpsArr)

img_guess=cv.imread(imgs[start_frame], cv.IMREAD_GRAYSCALE)[450:2150,800:2250]



########################################################################################################################################

    # TARGET FINDER (ORB)

########################################################################################################################################

if ORB:

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


    for im in tqdm(imgs[start_frame+1:end_frame]):
        q+=1
        temp = cv.imread(im)[450:2150,800:2250]
        #temp = cv.resize(temp, (temp.shape[1]//3,temp.shape[0]//3), interpolation=cv.INTER_CUBIC)
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
                
            
            
            #plt.savefig("C:/Users/PC/Desktop/C0003/output/img_1_tg"+str(t)+'.png')
        #img3 = cv.drawMatches(img_guess,kp1,img_new,kp2,dmatches[:50],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        #cv.imwrite('C:/Users/PC/Desktop/TESI_MAGISTRALE/TOCO/box/temp_frame_'+str(q)+'.png', temp)
        #cv.imwrite('C:/Users/PC/Desktop/TESI_MAGISTRALE/test_imgs/img3/img3_frame_'+str(q)+'.png', img3)
        
        targets=np.array(C[-len(targets):]).reshape(-1,2)  

        img_guess = copy.copy(img_new)
        count += 1


    plt.figure(figsize=(10,10))
    plt.imshow(temp)

    if savefig:
        plt.savefig(outpath+'target_last_frame.png')



########################################################################################################################################

    # SAVE RESULTS OR IMPORT ORB OUTPUT

########################################################################################################################################


length=(end_frame-start_frame)-1

if not ORB:
    targetList=np.load('C:/Users/PC/Desktop/C0003/targets_coord_'+str(video_name)+'.npy')
    

else:
    # if already exist the npy file with the coords
    targetList=np.array(C)
    np.save('C:/Users/PC/Desktop/C0003/targets_coord_'+str(video_name)+'.npy', targetList1)

    # SAVE ECSV FILE 2D COORDINATES OF TARGETS
    position = Table()

    tg_name = ['target_1','target_2','target_3','target_4','target_6','target_7','target_8','target_9','target_10','target_11','target_12','target_13','target_14','target_15','target_16','target_17','target_18','target_19']

    frame_name=[]
    for j in range(start_frame+1,end_frame,1):
        for i in range(len(targets)):
            frame_name.append(j)


    position.meta['settings'] = ['date= 20230222','zeroPoint= [-22.95978516, -67.787224, 5136.793125]','start frame='+str(start_frame), 'end_frame='+str(end_frame), 'reference frame origini = TOP-LEFT CORNER']
    position['targetName'] = np.array([np.concatenate(tg_name, axis=None) for i in range(0,length,1)]).reshape(-1,)
    position['frame n°']= np.array(frame_name)
    position['x'] = targetList1[:,0]
    position['y'] = targetList1[:,1]

    position['LAT_GPS'] = np.array([np.concatenate(gpsArr[:,0], axis=None) for i in range(0,length,1)]).reshape(-1,)
    position['LON_GPS'] = np.array([np.concatenate(gpsArr[:,1], axis=None) for i in range(0,length,1)]).reshape(-1,)
    position['HEIGHT_GPS'] = np.array([np.concatenate(gpsArr[:,2], axis=None) for i in range(0,length,1)]).reshape(-1,)

    position.write(outpath+'C003_tgs_coords.ecsv',overwrite=True)  # ascii=True

########################################################################################################################################

    # GET 2D POINTS IN THE RIGHT FORMAT

########################################################################################################################################

points_2D=np.zeros([length,len(targets),2])

for i in range(length):
    points_2D[i,:,:] = (targetList[i*len(targets):len(targets)+(i*len(targets))])

points_2D[:,:,0]=points_2D[:,:,0]+800 #y
points_2D[:,:,1]=points_2D[:,:,1]+450 #x

########################################################################################################################################

    # CHECK ORB RESULT

########################################################################################################################################

# listFrame=np.linspace(0,8858,8)
# i=1

# plt.figure(figsize=(40,40))
# for lf in (listFrame):
   
#     lf=int(lf)
#     temp=cv.imread(imgs[lf+8000])
#     xx= points_2D[lf,:,1]
#     yy=points_2D[lf,:,0]
#     for x,y in zip(xx,yy):
#         cv.drawMarker(temp, (int(x), int(y)),(0,255,0), markerSize=20, markerType=cv.MARKER_SQUARE, thickness=2, line_type=cv.LINE_AA)
    
#     plt.subplot(3,3,i)
#     #plt.imshow(temp)
#     plt.savefig(outpath+'test_p2D.png')
#     i+=1
# plt.show()

########################################################################################################################################

    # GET ROTATION AND TRANSLATION VECTOR (ENU)

########################################################################################################################################

rotate=True
guess = True
nrOFF=True
opt_useP3P=False

print('\n computing rotation and traslation vectors')

"""
outputs, in order:
all_rvec: rodrigues rotation vector array for all frames
all_pyr: pitch yaw roll array for all arrays
translation vector (in the camera coord system)
translation vector (in the enu coord system)
"""

# Initialize guess vectors, possible to expand on this option
rotation_vector = translation_vector = False

frames = len(points_2D)

# Create arrays to be populated
# rot1 and trans1 are the raw vectors returned by the PnP solver
# rot2 and trans2 are from Felipe's code to transform those into a coordinate system
# that isolates pitch, roll, yaw of the camera, need to confirm this works as intended
# naive roll is using targets on the grid and assuming grid is perpendicular to camera
# to get a very rough estimate of the roll parameter as a sanity check

rv = []
tv = []
euler=[]

for i in tqdm(range(frames)):
    tList = np.array(points_2D[i])

    # Swap coords x and y so it works with the solver
    # newPC = np.zeros(tList.shape)
    # newPC[:, 0] = tList[:, 1]
    # newPC[:, 1] = tList[:, 0]

    # Remove any points that couldn't be found
    # if np.any(tList==0):
    tList = tList[tList[:, 0] != 0] #newPC = newPC[tList[:, 0] != 0]
    points_3D = points_3D[tList[:, 0] != 0].copy()

    # Should be able to choose whether we want to guess or not, if yes, we can use the previous solution as the
    # input to the subsequent solution. If guess true, it will have to get the first solution without an input
    if not opt_useP3P:
        if np.any(rotation_vector) and np.any(translation_vector) and True:
            rvec_i = rotation_vector.copy()
            tvec_i = translation_vector.copy()
            success, rotation_vector, translation_vector, inliers = cv.solvePnPRansac(points_3D, tList.astype("double"),
                                                                        camera_matrix,
                                                                        dist_coeff, rvec_i, tvec_i,
                                                                        useExtrinsicGuess=True,reprojectionError=10.,flags=cv.SOLVEPNP_ITERATIVE)
        else:
            success, rotation_vector, translation_vector, inliers = cv.solvePnPRansac(points_3D, tList.astype("double"),
                                                                        camera_matrix,
                                                                        dist_coeff,reprojectionError=10.,flags=cv.SOLVEPNP_ITERATIVE)

    elif opt_useP3P:  # P3P requires exactly 4 3d-2d point correspondences
        # by default P3P will use the last 4 targets in the array, the selection algorithm can be improved if
        # needed
        if len(points_3D) > 4:
            P = points_3D[-4:, :]
            NP = newPC.astype("double")[-4:, ]
        elif len(points_3D) == 4:
            P = points_3D
            NP = newPC.astype("double")
        elif len(points_3D) < 4:
            raise Exception('Not enough points to compute P3P (<4)')
        if np.any(rotation_vector) and np.any(translation_vector) and guess:
            rvec_i = rotation_vector.copy()
            tvec_i = translation_vector.copy()
            success, rotation_vector, translation_vector = cv.solvePnP(P, NP, camera_matrix,
                                                                        dist_coeff, rvec_i, tvec_i,
                                                                        useExtrinsicGuess=guess,
                                                                        flags=cv.SOLVEPNP_P3P)
        else:
            success, rotation_vector, translation_vector = cv.solvePnP(P, NP, camera_matrix,
                                                                        dist_coeff,
                                                                        flags=cv.SOLVEPNP_P3P)

    # If you get a successful fit, add it to array. Probably need better error handling if it doesn't...
    if success:
        if np.any(rotation_vector > 360):
            print(rotation_vector)
            break
        rv.append(rotation_vector)
        tv.append(translation_vector)
        euler.append(rotvec2euler(rotation_vector))
    else:
        raise Exception("solvePnP Unsuccessful")

rvec = np.array(rv).squeeze()
tvec = np.array(tv).squeeze()

if savefig:
    rot2=np.array(euler)

    plt.figure(figsize=(15,6))
    plt.subplot(131)
    plt.title('Euler1')
    plt.ylabel('deg')
    plt.plot(rot2[:,0])
    plt.subplot(132)
    plt.title('Euler2')
    plt.plot(rot2[:,1])
    plt.xlabel('frames')
    plt.subplot(133)
    plt.title('Euler3')
    plt.plot(rot2[:,2])

    plt.savefig(outpath+'euler_angles.png')

########################################################################################################################################

    # GPS ALLINIAMENT

########################################################################################################################################

# rvec = rvec[:,:3]
# tvec = tvec[:,3:]

print('\n computing max correlation index')

dd = droneData(gps_csv_path, fields=None)

coords_source = np.array([dd.data[:,0]*u.deg,
                            dd.data[:,1]*u.deg,
                            dd.data[:,2]*u.meter]).T
                            
gps_enu1 = lonlat2enu(np.array(zeroPoint), coords_source)

photo_euler,photo_enu= rtvec2enu(rvec, tvec)
N = len(photo_enu)

ct_ini = dd.ct[0]
ct_end = dd.ct[-1]

inter = interp1d(dd.ct, gps_enu1, axis=0) #fill_value='extrapolate'
t_gps1 = np.linspace(ct_ini, ct_end, int(np.round((ct_end-ct_ini)*frame_rate)) + 1)
gps_enu_interp = inter(t_gps1)

# Obtain correlation between ENU coordinates from GPS and photogrammetry
corr_e = []
corr_n = []
corr_u = []
for i in range(len(t_gps1)-N):
    corr_e.append(correlate_normalized(gps_enu_interp[i:i + N, 0], photo_enu[:, 0], remove_mean=True))
    corr_n.append(correlate_normalized(gps_enu_interp[i:i + N, 1], photo_enu[:, 1], remove_mean=True))
    corr_u.append(correlate_normalized(gps_enu_interp[i:i + N, 2], photo_enu[:, 2], remove_mean=True))
corr_e = np.array(corr_e)
corr_n = np.array(corr_n)
corr_u = np.array(corr_u)
corr = (corr_e + corr_n + corr_u) / 3

# Choose the best alignment as the time of maximum correlation
idxn = int(np.argmax(corr))


t_gps=t_gps1[idxn:idxn+N]
gps_enu=gps_enu_interp[idxn:idxn+N]

print(idxn)

########################################################################################################################################

    # GET ATTITUDE

########################################################################################################################################

for rv in rvec:
    rotMat = cv.Rodrigues(rv)[0]
    gps_enu_cam = np.dot(rotMat,gps_enu.T)

print('\n Computing corrected rotation vector')

res = []
rvec_new = []
for rv, t, p2D in tqdm(zip(rvec, gps_enu_cam.T, points_2D)):
  
    res.append(get_attitude_single(rv, -t, p2D, points_3D, camera_matrix, dist_coeff))
    rvec_new.append(res[-1].x)

rvec_new, fit = np.array(rvec_new), np.array(res)

for i in range(len(fit)):
    if fit[i].success==False:
        print(i)
        rvec_new[i]=rvec[i]

########################################################################################################################################

    # CAMERA YAW PITCH ROLL

########################################################################################################################################

ypr_cam=[]

for rv,ge in zip(rvec_new,gps_enu):
    #rv=rv.reshape(3,1)
    #ge=np.flip(ge)
    Rt=cv.Rodrigues(-rv)[0]
    ypr_cam.append(yaw_pitch_roll([0.,0.,0.], Rt, trc=None, Xc=ge)) #zeroPoint_enu=(0,0,0)

ypr_cam=np.array(ypr_cam)*(180/np.pi)

if savefig:
    plt.figure(figsize=(15,6))
    plt.subplot(141)
    plt.title('yaw')
    plt.ylabel('deg')
    plt.plot(ypr_cam[:,0])
    plt.subplot(142)
    plt.title('pitch')
    plt.plot(ypr_cam[:,1])
    plt.xlabel('frames')
    plt.subplot(143)
    plt.title('roll_v')
    plt.plot(ypr_cam[:,2])
    plt.subplot(144)
    plt.title('roll_h')
    plt.plot(ypr_cam[:,3])

    plt.savefig(outpath+'yaw_pitch_roll.png')
    
plt.show()

sys.exit()
########################################################################################################################################

    # COMPUTE ERRORS (BOOTSTRAP)

########################################################################################################################################


if bootstrap:
    print('\n bootstrap ...')

    ypr_res=[]
    n_res=100

    for k in tqdm(range(n_res)):

        idx=np.random.choice(18, size=10, replace=False)
        pt2D=points_2D[:,idx]
        pt3D=points_3D[idx]



        rotate=True
        guess = True
        nrOFF=True
        opt_useP3P=False


        # outputs, in order:
        # all_rvec: rodrigues rotation vector array for all frames
        # all_pyr: pitch yaw roll array for all arrays
        # translation vector (in the camera coord system)
        # translation vector (in the enu coord system)


        # Initialize guess vectors, possible to expand on this option
        rotation_vector = translation_vector = False

        frames = len(pt2D)

        # Create arrays to be populated
        # rot1 and trans1 are the raw vectors returned by the PnP solver
        # rot2 and trans2 are from Felipe's code to transform those into a coordinate system
        # that isolates pitch, roll, yaw of the camera, need to confirm this works as intended
        # naive roll is using targets on the grid and assuming grid is perpendicular to camera
        # to get a very rough estimate of the roll parameter as a sanity check

        rv = []
        tv = []


        for i in (range(frames)):
            tList = np.array(pt2D[i])

            # Swap coords x and y so it works with the solver
            newPC = np.zeros(tList.shape)
            newPC[:, 0] = tList[:, 1]
            newPC[:, 1] = tList[:, 0]

            # Remove any points that couldn't be found
            # if np.any(tList==0):
            newPC = newPC[tList[:, 0] != 0]
            pt3D = pt3D[tList[:, 0] != 0].copy()

            # Should be able to choose whether we want to guess or not, if yes, we can use the previous solution as the
            # input to the subsequent solution. If guess true, it will have to get the first solution without an input
            if not opt_useP3P:
                if np.any(rotation_vector) and np.any(translation_vector) and guess:
                    rvec_i = rotation_vector.copy()
                    tvec_i = translation_vector.copy()
                    success, rotation_vector, translation_vector = cv.solvePnP(pt3D, newPC.astype("double"),
                                                                                camera_matrix,
                                                                                dist_coeff, rvec_i, tvec_i,
                                                                                useExtrinsicGuess=guess, flags=0)
                else:
                    success, rotation_vector, translation_vector = cv.solvePnP(pt3D, newPC.astype("double"),
                                                                                camera_matrix,
                                                                                dist_coeff, flags=0)

            elif opt_useP3P:  # P3P requires exactly 4 3d-2d point correspondences
                # by default P3P will use the last 4 targets in the array, the selection algorithm can be improved if
                # needed
                if len(pt3D) > 4:
                    P = pt3D[-4:, :]
                    NP = newPC.astype("double")[-4:, ]
                elif len(pt3D) == 4:
                    P = pt3D
                    NP = newPC.astype("double")
                elif len(pt3D) < 4:
                    raise Exception('Not enough points to compute P3P (<4)')
                if np.any(rotation_vector) and np.any(translation_vector) and guess:
                    rvec_i = rotation_vector.copy()
                    tvec_i = translation_vector.copy()
                    success, rotation_vector, translation_vector = cv.solvePnP(P, NP, camera_matrix,
                                                                                dist_coeff, rvec_i, tvec_i,
                                                                                useExtrinsicGuess=guess,
                                                                                flags=cv.SOLVEPNP_P3P)
                else:
                    success, rotation_vector, translation_vector = cv.solvePnP(P, NP, camera_matrix,
                                                                                dist_coeff,
                                                                                flags=cv.SOLVEPNP_P3P)

            # If you get a successful fit, add it to array. Probably need better error handling if it doesn't...
            if success:
                if np.any(rotation_vector > 360):
                    print(rotation_vector)
                    break
                rv.append(rotation_vector)
                tv.append(translation_vector)
                #euler.append(rotvec2euler(rotation_vector))
            else:
                raise Exception("solvePnP Unsuccessful")

        rvec = np.array(rv).squeeze()
        tvec = np.array(tv).squeeze()

        # rvec = rvec[:,:3]
        # tvec = tvec[:,3:]

        dd = droneData(gps_csv_path, fields=None)

        coords_source = np.array([dd.data[:,0]*u.deg,
                                    dd.data[:,1]*u.deg,
                                    dd.data[:,2]*u.meter]).T
                                    
        gps_enu1 = lonlat2enu(np.array(zeroPoint), coords_source)

        photo_euler,photo_enu= rtvec2enu(rvec, tvec)
        N = len(photo_enu)

        ct_ini = dd.ct[0]
        ct_end = dd.ct[-1]

        inter = interp1d(dd.ct, gps_enu1, axis=0)#fill_value='extrapolate'
        t_gps1 = np.linspace(ct_ini, ct_end, int(np.round((ct_end-ct_ini)*frame_rate)) + 1)
        gps_enu_interp = inter(t_gps1)

        # Obtain correlation between ENU coordinates from GPS and photogrammetry
        corr_e = []
        corr_n = []
        corr_u = []
        for i in range(len(t_gps1)-N):
            corr_e.append(correlate_normalized(gps_enu_interp[i:i + N, 0], photo_enu[:, 0], remove_mean=False))
            corr_n.append(correlate_normalized(gps_enu_interp[i:i + N, 1], photo_enu[:, 1], remove_mean=False))
            corr_u.append(correlate_normalized(gps_enu_interp[i:i + N, 2], photo_enu[:, 2], remove_mean=False))
        corr_e = np.array(corr_e)
        corr_n = np.array(corr_n)
        corr_u = np.array(corr_u)
        corr = (corr_e + corr_n + corr_u) / 3

        # Choose the best alignment as the time of maximum correlation
        idxn = int(np.argmax(corr))


        t_gps=t_gps1[idxn:idxn+N]
        gps_enu=gps_enu_interp[idxn:idxn+N]

        ypr=[]

        for rv,ge in zip(rvec_new,gps_enu):
            rv=rv.reshape(3,1)
            #ge=np.flip(ge)
            Rt=cv.Rodrigues(rv)[0]
            ypr.append(yaw_pitch_roll(zeroPoint, Rt, trc=None, Xc=ge)) 

        ypr=np.array(ypr)*(180/np.pi)

        ypr_res.append(ypr)
            
    ypr_res=np.array(ypr_res)


    ypr_res_std_yaw=np.std(ypr_res[:,:,0])
    ypr_res_std_pitch=np.std(ypr_res[:,:,1])
    ypr_res_std_rv=np.std(ypr_res[:,:,2])
    ypr_res_std_rh=np.std(ypr_res[:,:,3])


    print(' mean bootstrap error on yaw =' , ypr_res_std_yaw)
    print(' mean bootstrap error on pitch =' , ypr_res_std_pitch)
    print(' mean bootstrap error on vertical roll =' , ypr_res_std_rv)
    print(' mean bootstrap error on horizontal roll =' , ypr_res_std_rh)


########################################################################################################################################

    # COMPUTE ERROR (JACKKNIFE)

########################################################################################################################################

# WORK IN PROGRSS

########################################################################################################################################

    # SAVE ATTITUDE

########################################################################################################################################

rt= Table()
rt['roll_v'] = ypr_cam[:,2]
rt['rollv_err'] = np.array([ypr_res_std_rv for i in range(len(ypr_cam[:,0]))])
rt['roll_h'] = ypr_cam[:,2]
rt['rollh_err'] = np.array([ypr_res_std_rh for i in range(len(ypr_cam[:,0]))])
rt['pitch'] = ypr_cam[:,1]
rt['pitch_err']=np.array([ypr_res_std_pitch for i in range(len(ypr_cam[:,0]))])
rt['yaw'] = ypr_cam[:,0]
rt['yaw_err']=np.array([ypr_res_std_yaw for i in range(len(ypr_cam[:,0]))])


rt.write(outpath+'RotTrans.ecsv',overwrite=True)

