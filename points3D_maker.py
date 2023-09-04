import numpy as np
from astropy import units as u

POI = [-67.78724760, -22.95974877, 5135.5229]

# in format [lon, lat, alt]
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
[-67.78674781,-22.96111829,5128.1856]]) #19

### FUNCTIONS ##################################################################################################
def ellipsoid(model='WGS84'):

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

    poi_x, poi_y, poi_z, _, _, _ = lonlat2ecef(lonlat_target[0], lonlat_target[1], lonlat_target[2])
    
    enu_arr = np.zeros(lonlat_ccords.shape)
    
    x, y, z, _, _, _ = lonlat2ecef(lonlat_ccords[:,0], lonlat_ccords[:,1], lonlat_ccords[:,2])
    #enu_arr[:,0], enu_arr[:,1], enu_arr[:,2], _, _, _ = ecef2enu(x, y, z, poi_x, poi_y, poi_z, 0,0,0,0,0,0,lonlat_target[0], lonlat_target[1])
    enu_arr[:,0], enu_arr[:,1], enu_arr[:,2], _, _, _ = ecef2enu(poi_x, poi_y, poi_z, x, y, z, 0,0,0,0,0,0,lonlat_target[0], lonlat_target[1])
    
    return enu_arr 
################################################################################################################

### convert points from coordinates to enu ###
points_3D = lonlat2enu(POI, gpsArr)
print(np.array2string(points_3D, separator=", "))
