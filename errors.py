import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from scipy.stats import chisquare
#from scipy.stats import chi2_contingency

################################################################################################################

errors = np.array([100, 50, 30, 10, 5, 1, 0.5, 0.1, -1, -10, -50, -100])

err_eul = np.array([[0.015861943260699098, 	 0.022905803227468886, 	 0.007309040875343341], 
                    [0.007944661401737494, 	 0.011458592903497716, 	 0.003657762130211936],
                    [0.004770100939616987, 	 0.006876542793571148, 	 0.0021954329987674635],
                    [0.0015911385240422302, 	 0.0022926473037298673, 	 0.0007320691403595542],
                    [0.0007957105746690942, 	 0.0011463785550135123, 	 0.0003660554757930855],
                    [0.00015916797154461013, 	 0.00022928915156612131, 	 7.320722357397791e-05],
                    [7.958546559444785e-05, 	 0.00011464517036609738, 	 3.660393043658455e-05],
                    [1.5917315229026042e-05, 	 2.2929128083842486e-05, 	 7.320838108099716e-06],
                    [0.00015917908261557596, 	 0.00022929385224428045, 	 7.320982126643565e-05],
                    [0.0015922468979466587, 	 0.002293117695020852, 	 0.0007323268500833331],
                    [0.007972370380358473, 	 0.011470352604044554, 	 0.0036642049168380477],
                    [0.015972774562499115, 	 0.022952841007451354, 	 0.007334812565820819]
                    ])

err_tenu = np.array([[0.21977134373190027, 	 0.18026470908124445, 	 0.38486351172801025],
                     [0.11000819118390548, 	 0.09033299705613061, 	 0.19247021775571635],
                     [0.06603472587135994, 	 0.05424823578526169, 	 0.11549115305302322],
                     [0.022021589567443475, 	 0.018098944719224543, 	 0.038500019025627462],
                     [0.011012019197646264, 	 0.00905152149356459, 	 0.01925038386666131],
                     [0.002202644662321746, 	 0.0018106656266574563, 	 0.0038501465651583472],
                     [0.0011013349057223613, 	 0.00090535355617661, 	 0.0019250749357017643],
                     [0.00022026899825653733, 	 0.0001810739674590189, 	 0.00038501556887935606],
                     [0.0022027455361735672, 	 0.001810828348289741, 	 0.003850175133914749],
                     [0.022031680928049028, 	 0.018115197543938984, 	 0.03850294773888775 ],
                     [0.11026047171477571, 	 0.09073931173652584, 	 0.19254343967064502],
                     [0.22078042229483263, 	 0.18188989358044247, 	 0.3851564504291461]
                     ])

outpath = '/Users/noemi/Documents/Uni/TESI/SolvePnP/outputs/VAR/'

### FUNCTIONS ##################################################################################################

def line(x, a):
    return a * np.abs(x)

def chi2(o, e):
    num = (o - e)**2
    chi = num / e
    return np.sum(chi) #/ (len(chi) - 1)
                    
### PLOT & FIT #################################################################################################                    

### compute fits ###
par_y, pcov_y = curve_fit(line, errors, err_eul[:,0])
par_p, pcov_p = curve_fit(line, errors, err_eul[:,1])
par_r, pcov_r = curve_fit(line, errors, err_eul[:,2])

par_e, pcov_e = curve_fit(line, errors, err_tenu[:,0])
par_n, pcov_n = curve_fit(line, errors, err_tenu[:,1])
par_u, pcov_u = curve_fit(line, errors, err_tenu[:,2])


### chi square test ###
chi_y = chi2(err_eul[:,0], line(errors, *par_y))
chi_p = chi2(err_eul[:,1], line(errors, *par_p))
chi_r = chi2(err_eul[:,2], line(errors, *par_r))

chi_e = chi2(err_tenu[:,0], line(errors, *par_e))
chi_n = chi2(err_tenu[:,1], line(errors, *par_n))
chi_u = chi2(err_tenu[:,2], line(errors, *par_u))

print('Chi-squared test:')
print('\nYaw: ', chi_y)
print('\nPitch: ', chi_p)
print('\nRoll: ', chi_r)
print('\n')
print('\nEast: ', chi_e)
print('\nNoth: ', chi_n)
print('\nUp: ', chi_u)


### rotation ###
plt.figure(figsize=(18,5))
plt.suptitle('RMS of the euler angles', fontsize='14')
#yaw
plt.subplot(131)
l1, = plt.plot(errors, line(errors, *par_y), color='navajowhite', label='a = '+str(np.format_float_scientific(par_y[0],3))+' $\pm$ '+str(np.format_float_scientific(np.sqrt(pcov_y[0]),0)))
plt.scatter(errors, err_eul[:,0], color='orange')
#plt.xlabel('distorsion coefficients error (%)')
plt.ylabel('root mean square (deg)')
plt.title('Yaw')
plt.legend(handles=[l1], loc=9)
# pitch
plt.subplot(132)
l2, = plt.plot(errors, line(errors, *par_p), color='peachpuff', label='a = '+str(np.format_float_scientific(par_p[0],4))+' $\pm$ '+str(np.format_float_scientific(np.sqrt(pcov_p[0]),0)))
plt.scatter(errors, err_eul[:,1], color='coral')
plt.xlabel('distorsion coefficients error (%)')
#plt.ylabel('root mean square (deg)')
plt.title('Pitch')
plt.legend(handles=[l2], loc=9)
# roll
plt.subplot(133)
l3, = plt.plot(errors, line(errors, *par_r), color='pink', label='a = '+str(np.format_float_scientific(par_r[0],3))+' $\pm$ '+str(np.format_float_scientific(np.sqrt(pcov_r[0]),0)))
plt.scatter(errors, err_eul[:,2], color='crimson')
#plt.xlabel('distorsion coefficients error (%)')
#plt.ylabel('root mean square (deg)')
plt.title('Roll')
plt.legend(handles=[l3], loc=9)
# save and show
plt.savefig(outpath+'euler_errors.png', bbox_inches='tight')
plt.show()


### translation ###
plt.figure(figsize=(18,5))
plt.suptitle('RMS of the drone position in ENU coordinates', fontsize='14')
# east
plt.subplot(131)
l4, = plt.plot(errors, line(errors, *par_e), color='paleturquoise', label='a = '+str(np.format_float_scientific(par_e[0],4))+' $\pm$ '+str(np.format_float_scientific(np.sqrt(pcov_e[0]),1)))
plt.scatter(errors, err_tenu[:,0], color='c')
#plt.xlabel('distorsion coefficients error (%)')
plt.ylabel('root mean square (m)')
plt.title('East')
plt.legend(handles=[l4], loc=9)
# north
plt.subplot(132)
l5, = plt.plot(errors, line(errors, *par_n), color='lightsteelblue', label='a = '+str(np.format_float_scientific(par_n[0],3))+' $\pm$ '+str(np.format_float_scientific(np.sqrt(pcov_n[0]),0)))
plt.scatter(errors, err_tenu[:,1], color='royalblue')
plt.xlabel('distorsion coefficients error (%)')
#plt.ylabel('root mean square (m)')
plt.title('North')
plt.legend(handles=[l5], loc=9)
# up
plt.subplot(133)
l6, = plt.plot(errors, line(errors, *par_u), color='thistle', label='a = '+str(np.format_float_scientific(par_u[0],4))+' $\pm$ '+str(np.format_float_scientific(np.sqrt(pcov_u[0]),0)))
plt.scatter(errors, err_tenu[:,2], color='mediumpurple')
#plt.xlabel('distorsion coefficients error (%)')
#plt.ylabel('root mean square (m)')
plt.title('Up')
plt.legend(handles=[l6], loc=9)
# save and show
plt.savefig(outpath+'t_enu_errors.png', bbox_inches='tight')
plt.show()
