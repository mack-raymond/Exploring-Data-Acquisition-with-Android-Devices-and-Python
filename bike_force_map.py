# import modules
import utm
import pandas as pd
import numpy as np
import scipy as sp
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d
from numpy import sin,cos,pi
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# Rotation matrices
def R_x(x):
    """body frame rotation about x axis"""
    return np.array([[1,      0,       0],
                     [0,cos(-x),-sin(-x)],
                     [0,sin(-x), cos(-x)]])

def R_y(y):
    """body frame rotation about y axis"""
    return np.array([[cos(-y),0,-sin(-y)],
                    [0,      1,        0],
                    [sin(-y), 0, cos(-y)]])

def R_z(z):
    """body frame rotation about z axis"""
    return np.array([[cos(-z),-sin(-z),0],
                     [sin(-z), cos(-z),0],
                     [0,      0,       1]])


def transform_accelerations(df,do_plot=False):
    """Transform accelerations from body to earth frame"""

    # Set up arrays to hold acceleration data for transfromation
    accel = np.array([df['ACCELEROMETER X (m/s²)'],
                      df['ACCELEROMETER Y (m/s²)'],
                      df['ACCELEROMETER Z (m/s²)']])
    grav = np.array([df['GRAVITY X (m/s²)'],
                     df['GRAVITY Y (m/s²)'],
                     df['GRAVITY Z (m/s²)']])
    line = np.array([df['LINEAR ACCELERATION X (m/s²)'],
                     df['LINEAR ACCELERATION Y (m/s²)'],
                     df['LINEAR ACCELERATION Z (m/s²)']])

    # Set up arrays to hold euler angles for rotation matrices
    pitch = df['ORIENTATION X (pitch °)']
    roll = df['ORIENTATION Y (roll °)']
    yaw = df['ORIENTATION Z (azimuth °)']

    # Initilize arrays for new transformed accelerations
    earth_accels = np.empty(accel.shape)
    earth_gravity = np.empty(accel.shape)
    earth_linear = np.empty(accel.shape)

    # Perform frame transformations (body frame --> earth frame)
    for i in range(df.shape[0]):
        # accel_earth = (RzRyRx)(accel_body)
        earth_accels[:,i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ accel[:,i]
        earth_gravity[:,i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ grav[:,i]
        earth_linear[:,i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ line[:,i]

    # Create new columns in dataframe for earth frame accelerations
    df['EARTH ACCELERATION X'] = earth_accels[0,:]
    df['EARTH ACCELERATION Y'] = earth_accels[1,:]
    df['EARTH ACCELERATION Z'] = earth_accels[2,:]
    df['EARTH GRAVITY X'] = earth_gravity[0,:]
    df['EARTH GRAVITY Y'] = earth_gravity[1,:]
    df['EARTH GRAVITY Z'] = earth_gravity[2,:]
    df['EARTH LINEAR ACCELERATION X'] = earth_linear[0,:]
    df['EARTH LINEAR ACCELERATION Y'] = earth_linear[1,:]
    df['EARTH LINEAR ACCELERATION Z'] = earth_linear[2,:]

    if do_plot:
    # Plot new accelerations
        cols_earth = ['EARTH ACCELERATION X','EARTH ACCELERATION Y',
                      'EARTH ACCELERATION Z','EARTH GRAVITY X',
                      'EARTH GRAVITY Y','EARTH GRAVITY Z',
                  'EARTH LINEAR ACCELERATION X','EARTH LINEAR ACCELERATION Y',
                  'EARTH LINEAR ACCELERATION Z']
        cols_body = ['ACCELEROMETER X (m/s²)','ACCELEROMETER Y (m/s²)',
                     'ACCELEROMETER Z (m/s²)', 'GRAVITY X (m/s²)',
                     'GRAVITY Y (m/s²)','GRAVITY Z (m/s²)',
                 'LINEAR ACCELERATION X (m/s²)','LINEAR ACCELERATION Y (m/s²)',
                 'LINEAR ACCELERATION Z (m/s²)',]

        bodyplot = df.plot(y=cols_body,subplots=True,sharex=True,layout=(3,3),
                           title=cols_body,style='k',alpha=0.5)

        df.plot(y=cols_earth,subplots=True,layout=(3,3),ax=bodyplot,sharex=True,
                style='g',title='Body Frame to Earth Frame Accelerations')


def convert_latlon(lat, lon, do_plot=False):
    """
    Transforms latitude and longitude onto a 2D plane in meters
    uses UTM coordinates (Universal Transverse Mercator coordinate system).
    Returns numpy array for x and y"""

    easting, northing, zone_letter, zone_number = utm.from_latlon(lat, lon)
    x = easting - easting[0]
    y = northing - northing[0]

    utm_dict = {}
    utm_dict['easting'] = easting
    utm_dict['northing'] = northing
    utm_dict['zone letter'] = zone_letter
    utm_dict['zone number'] = zone_number

    if do_plot:
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(x, y, label='meters')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.axis('scaled')
        ax1.legend()

        ax2.plot(lon, lat, label='degrees')
        ax2.set_xlabel('longitude')
        ax2.set_ylabel('latitude')
        ax2.axis('scaled')
        ax2.legend()
        fig.suptitle('GPS degrees to XY meters (UTM)')

    return x, y, utm_dict

def SimplifyTrack(data_x,data_y,accuracy,speed,dt,do_plot=False):
    """Removes clusters of GPS points and fits a spline to path"""

    t = int(1/dt)
    x = list(data_x[::t])
    y = list(data_y[::t])
    speed = list(speed[::t])
    accuracy = list(accuracy[::t])
    i = 0
    while True:
        knot_size = speed[i]*dt + accuracy[i]/2
        jump = np.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2) # point distance
        if jump <= knot_size: # delete points that smaller than "knot" size
            x.pop(i+1)
            y.pop(i+1)
        elif x[i+1] in x[:i] and y[i+1] in y[:i]: # delete duplicates
            x.pop(i+1)
            y.pop(i+1)
        else:
            i += 1
        if i >= len(x)-1:
            break

    weight = np.mean(accuracy)
    tck,u = sp.interpolate.splprep([x,y],s=weight,k=3)
    num = np.round(1/len(data_x),10)
    newu = np.arange(0,1,num) # same number of points as original data
    out = sp.interpolate.splev(newu, tck)

    if len(out[0]) > len(data_x):
        out[0] = out[0][:-1]
        out[1] = out[1][:-1]

    if do_plot:
        fig,ax = plt.subplots()
        ax.plot(data_x,data_y,label = 'GPS')
        ax.plot(x,y,label = 'Simplified')
        ax.plot(out[0],out[1],label = 'Splined')
        ax.axis('equal')
        plt.legend()

    return out[0],out[1]

# import data from CSV
df = pd.read_csv('C:/Users/mackr/OneDrive/Documents/python_projects/INS/data/Drivewaysnake0.01.csv',
                 header=0)
# Take a look at all sensor outputs
df.plot(subplots=True,sharex=True,layout=(6,6))
dt = 0.01
transform_accelerations(df)

# Clean noise
    # Discrete Fourier Transform sample frequencies
freq = np.fft.rfftfreq(df['EARTH LINEAR ACCELERATION X'].size,d=dt)
    # Compute the Fast Fourier Transform (FFT) of acceleration signals
fft_x = np.fft.rfft(df['EARTH LINEAR ACCELERATION X'])
fft_y = np.fft.rfft(df['EARTH LINEAR ACCELERATION Y'])
fft_z = np.fft.rfft(df['EARTH LINEAR ACCELERATION Z'])

# Plot Frequency spectrum
fig,[ax1,ax2,ax3] = plt.subplots(3,1,sharex=True,sharey=True)
fig.suptitle('Noise Spectrum',fontsize=20)
ax1.plot(freq,abs(fft_x),c='r',label='x noise')
ax1.legend()
ax2.plot(freq,abs(fft_y),c='b',label='y noise')
ax2.legend()
ax3.plot(freq,abs(fft_z),c='g',label='z noise')
ax3.legend()
ax3.set_xlabel('Freqeuncy (Hz)')

# Noise is white. Use gaussian filter to smooth data
sigma = 1/(dt)
df['filtered X'] = gaussian_filter1d(df['EARTH LINEAR ACCELERATION X'],sigma)
df['filtered Y'] = gaussian_filter1d(df['EARTH LINEAR ACCELERATION Y'],sigma)
df['filtered Z'] = gaussian_filter1d(df['EARTH LINEAR ACCELERATION Z'],sigma)

# Check new signals
cols_noisy = ['EARTH LINEAR ACCELERATION X','EARTH LINEAR ACCELERATION Y',
              'EARTH LINEAR ACCELERATION Z']
cols_smooth = ['filtered X','filtered Y','filtered Z']
noise_plot = df.plot(y=cols_noisy,subplots=True,sharex=True,layout=(1,3))
df.plot(y=cols_smooth,subplots=True,sharex=True,layout=(1,3),style='k',ax=noise_plot)

# Plo GPS bike path
lon = df['LOCATION Longitude : '].values
lat = df['LOCATION Latitude : '].values
fig2,[ax5,ax6] = plt.subplots(1,2)
ax5.set_title('Track in lat/lon degrees')
ax5.plot(lon,lat,'ro',lon,lat,'k')
ax5.axis('equal')

# convert to X,Y coordinates
x,y,utm_dict = convert_latlon(lat,lon)
ax6.set_title('Track in XY coords')
ax6.plot(x,y,'bo',x,y,'k')
ax6.axis('equal')

# Smooth track with a Ramer–Douglas–Peucker-like algorithm
    # GPS reciever samples at 1sec
xsmooth,ysmooth = SimplifyTrack(x,y,df['LOCATION Accuracy ( m)'],
                                df['LOCATION Speed ( m/s)'],dt=1,do_plot=False)
ax6.plot(xsmooth,ysmooth,'g',lw=3,label='spline fit')
ax6.legend()

# Map forces to path

fig3, ax7 = plt.subplots()
colors = np.linalg.norm([df['filtered X'][::10],df['filtered Y'][::10]],axis=0)
ax7.quiver(xsmooth[::10],ysmooth[::10],
           df['filtered X'][::10],df['filtered Y'][::10],
           colors,cmap=plt.cm.jet,scale_units='xy',scale=0.1)
ax7.plot(xsmooth,ysmooth,'k',lw=3)
ax7.axis('equal')
ax7.set_title('GPS path with force vectors',fontsize=20)
ax7.set_xlabel('Meters East',fontsize=20)
ax7.set_ylabel('Meters North',fontsize=20)

plt.figure()

plt.scatter(xsmooth,ysmooth,c=df['filtered Z'])
plt.colorbar()
plt.show()
