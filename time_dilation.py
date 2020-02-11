import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)

# CONTANTS
c = 2.998 * 10**8    # Speed of light
M_e = 5.98 * 10**24  # Mass of the Earth
G = 6.673 * 10**-11  # Gravitational constant
r_e = 6357 * 1000    # Radius of the Earth (m)
r_s = 26541  * 1000   # Medium Earth Orbit for GPS satellites
clock_freq = 10.23 * 10**6 # Hz
us_per_day = 10**6 * 3600 * 24  # micro seconds per day

# Arrays
sec_in_hour = np.arange(0,3600)
R = np.linspace(r_e,r_s) # Radius values from Earth surface to satellite orbit
time = np.linspace(0,10**-7,1000)

def orbital_velocity(r):
    return np.sqrt(G*M_e/r)

def lorentz_factor(v):
    return 1/np.sqrt(1-(v/c)**2)

def schwarzschild_factor(r):
    return 1/np.sqrt(1 - 2*G*M_e/(r*c**2))

# Calculate allowed oribital velocities
velocities = orbital_velocity(r=R)

# Calculation dilation factors
dilation_speed = lorentz_factor(v=velocities)
dilation_gravity = schwarzschild_factor(r=R)

# Calculate time gained w.r.t Earth's Surface
# time_gained = t_earth - t_sat
time_gained_speed = us_per_day * (1 - dilation_speed)
time_gained_gravity = us_per_day * (dilation_gravity[0] - dilation_gravity)
net_time_gain = time_gained_gravity + time_gained_speed

# Position error over time
us_gain_per_sec = net_time_gain[-1]/(24*3600)
position_error = us_gain_per_sec/10**6 * sec_in_hour * c

# Clock frequency compensation
freq_dilation_factor = net_time_gain[-1] / us_per_day
adjusted_freq = clock_freq * (1 - freq_dilation_factor)
clock_natural = np.sin(time*2*np.pi*clock_freq)
clock_adjusted = np.sin(time*2*np.pi*adjusted_freq)

# Plotting
R_from_surface = (R - R[0])/1000 # Reset x axis to start at Earth Surface
fig,ax = plt.subplots()
fig.suptitle('Time Dilation Effects on Earth',fontsize=20)
ax.plot(R_from_surface,time_gained_speed, label = 'Orbital Speed Slowdown')
ax.plot(R_from_surface,time_gained_gravity,label = 'Gravity Speedup')
ax.plot(R_from_surface,net_time_gain,label = 'NET Orbital Time Gain')
ax.axvline(x=0,linewidth=4, color='k') # Earth surface
bbox = dict(boxstyle="round",alpha=0.2)
arrowprops = dict(arrowstyle = "->",
                  connectionstyle="angle,angleA=0,angleB=90,rad=0")
ax.annotate('GPS Orbit \n +38\u03BCs',
            xy=(R_from_surface[-1],net_time_gain[-1]),
            xytext=(-100,-100),textcoords='offset points',bbox=bbox,
            arrowprops=arrowprops,fontsize=20)

ax.set_ylabel('Microseconds gained per Earth day',fontsize=20)
ax.set_xlabel('Kilometeres from Earth Surface',fontsize=20)
ax.legend(fontsize=20)

fig2,ax2 = plt.subplots()
fig2.suptitle('Clock frequency compensation',fontsize=20)
ax2.plot(time*10**9,clock_natural, label='Normal Oscillation')
ax2.plot(time*10**9,clock_adjusted, label='Adjusted Oscillation')
ax2.set_xlabel('Nanoseconds')
ax2.legend(loc='lower left',fontsize=20)