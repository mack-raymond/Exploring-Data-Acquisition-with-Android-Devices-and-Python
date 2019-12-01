# import modules
import pandas as pd
import numpy as np
from scipy.integrate import cumtrapz
from numpy import sin,cos,pi
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use('seaborn')


# import data from CSV
df = pd.read_csv('Data/lift 1m.csv')
# Take a look at all sensor outputs
df.plot(subplots=True,sharex=True,layout=(6,6),title=list(df.columns[:-1]),
        legend=False)
dt = 0.01 # Sampling at 100Hz
# Convert orientation units to radians
cols_angles = ['ORIENTATION X (pitch °)','ORIENTATION Y (roll °)',
               'ORIENTATION Z (azimuth °)']
for axis in cols_angles:
    df[axis] = df[axis] * pi/180