import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get data from csv, set dt, time
file ="C:/Users/mackr/OneDrive/Documents/python_projects/MITRE/data/clean_3_1m_lifts.csv"
df = pd.read_csv(file, sep=',', header=0)
dt = 0.01

# State Transition Matrix
A = np.array([[1,dt],
              [0, 1]])

# Control Matrix
B = np.array([[0.5*dt**2],
              [dt]])

# Control vector
u = np.array([[0]])

# Initial Process Covariance Matrix
P = np.array([[10,0],
              [0,10]])

# Measurement Matrix
H = np.array([[1,0]])

# Measurement noise covariance matrix
R = np.array([[0.25]])

# Initial System State Matrix (pos_z = 0, vel_z = 0 at t = 0)
X = np.array([[0],
              [0]])

I = np.eye(2)

X_pos = []
X_vel = []
P_pos = []
P_vel = []
K_pos = []
K_vel = []

for i in range(df.shape[0]):

    # Pull in z acceleration control input
    u = np.array([[df['z_ifft'][i]]])

    # Predict the next state
    X = A @ X + B @ u
    P = A @ P @ A.T
    P = np.tril(np.triu(P, k=0), k=0) # set off diag elements to zero

    # Altitude measurement once every 15 accelerometer updates (100Hz vs 66Hz)
    if i % 15 == 0:

        # Pull in altitude measurement
        z = np.array([[df['LOCATION Altitude-atmospheric pressure ( m)'][i]]])

        # Update the next state
        K = P @ H.T / (H @ P @ H.T + R) # Kalman Gain
        X = X + K @ (z - H @ X)         # updated State
        P = (I - K @ H) @ P             # Updated Process Covariance
        P = np.tril(np.triu(P, k=0), k=0)

    # --- Store system states ---
    X_pos.append(X[0][0])
    X_vel.append(X[1][0])
    P_pos.append(P[0][0])
    P_vel.append(P[1][0])
    K_pos.append(K[0][0])
    K_vel.append(K[1][0])

df['Z_Position'] = X_pos

from scipy.integrate import cumtrapz

Z = cumtrapz(cumtrapz(df['z_ifft'],dx=dt),dx=dt)
fig,axs = plt.subplots(nrows=2)

# Estimating Altitude Plot
axs[0].set_title('Estimating Altitude',fontsize=20)
axs[0].plot(X_pos,label = 'System State Position')
axs[0].plot(Z,label = 'Double Integrate Z accelerometer')
axs[0].plot(df['LOCATION Altitude-atmospheric pressure ( m)'][::15],
            'r.',lw=5,
            label='Barometer')
axs[0].plot(df['z_ifft'],label='Acceleration (m/s/s)',
            c='k',alpha=0.1)
axs[0].set_ylabel('Altitude (m)',fontsize=15)
axs[0].set_xlabel('Data Point (index)',fontsize=15)

axs[0].axhline(y=1,c='k',label ='Truth Max Altitude')
axs[0].legend(fontsize=15)

# Kalman Gain and Covariances
axs[1].set_title('Kalman Gain and Process covariance for state variables',
                 fontsize = 20)
axs[1].set_xlabel('Data Point (index)',fontsize=15)
axs[1].plot(K_pos,label='Kalman Gain for Position')
axs[1].plot(K_vel,label='Kalman Gain for Velocity')
axs[1].plot(P_pos,label='Covariance for Position')
axs[1].plot(K_vel,label='Covariance for Velocity')
axs[1].legend(fontsize=15)


