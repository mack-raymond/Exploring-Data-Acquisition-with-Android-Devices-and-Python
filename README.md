## Exploring-Data-Acquisition-with-Android-Devices-and-Python
Github page for my Medium.com data science article
- link: https://medium.com/@draymo/exploring-data-acquisition-and-trajectory-tracking-with-android-devices-and-python-9fdef38f25ee

At a glance, we will:
- Download AndroSensor, gather some experiment motion data, and export to a computer.
- Open the .csv file with Pandas to examine our data steams.
- Transform the x,y,z accelerations from the IMU to an inertial Earth frame.
- Double integrate the earth frame accelerations with respect to time to calculate the x,y,z positions of the phone.
- 3D plot the trajectory of the phone, and observe the motion drift from the accelerometer.
- Perform a Fourier analysis of the acceleration signals to examine the noise spectrum in each accelerometer axis.
- Create a high-pass filter to attenuate low frequency noise, and perform a inverse Fourier transform to calculate new accelerations with 
- less noise.
- 3D plot trajectory, and add x,y,z axis vectors to indicate phone pose.

This article is tailored toward Android devices, but steps 2–8 are invariant to the smart phone OS as long as you can export your phone’s sensor outputs.


#
