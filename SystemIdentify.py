# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:37:45 2022
@author: Feng
"""

"""
Introduction: 
    This program is used to model the transfer function of the presumed humidity sensor.
Model:
    
    1.  
        Curve Fitting with Least Squares Method
        https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/generated/scipy.optimize.curve_fit.html#scipy.optimize.curve_fit
    
    2. 
        Non-linear Autoregressive Models with Moving Average and Exogenous Input
        http://sysidentpy.org/introduction_to_narmax.html
    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# Plot Setting
plt.rcParams['figure.dpi'] = 300 # Set the output image resolution
init_humi = 96.24               # Set initial humidity before step
ramp_time = 3                    # Set time length of ramp
   
# Import data
data = pd.read_csv("./07B2_20220217_GC.csv", header=0, index_col=False)

# Humidity at steady state (300s)
termi_humi = np.mean(data["humi(SHT35)"].values[71300:71600])
# Transient Response （60s + ramp + 600s）
humidity_sensor = data["humi(SHT35)"].values[70612:71275]

# True humidity
humidity_true = np.append(np.linspace(init_humi, init_humi, 60),
                    np.linspace(init_humi, termi_humi, ramp_time))
humidity_true = np.append(humidity_true, 
                          np.linspace(termi_humi, termi_humi, 600))

timestamp = np.linspace(0, len(humidity_sensor)-1, len(humidity_sensor))
timestamp = timestamp * 5



# Fit Curve

bias = np.mean(humidity_sensor[-101:-1])    # bias of 
fitdata = humidity_sensor[60:-1]            # transient data for fitting
time = np.arange(0, len(fitdata), 1)        # x axis data


# T -- Time Constant of first order system
def first_order_system(t, T):
    return bias + (init_humi-bias) * np.exp(-t/T)
popt1, povc1 = optimization.curve_fit(first_order_system, xdata=time, ydata=fitdata,
                                      p0=[8])

# T1, T2 -- Parameters of second order system
def second_order_system(t, sigma, wn):
    T1 = 1 / (wn * (sigma - (sigma**2 - 1)**0.5))
    T2 = 1 / (wn * (sigma + (sigma**2 - 1)**0.5))
    return bias + (init_humi-bias)*(T1-T2)/(T1+T2) * (- np.exp(-t/T1) / (T2/T1 - 1) + np.exp(-t/T2) / (T1/T2 - 1))
popt2, povc2 = optimization.curve_fit(second_order_system, xdata=time, ydata=fitdata,
                                      p0=[10, 1])

# System Output
first_order_output = first_order_system(time, popt1[0])
second_order_output = second_order_system(time, popt2[0], popt2[1])


# Plot
plt.Figure()
plt.plot(timestamp[60:-1], humidity_true[60:-1], label="Enviroment")
plt.plot(timestamp[60:-1], humidity_sensor[60:-1], label="sensor")
plt.plot(timestamp[60:-2], first_order_output[0:-1], 'o', markersize=3, label="first_order")
plt.plot(timestamp[60:-2], second_order_output[0:-1],'--', markersize=3, label="second_order")
plt.xlabel("Time / s")
plt.ylabel("Humidity / %")
plt.legend()
plt.title("True humidity(Input) and Sensor(Output)")
plt.show()


plt.Figure()
plt.plot(timestamp[60:120], humidity_true[60:120], label="Enviroment")
plt.plot(timestamp[60:120], humidity_sensor[60:120], label="sensor")
plt.plot(timestamp[60:120], first_order_output[0:60], 'o', markersize=3, label="first_order")
plt.plot(timestamp[60:120], second_order_output[0:60],'--', markersize=3, label="second_order")
plt.xlabel("Time / s")
plt.ylabel("Humidity / %")
plt.legend()
plt.title("True humidity(Input) and Sensor(Output)")
plt.show()


plt.Figure()
plt.plot(timestamp[-61:-1], humidity_true[-61:-1], label="Enviroment")
plt.plot(timestamp[-61:-1], humidity_sensor[-61:-1], label="sensor")
plt.plot(timestamp[-61:-1], first_order_output[-61:-1], 'o', markersize=3, label="first_order")
plt.plot(timestamp[-61:-1], second_order_output[-61:-1],'--', markersize=3, label="second_order")
plt.xlabel("Time / s")
plt.ylabel("Humidity / %")
plt.legend()
plt.title("True humidity(Input) and Sensor(Output)")
plt.show()
