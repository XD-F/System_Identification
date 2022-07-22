# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:29:13 2022
@author: Feng
"""
"""
Introduction:
This program is used to calculate the differential equation model for each order 
of the sensor (the differential equation model is obtained from the transfer 
function model of the sensor by Z-transformation)
The data used in this program are the step response data of the humidity sensor
in the experimental water tank.
Please download from the following path in I3's server:
    
path: i3-darwin:/mnt/data/MURON/WaterVaporObservation/RawData/20220217_GC
For more information on system identification, 
please refer to the following information:
    
Paper: Lacerda et al., (2020). SysIdentPy: A Python package for System 
Identification using NARMAX models. Journal of Open Source Software, 5(54), 2384, 
https://doi.org/10.21105/joss.02384
Python Package: sysidentpy https://sysidentpy.org/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.display_results import results

# Plot Setting
plt.rcParams['figure.dpi'] = 300
init_humi = 94.35
ramp_time = 3


# Import data
data = pd.read_csv("./07B2_20220217_GC_timeSyncData.csv", header=0, index_col=False)

# Humidity at steady state (300s)
termi_humi = np.mean(data["humi(SHT35)"].values[54000:54300])
# Transient Response （60s + ramp + 600s）
humidity_sensor = data["humi(SHT35)"].values[53385:54048]

humidity_true = np.append(np.linspace(init_humi, init_humi, 60),
                    np.linspace(init_humi, termi_humi, ramp_time))
humidity_true = np.append(humidity_true, 
                          np.linspace(termi_humi, termi_humi, 600))


time = np.linspace(0, len(humidity_sensor)-1, len(humidity_sensor))
time = time

# Use transient data to train model
x_train = np.array([humidity_true[60:-1]])
y_train = np.array([humidity_sensor[60:-1]])
# Transpose
x_train = x_train.T
y_train = y_train.T


### System Identification ###

# Model 1
basis_function = Polynomial(degree=1)
model_1 = FROLS(
    basis_function=basis_function,
    order_selection=True,
    n_info_values=10,
    extended_least_squares=False,
    ylag=1, xlag=1,
    info_criteria='aic',
    estimator='least_squares',
)

model_1.fit(X=x_train,y=y_train)
result_1 = model_1.predict(x_train, y_train)

r = pd.DataFrame(
results(
    model_1.final_model, model_1.theta, model_1.err,
    model_1.n_terms, err_precision=8, dtype='sci'
    ),
columns=['Regressors', 'Parameters', 'ERR'])
print(r)



# Model 2
basis_function = Polynomial(degree=1)
model_2 = FROLS(
    basis_function=basis_function,
    order_selection=True,
    n_info_values=10,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='aic',
    estimator='least_squares',
)

model_2.fit(X=x_train,y=y_train)
result_2 = model_2.predict(x_train, y_train)
r = pd.DataFrame(
results(
    model_1.final_model, model_1.theta, model_1.err,
    model_1.n_terms, err_precision=8, dtype='sci'
    ),
columns=['Regressors', 'Parameters', 'ERR'])
print(r)


# Model 3
basis_function = Polynomial(degree=3)
model_3 = FROLS(
    basis_function=basis_function,
    order_selection=True,
    n_info_values=15,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='aic',
    estimator='least_squares',
)

model_3.fit(X=x_train,y=y_train)
result_3 = model_3.predict(x_train, y_train)

r = pd.DataFrame(
results(
    model_3.final_model, model_3.theta, model_3.err,
    model_3.n_terms, err_precision=8, dtype='sci'
    ),
columns=['Regressors', 'Parameters', 'ERR'])
print(r)

xaxis = np.arange(1, model_3.n_info_values + 1)
plt.plot(xaxis, model_3.info_values)
plt.xlabel('n_terms')
plt.ylabel('AIC value')
plt.title('Akaike information criterion (AIC)')
plt.show()


# Model 4
basis_function = Polynomial(degree=2)
model_4 = FROLS(
    basis_function=basis_function,
    order_selection=True,
    n_info_values=10,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='aic',
    estimator='least_squares',
    n_terms=6
)

model_4.fit(X=x_train,y=y_train)
result_4 = model_4.predict(x_train, y_train)

r = pd.DataFrame(
results(
    model_4.final_model, model_4.theta, model_4.err,
    model_4.n_terms, err_precision=8, dtype='sci'
    ),
columns=['Regressors', 'Parameters', 'ERR'])
print(r)


# Model 5
basis_function = Polynomial(degree=4)
model_5 = FROLS(
    basis_function=basis_function,
    order_selection=True,
    n_info_values=15,
    extended_least_squares=False,
    ylag=3, xlag=3,
    info_criteria='aic',
    estimator='least_squares'
)

model_5.fit(X=x_train,y=y_train)
result_5 = model_5.predict(x_train, y_train)

r = pd.DataFrame(
results(
    model_5.final_model, model_5.theta, model_5.err,
    model_5.n_terms, err_precision=8, dtype='sci'
    ),
columns=['Regressors', 'Parameters', 'ERR'])
print(r)



# Model 6
basis_function = Polynomial(degree=2)
model_6 = FROLS(
    basis_function=basis_function,
    order_selection=True,
    n_info_values=10,
    extended_least_squares=False,
    ylag=2, xlag=2,
    info_criteria='aic',
    estimator='least_squares'
)

model_6.fit(X=x_train,y=y_train)
result_6 = model_6.predict(x_train, y_train)

r = pd.DataFrame(
results(
    model_6.final_model, model_6.theta, model_6.err,
    model_6.n_terms, err_precision=8, dtype='sci'
    ),
columns=['Regressors', 'Parameters', 'ERR'])
print(r)

### Plot ###

plt.Figure()
plt.plot(time[60:-1], humidity_true[60:-1], label="Enviroment (averaged)")
plt.plot(time[60:-1], humidity_sensor[60:-1], label="sensor")
plt.plot(time[61:-1], result_1[0:-1], label="model:1st, nonlinear:1st")
plt.plot(time[61:-1], result_2[0:-1], '--', markersize=3, label="model:2nd, nonlinear:1st")
plt.plot(time[61:-1], result_6[0:-1], '1', markersize=3, label="model:2nd, nonlinear:2nd")
plt.plot(time[61:-1], result_3[0:-1], '*', markersize=3, label="model:2nd, nonlinear:3rd")
plt.plot(time[61:-1], result_4[0:-1], 'o', markersize=3, label="model:2nd, nonlinear:2rd, AIC, nterm=6")
plt.plot(time[61:-1], result_5[0:-1], '^', markersize=3, label="model:3rd, nonlinear:4th")

plt.xlabel("Time / s")
plt.ylabel("Humidity / %")
plt.legend()
plt.title("True humidity(Input) and Sensor(Output)")
plt.show()




plt.Figure()
plt.plot(time[80:260], humidity_true[80:260], label="Enviroment (averaged)")
plt.plot(time[80:260], humidity_sensor[80:260], label="sensor")
plt.plot(time[80:260], result_1[20:200], label="model:1st, nonlinear:1st")
plt.plot(time[80:260], result_2[20:200], '--', markersize=3, label="model:2nd, nonlinear:1st")
plt.plot(time[80:260], result_6[20:200], '1', markersize=3, label="model:2nd, nonlinear:2nd")
plt.plot(time[80:260], result_3[20:200], '*', markersize=3, label="model:2nd, nonlinear:3rd")
plt.plot(time[80:260], result_4[20:200], 'o', markersize=3, label="model:2nd, nonlinear:2rd, AIC:nterm=6")
plt.plot(time[80:260], result_5[20:200], '^', markersize=3, label="model:3rd, nonlinear:4th")

plt.xlabel("Time / s")
plt.ylabel("Humidity / %")
plt.legend()
plt.title("True humidity(Input) and Sensor(Output)")
plt.show()


plt.Figure()
plt.plot(time[-200:-1], humidity_true[-200:-1], label="Enviroment (averaged)")
plt.plot(time[-200:-1], humidity_sensor[-200:-1], label="sensor")
plt.plot(time[-200:-1], result_1[-200:-1], label="model:1st, nonlinear:1st")
plt.plot(time[-200:-1], result_2[-200:-1], '--', markersize=3, label="model:2nd, nonlinear:1st")
plt.plot(time[-200:-1], result_6[-200:-1], '1', markersize=3, label="model:2nd, nonlinear:2nd")
plt.plot(time[-200:-1], result_3[-200:-1], '*', markersize=3, label="model:2nd, nonlinear:3rd")
plt.plot(time[-200:-1], result_4[-200:-1], 'o', markersize=3, label="model:2nd, nonlinear:2rd, AIC:nterm=6")
plt.plot(time[-200:-1], result_5[-200:-1], '^', markersize=3, label="model:3rd, nonlinear:4th")

plt.xlabel("Time / s")
plt.ylabel("Humidity / %")
plt.legend()
plt.title("True humidity(Input) and Sensor(Output)")
plt.show()

