# %matplotlib widget
# %matplotlib inline
# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

dataset = pd.read_csv(r'D:\ANUL3\Semestrul_2\Tehnici de optimizare\Dataseturi\Orange_Telecom.csv', usecols = ['total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls'])
dataset.corr()

m = len(dataset.index)

def functia_regresie_plot(b, t):
    plt.figure(figsize=(12, 5))
    
    for grad in range(1, 9):
        time_start = time.time()
        
        poly = PolynomialFeatures(degree = grad)
        A = poly.fit_transform(t)
        poly.fit(A, b)

        lin = LinearRegression()
        lin.fit(A, b)

        b_pred = lin.predict(A)

        SSE = metrics.mean_squared_error(b, b_pred) * m
        MSE = metrics.mean_squared_error(b, b_pred)
        RMSE = np.sqrt(metrics.mean_squared_error(b, b_pred))
        MAE = metrics.mean_absolute_error(b, b_pred)
        R2 = metrics.r2_score(b, b_pred)

        time_end = time.time()

        plt.subplot(2, 4, grad)
        plt.plot(t, b, '.r')
        plt.plot(np.sort(t, axis=0), np.sort(b_pred, axis=0))
        plt.title(f'Degree {grad}')
        plt.grid(True)

        print(f'SSE: {SSE} | MSE: {MSE} | RMSE: {RMSE} | MAE: {MAE} | R2: {R2} | Time: {round(time_end - time_start, 5)}')

    plt.tight_layout()

b = dataset['total_intl_minutes'].to_numpy().reshape(-1, 1)

t = dataset['total_day_calls'].to_numpy().reshape(-1, 1)
functia_regresie_plot(b, t)

b = dataset['total_intl_calls'].to_numpy().reshape(-1, 1)

t = dataset['total_day_calls'].to_numpy().reshape(-1, 1)
functia_regresie_plot(b, t)