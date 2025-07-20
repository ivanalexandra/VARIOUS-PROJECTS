# %matplotlib widget
# %matplotlib inline
# %matplotlib notebook

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

dataset = pd.read_csv(r'D:\ANUL3\Semestrul_2\Tehnici de optimizare\Dataseturi\Orange_Telecom.csv', usecols = ['total_day_minutes', 'total_day_calls', 'total_day_charge', 'total_eve_minutes', 'total_eve_calls', 'total_eve_charge', 'total_night_minutes', 'total_night_calls', 'total_night_charge', 'total_intl_minutes', 'total_intl_calls'])
dataset.corr()

def functia_regresie_plot(b, t):
    plt.figure(figsize=(12, 5))
    
    for i in range(1, 9):
        time_start = time.time()
        A = np.ones((b.size, 1))

        for g in range(1, i + 1):
            A = np.hstack((A, t**g))

        x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
        f_t = np.polyval(x[::-1], t)

        SSE = np.linalg.norm(np.dot(A, x) - b) ** 2
        MSE = SSE / b.size
        RMSE = MSE ** (1/2)
        MAE = np.mean(abs(np.dot(A, x) - b))
        R2 = 1 - (SSE / np.sum((b - np.mean(b)) ** 2))

        time_end = time.time()

        plt.subplot(2, 4, i)
        plt.plot(t, b, '.r')
        plt.plot(np.sort(t, axis=0), np.sort(f_t, axis=0))
        plt.title(f'Degree {i}')
        plt.grid(True)

        print(f'SSE: {SSE} | MSE: {MSE} | RMSE: {RMSE} | MAE: {MAE} | R2: {R2} | Time: {round(time_end - time_start, 10)}')

    plt.tight_layout()

b = dataset['total_intl_minutes'].to_numpy().reshape(-1, 1)

t = dataset['total_day_calls'].to_numpy().reshape(-1, 1)
functia_regresie_plot(b, t)

b = dataset['total_intl_calls'].to_numpy().reshape(-1, 1)

t = dataset['total_day_calls'].to_numpy().reshape(-1, 1)
functia_regresie_plot(b, t)