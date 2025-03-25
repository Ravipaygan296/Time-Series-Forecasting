import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load sample dataset
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(data_url, parse_dates=['Month'], index_col='Month')
data.columns = ['Passengers']
print(data.head())

# Plot the time series
data.plot(figsize=(10, 5))
plt.title("Monthly Airline Passengers")
plt.show()

# Check stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    if result[1] <= 0.05:
        print("Data is stationary.")
    else:
        print("Data is non-stationary.")

test_stationarity(data['Passengers'])

# Differencing to make data stationary
data_diff = data.diff().dropna()
test_stationarity(data_diff['Passengers'])

# Fit ARIMA model
model = ARIMA(data['Passengers'], order=(1,1,1))  # (p,d,q) values can be tuned
model_fit = model.fit()
print(model_fit.summary())

# Forecast
forecast_steps = 10
forecast = model_fit.forecast(steps=forecast_steps)

# Plot forecast
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['Passengers'], label='Actual')
plt.plot(pd.date_range(data.index[-1], periods=forecast_steps, freq='M'), forecast, label='Forecast', color='red')
plt.legend()
plt.show()

# Evaluate model
y_true = data['Passengers'][-forecast_steps:]
y_pred = forecast[:len(y_true)]
print(f'MAE: {mean_absolute_error(y_true, y_pred)}')
print(f'RMSE: {np.sqrt(mean_squared_error(y_true, y_pred))}')
