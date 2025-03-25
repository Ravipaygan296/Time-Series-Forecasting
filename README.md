# Time-Series-Forecasting# Time Series Forecasting using ARIMA

## Overview
This project demonstrates time series forecasting using the **ARIMA** model on the classic **Airline Passengers Dataset**. The dataset contains monthly totals of international airline passengers from 1949 to 1960.

## Features
- Load and visualize time series data
- Perform stationarity tests (ADF Test)
- Apply differencing to make the series stationary
- Train an **ARIMA (AutoRegressive Integrated Moving Average) model**
- Forecast future values and visualize the results
- Evaluate the model using MAE and RMSE

## Installation
Clone the repository and install the required dependencies:

```sh
git clone https://github.com/Ravipaygan296/Time-Series-Forecasting.git
cd Time-Series-Forecasting
pip install -r requirements.txt
```

## Usage
Run the Python script to perform forecasting:

```sh
python time_series_forecasting.py
```

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- sklearn

## Results
The ARIMA model provides future predictions based on past trends. The performance is evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

## License
This project is open-source under the MIT License.

---

Feel free to contribute or report issues!

