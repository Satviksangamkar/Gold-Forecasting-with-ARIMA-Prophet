import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Define notebook cells with descriptions
cells = [
    # Markdown cell for Introduction
    nbf.v4.new_markdown_cell("# XAUUSD Time Series Analysis\n\n## Introduction\n\nThis notebook performs a comprehensive time series analysis on XAUUSD historical data. It covers data preprocessing, exploratory data analysis, time series modeling with ARIMA and Prophet, and a comparison of model performance."),
    
    # Code cell for importing libraries
    nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from prophet.diagnostics import cross_validation
from pmdarima import auto_arima
from statsmodels.stats.diagnostic import acorr_ljungbox
import joblib
    """),
    
    # Code cell for loading and preparing data
    nbf.v4.new_code_cell("""
# Load the CSV file with the correct delimiter (comma in this case)
data = pd.read_csv('XAUUSD_historical_data.csv', sep=',')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %H:%M')

# Sort data by date in ascending order
data = data.sort_values('Date')

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Display data info
data.info()
    """),
    
    # Code cell for plotting closing price
    nbf.v4.new_code_cell("""
# Plot the closing price over time to visualize the historical price movements
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.title('XAUUSD Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
    """),
    
    # Code cell for time series decomposition
    nbf.v4.new_code_cell("""
# Decompose the time series to analyze the trend, seasonality, and residuals
decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=30)
decomposition.plot()
plt.show()
    """),
    
    # Code cell for stationarity test
    nbf.v4.new_code_cell("""
# Perform Augmented Dickey-Fuller test to check if the time series is stationary
adf_test = adfuller(data['Close'])
print(f'ADF Statistic: {adf_test[0]}')
print(f'p-value: {adf_test[1]}')
    """),
    
    # Code cell for ARIMA model
    nbf.v4.new_code_cell("""
# Fit ARIMA model to the time series data
model = ARIMA(data['Close'], order=(5, 1, 0))  # Example order; tune as needed
arima_result = model.fit()

# Forecast future values using the fitted ARIMA model
forecast = arima_result.forecast(steps=10)
print(forecast)
    """),
    
    # Code cell for ARIMA forecast visualization
    nbf.v4.new_code_cell("""
# Forecast future values and visualize them alongside historical data
future_forecast = arima_result.forecast(steps=30)
print(future_forecast)

# Plot the historical data and the forecasted values
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Historical Data')
plt.plot(pd.date_range(start=data.index[-1], periods=31, freq='D')[1:], future_forecast, label='Forecast')
plt.title('XAUUSD Close Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
    """),
    
    # Code cell for model performance evaluation
    nbf.v4.new_code_cell("""
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate performance metrics for the ARIMA model
mae = mean_absolute_error(data['Close'], arima_result.fittedvalues)
mse = mean_squared_error(data['Close'], arima_result.fittedvalues)
rmse = mse ** 0.5

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
    """),
    
    # Code cell for saving and loading models
    nbf.v4.new_code_cell("""
import joblib

# Save the ARIMA model to a file
joblib.dump(arima_result, 'arima_model.pkl')

# To load the model later
# model = joblib.load('arima_model.pkl')
    """),
    
    # Code cell for Auto ARIMA
    nbf.v4.new_code_cell("""
# Automatically find the best ARIMA parameters using auto_arima
auto_model = auto_arima(data['Close'], seasonal=False, trace=True)
print(auto_model.summary())
    """),
    
    # Code cell for ARIMA residual analysis
    nbf.v4.new_code_cell("""
# Analyze residuals of the ARIMA model
residuals = arima_result.resid
plt.figure(figsize=(14, 7))
plt.plot(residuals)
plt.title('ARIMA Residuals')
plt.show()

# Perform Ljung-Box test to check if residuals are independent
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)
    """),
    
    # Code cell for Prophet model
    nbf.v4.new_code_cell("""
# Prepare data for the Prophet model
data_reset = data.reset_index()
data_reset = data_reset.rename(columns={'index': 'Date'})  # Rename index column to 'Date'
data_reset['Date'] = pd.to_datetime(data_reset['Date'])
prophet_data = data_reset[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# Initialize and fit the Prophet model
model = Prophet()
model.fit(prophet_data)

# Create a future dataframe and make predictions
future = model.make_future_dataframe(periods=30)  # Predict for 30 days into the future
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.show()
    """),
    
    # Code cell for Prophet model adjustment
    nbf.v4.new_code_cell("""
# Adjust the Prophet model with different hyperparameters
model = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=10.0)
model.fit(prophet_data)

# Check the length of the dataset
print(f"Number of data points: {len(prophet_data)}")

# Perform cross-validation on the Prophet model
from prophet.diagnostics import cross_validation

try:
    # Adjust horizon and period to match the dataset size
    cv_results = cross_validation(model, horizon='7 days', period='1 day')
    print(cv_results.head())
except ValueError as e:
    print(f"Error: {e}")
    """),
    
    # Code cell for feature engineering for LSTM
    nbf.v4.new_code_cell("""
def create_lagged_features(data, lags):
    # Create lagged features for time series forecasting
    df = data.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

# Create lagged features with a lag of 5 periods
lagged_data = create_lagged_features(data, lags=5)

from sklearn.preprocessing import MinMaxScaler

# Extract features and target variable
features = lagged_data.drop('Close', axis=1)
target = lagged_data['Close']

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Prepare data for LSTM
def prepare_lstm_data(features, target, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

# Set time steps for LSTM
time_steps = 5
X, y = prepare_lstm_data(features_scaled, target, time_steps)
    """),
    
    # Code cell for adding moving averages
    nbf.v4.new_code_cell("""
def add_moving_averages(data, windows):
    # Add moving averages to the dataset
    df = data.copy()
    for window in windows:
        df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

# Add moving averages with different window sizes
moving_avg_windows = [5, 10, 20]
ma_data = add_moving_averages(data, moving_avg_windows)

# Extract features and target variable
features = ma_data.drop('Close', axis=1)
target = ma_data['Close']

# Normalize features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Prepare data for LSTM
X, y = prepare_lstm_data(features_scaled, target, time_steps)
    """)
]

# Add cells to notebook
nb.cells.extend(cells)

# Save the notebook to a file
with open('XAUUSD_Time_Series_Analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
