# XAUUSD Time Series Analysis

## Overview

This project involves the analysis and forecasting of historical XAUUSD (gold vs. US dollar) prices. The objective is to develop and evaluate various time series models to predict future price movements and understand the dynamics of gold prices relative to the US dollar.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, StatsModels, Prophet, pmdarima, TensorFlow (for LSTM), Joblib

## Project Structure

1. **Data Loading and Preparation**
   - Load historical XAUUSD data from a CSV file.
   - Convert date columns to datetime format and sort the data.

2. **Exploratory Data Analysis (EDA)**
   - Visualize historical close prices.
   - Perform time series decomposition to analyze trends, seasonality, and residuals.

3. **Statistical Testing**
   - Conduct Augmented Dickey-Fuller (ADF) test to check for stationarity.

4. **Time Series Modeling**
   - **ARIMA Model:** Fit an ARIMA model and forecast future values.
   - **Prophet Model:** Fit a Prophet model and generate forecasts.

5. **Model Evaluation and Validation**
   - Evaluate model performance using metrics such as MAE, MSE, and RMSE.
   - Perform residual analysis and cross-validation.

6. **Feature Engineering and LSTM**
   - Create lagged features and moving averages.
   - Prepare data and train an LSTM model for forecasting.

7. **Results and Visualization**
   - Compare model performance and visualize forecasts alongside historical data.
   - Save trained models for future use.

## Installation

To run this project, you need to have Python installed along with the required libraries. You can set up your environment using the following commands:

```bash
pip install pandas numpy matplotlib seaborn statsmodels prophet pmdarima tensorflow joblib
